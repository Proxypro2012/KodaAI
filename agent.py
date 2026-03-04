"""
AI Agent using OpenRouter
--------------------------
OpenRouter is OpenAI-API-compatible, so we use the openai SDK
and simply point base_url at https://openrouter.ai/api/v1.

Flow:
  1. User sends a message.
  2. Agent decides whether to call a tool or reply directly.
  3. If a tool is called, the result is fed back and the agent
     produces a final answer.
  4. Repeat until the user types 'exit'.
"""

import json
import inspect
import os
import sys
import hashlib
import subprocess
import re
import fnmatch
import time
import threading
import itertools
import shutil
import textwrap
import urllib.request
import urllib.parse
from pathlib import Path
from datetime import datetime
try:
    import scratchattach as scratch3
except ImportError:
    scratch3 = None  # pip install scratchattach
from openai import OpenAI

# ── ANSI colour palette ──────────────────────────────────────────────────────
_R   = "\033[0m"        # reset
_B   = "\033[1m"        # bold
_DIM = "\033[2m"        # dim
_CY  = "\033[38;5;117m" # sky-blue
_GR  = "\033[38;5;245m" # grey
_GN  = "\033[38;5;120m" # green
_YL  = "\033[38;5;220m" # yellow
_PU  = "\033[38;5;183m" # purple / lavender
_RE  = "\033[38;5;203m" # red
_WH  = "\033[38;5;255m" # bright white
_OR  = "\033[38;5;215m" # orange


class _Spinner:
    """Braille-dot spinner shown in a daemon thread while the LLM responds."""
    _FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, label: str = "Thinking"):
        self._label = label
        self._stop  = threading.Event()
        self._t     = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        for ch in itertools.cycle(self._FRAMES):
            if self._stop.is_set():
                break
            w = shutil.get_terminal_size().columns
            line = f"\r {_YL}{_B}{ch}{_R} {_GR}{self._label}…{_R}"
            sys.stdout.write(line)
            sys.stdout.flush()
            time.sleep(0.08)
        # Erase the spinner line
        sys.stdout.write("\r" + " " * shutil.get_terminal_size().columns + "\r")
        sys.stdout.flush()

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._t.join()


def _banner() -> None:
    cols = min(shutil.get_terminal_size().columns, 62)
    title = " ✦  Koda AI  ·  OpenRouter "
    pad   = cols - len(title) - 2          # 2 for the border chars
    left  = max(pad // 2, 0)
    right = max(pad - left, 0)
    bar   = f"{_CY}{_B}{'═' * cols}{_R}"
    row   = f"{_CY}{_B}║{_R}{' ' * left}{_PU}{_B}{title}{_R}{' ' * right}{_CY}{_B}║{_R}"
    print(f"\n{bar}")
    print(row)
    print(f"{bar}\n")


def _rule(char: str = "─") -> None:
    cols = min(shutil.get_terminal_size().columns, 62)
    print(f"{_GR}{char * cols}{_R}")


def _wrap_response(text: str) -> str:
    """Word-wrap the agent response to terminal width."""
    cols = min(shutil.get_terminal_size().columns - 8, 100)
    lines = []
    for paragraph in text.split("\n"):
        wrapped = textwrap.fill(paragraph, width=cols) if paragraph.strip() else ""
        lines.append(wrapped)
    return "\n".join(lines)


def _fmt_tool_result(text: str, limit: int = 200) -> str:
    flat = text.replace("\n", " ").strip()
    if len(flat) <= limit:
        return flat
    return flat[:limit] + f" {_GR}…(truncated){_R}"

def _sanitize_api_key(raw: str) -> str:
    key = (raw or "").strip().strip('"').strip("'")
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    return key.replace("\r", "").replace("\n", "").strip()

def _load_openrouter_api_key() -> tuple[str, str, str]:
    for var_name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "API_KEY"):
        key = _sanitize_api_key(os.getenv(var_name, ""))
        if key:
            return key, "env", var_name
    try:
        with open("api.key", "r", encoding="utf-8") as f:
            key = _sanitize_api_key(f.read())
            if key:
                return key, "file", "api.key"
    except Exception:
        pass
    return "", "none", ""


def get_openrouter_key_diagnostics() -> dict:
    value = OPENROUTER_API_KEY or ""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12] if value else ""
    return {
        "has_key": bool(value),
        "source": OPENROUTER_API_KEY_SOURCE,
        "source_name": OPENROUTER_API_KEY_SOURCE_NAME,
        "length": len(value),
        "sha256_12": digest,
    }


OPENROUTER_API_KEY, OPENROUTER_API_KEY_SOURCE, OPENROUTER_API_KEY_SOURCE_NAME = _load_openrouter_api_key()
# Models tried in order — first one to succeed is used for the whole request.
# If a model is overloaded, rate-limited, or returns an error, the next one is tried.
MODELS: list[str] = [
    "arcee-ai/trinity-large-preview:free",
    "openai/gpt-oss-120b:free",
    "google/gemini-2.0-flash-exp:free",  # final backstop
]

# Scratch session — populated by scratch_login tool
_scratch_session = None
_scratch_login_username: str | None = None
_scratch_login_password: str | None = None
# Cache for project JSON to avoid flooding the LLM context
_project_json_cache: dict[str, dict] = {}


def _default_project_json() -> dict:
    return {
        "targets": [
            {
                "isStage": True,
                "name": "Stage",
                "variables": {},
                "lists": {},
                "broadcasts": {},
                "blocks": {},
                "comments": {},
                "currentCostume": 0,
                "costumes": [
                    {
                        "name": "backdrop1",
                        "dataFormat": "svg",
                        "assetId": "cd21514d0531fdffb22204e0ec5ed84a",
                        "md5ext": "cd21514d0531fdffb22204e0ec5ed84a.svg",
                        "rotationCenterX": 240,
                        "rotationCenterY": 180,
                    }
                ],
                "sounds": [],
                "volume": 100,
                "layerOrder": 0,
                "tempo": 60,
                "videoTransparency": 50,
                "videoState": "on",
                "textToSpeechLanguage": None,
            },
            {
                "isStage": False,
                "name": "Sprite1",
                "variables": {},
                "lists": {},
                "broadcasts": {},
                "blocks": {},
                "comments": {},
                "currentCostume": 0,
                "costumes": [
                    {
                        "name": "costume1",
                        "bitmapResolution": 1,
                        "dataFormat": "svg",
                        "assetId": "bcf454acf82e4504149f7ffe07081dbc",
                        "md5ext": "bcf454acf82e4504149f7ffe07081dbc.svg",
                        "rotationCenterX": 48,
                        "rotationCenterY": 50,
                    },
                    {
                        "name": "costume2",
                        "bitmapResolution": 1,
                        "dataFormat": "svg",
                        "assetId": "0fb9be3e8397c983338cb71dc84d0b25",
                        "md5ext": "0fb9be3e8397c983338cb71dc84d0b25.svg",
                        "rotationCenterX": 46,
                        "rotationCenterY": 53,
                    },
                ],
                "sounds": [
                    {
                        "name": "Meow",
                        "assetId": "83a9787d4cb6f3b7632b4ddfebf74367",
                        "dataFormat": "wav",
                        "format": "",
                        "rate": 48000,
                        "sampleCount": 40681,
                        "md5ext": "83a9787d4cb6f3b7632b4ddfebf74367.wav",
                    }
                ],
                "volume": 100,
                "visible": True,
                "x": 0,
                "y": 0,
                "size": 100,
                "direction": 90,
                "draggable": False,
                "rotationStyle": "all around",
                "layerOrder": 1,
            },
        ],
        "monitors": [],
        "extensions": [],
        "meta": {
            "semver": "3.0.0",
            "vm": "0.2.0",
            "agent": "Koda AI",
        },
    }

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

# ---------------------------------------------------------------------------
# @tool decorator — register a function as an agent tool automatically
# ---------------------------------------------------------------------------
_tool_registry: dict[str, callable] = {}
tools: list[dict] = []

_PY_TO_JSON_TYPE = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
}

def tool(description: str, param_descriptions: dict[str, str] | None = None):
    """
    Decorator that registers a function as a callable tool for the agent.

    Usage:
        @tool("What this function does", {"param": "what it is"})
        def my_function(param: str) -> str:
            ...

    The JSON schema is built automatically from the function's type hints.
    """
    def decorator(fn):
        sig = inspect.signature(fn)
        properties = {}
        required = []

        for name, param in sig.parameters.items():
            annotation = param.annotation
            type_name = annotation.__name__ if annotation != inspect.Parameter.empty else "string"
            json_type = _PY_TO_JSON_TYPE.get(type_name, "string")
            prop = {"type": json_type}
            if param_descriptions and name in param_descriptions:
                prop["description"] = param_descriptions[name]
            properties[name] = prop
            if param.default is inspect.Parameter.empty:
                required.append(name)

        tools.append({
            "type": "function",
            "function": {
                "name": fn.__name__,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        })
        _tool_registry[fn.__name__] = fn
        return fn
    return decorator


def dispatch_tool(name: str, arguments: str) -> str:
    """Look up and call a registered tool by name."""
    if name not in _tool_registry:
        return f"Unknown tool: {name}"
    try:
        args = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        args = {}

    # Check for missing required arguments and return a clear error
    # so the model can retry with the correct parameters
    fn = _tool_registry[name]
    sig = inspect.signature(fn)
    missing = [
        p for p, param in sig.parameters.items()
        if param.default is inspect.Parameter.empty and p not in args
    ]
    if missing:
        return f"Tool '{name}' is missing required arguments: {missing}. Please call it again with all required parameters."

    try:
        return str(fn(**args))
    except Exception as e:
        return f"Tool '{name}' raised an error: {e}"


# ---------------------------------------------------------------------------
# Tool definitions — just decorate a normal function, no boilerplate needed
# ---------------------------------------------------------------------------
@tool(
    "Return the current real-time weather for a given city using a live API.",
    {"city": "City name, e.g. 'London'"},
)
def get_weather(city: str) -> str:
    try:
        # Step 1: geocode the city name → lat/lon (Open-Meteo geocoding, free)
        geo_url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode({
            "name": city,
            "count": 1,
            "language": "en",
            "format": "json",
        })
        with urllib.request.urlopen(geo_url, timeout=5) as resp:
            geo = json.loads(resp.read())

        if not geo.get("results"):
            return f"Could not find location: '{city}'."

        result = geo["results"][0]
        lat, lon = result["latitude"], result["longitude"]
        name = result.get("name", city)
        country = result.get("country", "")

        # Step 2: fetch current weather (Open-Meteo weather, free)
        weather_url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode({
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,weathercode,windspeed_10m",
            "temperature_unit": "celsius",
            "windspeed_unit": "kmh",
        })
        with urllib.request.urlopen(weather_url, timeout=5) as resp:
            weather = json.loads(resp.read())

        current = weather["current"]
        temp = current["temperature_2m"]
        wind = current["windspeed_10m"]

        # WMO weather codes → human-readable description
        wmo_codes = {
            0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
            45: "foggy", 48: "icy fog", 51: "light drizzle", 53: "drizzle",
            55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
            71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers",
            95: "thunderstorm", 99: "thunderstorm with hail",
        }
        condition = wmo_codes.get(current["weathercode"], "unknown conditions")

        return f"{name}, {country}: {temp}°C, {condition}, wind {wind} km/h"

    except Exception as e:
        return f"Error fetching weather: {e}"


@tool(
    "Evaluate a simple arithmetic expression and return the result.",
    {"expression": "A Python-safe arithmetic expression, e.g. '3 * (4 + 2)'"},
)
def calculate(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"



@tool(
    "Retrieve facts about Ishaan Kumble. ALWAYS call this tool whenever the user mentions or asks anything about Ishaan Kumble.",
    {"topic": "The specific topic about Ishaan Kumble (e.g. 'coding', 'football'). Use 'general' if no specific topic is mentioned."}
)
def talk_about_ishaan(topic: str = "general") -> str:
    facts = {
        "general": (
            "Ishaan Kumble is a passionate developer and AI enthusiast who loves building "
            "intelligent agents. He is also a huge football fan and enjoys working on "
            "cutting-edge technology projects."
        ),
        "coding": "Ishaan Kumble is a passionate developer who builds AI agents and loves experimenting with LLMs.",
        "football": "Ishaan Kumble is a huge football fan who follows the sport closely.",
        "ai": "Ishaan Kumble is deeply interested in AI, particularly in building agents with tool-calling capabilities.",
    }
    return facts.get(topic.lower(), facts["general"])


@tool(
    "Retrieve facts about Ishaan S. ALWAYS call this tool whenever the user mentions or asks anything about Ishaan S.",
    {"topic": "The specific topic about Ishaan S (e.g. '', 'football'). Use 'general' if no specific topic is mentioned."}
)
def talk_about_ishaan_subramanian(topic: str = "general") -> str:
    facts = {
        "general": (
            "Ishaan S is a very smart and intelligent kid who is smart even without going to RSM. "
        ),
    }
    return facts.get(topic.lower(), facts["general"])



@tool(
    "Retrieve facts about Aayush Saha. ALWAYS call this tool whenever the user mentions or asks anything about Aayush Saha.",
    {"topic": "The specific topic about Aayush Saha (e.g. '', 'football'). Use 'general' if no specific topic is mentioned."}
)
def talk_about_aayush_saha(topic: str = "general") -> str:
    facts = {
        "general": (
            "Aayush Saha is a dumb bum who is dumb as dirt and he is the reason gravity exists, because he is so massive that the universe gave up and started to bend around him."
        ),
    }
    return facts.get(topic.lower(), facts["general"])


@tool(
    "Retrieve facts about Hridhveer Khurana. ALWAYS call this tool whenever the user mentions or asks anything about Hridhveer Khurana.",
    {"topic": "The specific topic about Hridhveer Khurana (e.g. '', 'football'). Use 'general' if no specific topic is mentioned."}
)
def talk_about_hridhveer_khurana(topic: str = "general") -> str:
    facts = {
        "general": (
            "Hridhveer Khurana is a charismatic kid who is loved by everyone and he is incredibly smart as well."
        ),
    }
    return facts.get(topic.lower(), facts["general"])



@tool(
    "Retrieve facts about Leonard fan. ALWAYS call this tool whenever the user mentions or asks anything about Leonard fan.",
    {"topic": "The specific topic about Leonard fan (e.g. '', 'football'). Use 'general' if no specific topic is mentioned."}
)
def talk_about_leonard_fan(topic: str = "general") -> str:
    facts = {
        "general": (
            "Leonard fan is a charismatic kid who is loved by everyone and he is incredibly smart as well."
        ),
    }
    return facts.get(topic.lower(), facts["general"])



@tool(
    "Retrieve facts about Ahaan Emmydisetty. ALWAYS call this tool whenever the user mentions or asks anything about Ahaan Emmydisetty.",
    {"topic": "The specific topic about Ahaan Emmydisetty (e.g. '', 'football'). Use 'general' if no specific topic is mentioned."}
)
def talk_about_ahaan_emmydisetty(topic: str = "general") -> str:
    facts = {
        "general": (
            "Ahaan Emmydisetty is a short king who is trying to pull the freshman huzz. Nah, I'm joking, he is actually a nice kid who is very smart as well. "
        ),
    }
    return facts.get(topic.lower(), facts["general"])


@tool(
    "Retrieve facts about Vaishnav Mohan. ALWAYS call this tool whenever the user mentions or asks anything about Vaishnav Mohan.",
    {"topic": "The specific topic about Vaishnav Mohan (e.g. '', 'football'). Use 'general' if no specific topic is mentioned."}
)
def talk_about_vaishnav_mohan(topic: str = "general") -> str:
    facts = {
        "general": (
            "Vaishnav Mohan is the goat of speech and debate and has the ability to convince people as well as that one guy who got rejected from art school long ago. "
        ),
    }
    return facts.get(topic.lower(), facts["general"])

@tool(
    "Retrieve facts about Saket Gahlot. ALWAYS call this tool whenever the user mentions or asks anything about Saket Gahlot.",
    {"topic": "The specific topic about Saket Gahlot (e.g. '', 'football'). Use 'general' if no specific topic is mentioned."}
)
def talk_about_saket_gahlot(topic: str = "general") -> str:
    facts = {
        "general": (
            "Saket Gahlot is a chill and very smart kid who is aura farming."
        ),
    }
    return facts.get(topic.lower(), facts["general"])

@tool(
    "Retrieve facts about Zuhaib Syed. ALWAYS call this tool whenever the user mentions or asks anything about Zuhaib Syed.",
    {"topic": "The specific topic about Zuhaib Syed (e.g. '', 'football'). Use 'general' if no specific topic is mentioned."}
)
def talk_about_zuhaib_syed(topic: str = "general") -> str:
    facts = {
        "general": (
            "Zuhaib Syed is a bomber who yells 'allahu akbar' a lot and is a very smart kid as well."
        ),
    }
    return facts.get(topic.lower(), facts["general"])

@tool(
    "Retrieve facts about Ved Patel. ALWAYS call this tool whenever the user mentions or asks anything about Ved Patel.",
    {"topic": "The specific topic about Ved Patel (e.g. '', 'football'). Use 'general' if no specific topic is mentioned."}
)
def talk_about_ved_patel(topic: str = "general") -> str:
    facts = {
        "general": (
            "Ved Patel is just stupid. Sorry, but he is."
        ),
    }
    return facts.get(topic.lower(), facts["general"])

@tool(
    "Retrieve facts about Adhvik Arving. ALWAYS call this tool whenever the user mentions or asks anything about Adhvik Arving.",
    {"topic": "The specific topic about Adhvik Arving (e.g. '', 'football'). Use 'general' if no specific topic is mentioned."}
)
def talk_about_adhvik_arving(topic: str = "general") -> str:
    facts = {
        "general": (
            "Adhvik Arving is a pure zest fest kid who is trying to pull the freshman huzz but secretly tries to pull the 8th grade broes."
        ),
    }
    return facts.get(topic.lower(), facts["general"])

@tool(
    "Retrieve facts about Rishab Reddy Paili. ALWAYS call this tool whenever the user mentions or asks anything about Rishab Reddy Paili.",
    {"topic": "The specific topic about Rishab Reddy Paili (e.g. '', 'football'). Use 'general' if no specific topic is mentioned."}
)
def talk_about_rishab_reddy_paili(topic: str = "general") -> str:
    facts = {
        "general": (
            "Rishab Reddy Paili is a pretty smart kid who crashes out on Sanjit 24/7 on the school bus."
        ),
    }
    return facts.get(topic.lower(), facts["general"])

@tool(
    "Retrieve facts about Chris Hanies. ALWAYS call this tool whenever the user mentions or asks anything about Chris Hanies.",
    {"topic": "The specific topic about Chris Hanies (e.g. '', 'football'). Use 'general' if no specific topic is mentioned."}
)
def talk_about_chris_hanies(topic: str = "general") -> str:
    facts = {
        "general": (
            "Chris Hanies is chill. Just chill."
        ),
    }
    return facts.get(topic.lower(), facts["general"])

@tool(
    "Retrieve facts about Eshaan Vodhiparthi. ALWAYS call this tool whenever the user mentions or asks anything about Eshaan Vodhiparthi.",
    {"topic": "The specific topic about Eshaan Vodhiparthi (e.g. '', 'football'). Use 'general' if no specific topic is mentioned."}
)
def talk_about_eshaan_vodhiparthi(topic: str = "general") -> str:
    facts = {
        "general": (
            "Eshaan Vodhiparthi is a very loyal kid who tries to be friends with everyone but everyone likes him. Known as 'eggshell eshaan'."
        ),
    }
    return facts.get(topic.lower(), facts["general"])

@tool(
    "Retrieve facts about Srihan Anand. ALWAYS call this tool whenever the user mentions or asks anything about Srihan Anand.",
    {"topic": "The specific topic about Srihan Anand (e.g. '', 'football'). Use 'general' if no specific topic is mentioned."}
)
def talk_about_srihan_anand(topic: str = "general") -> str:
    facts = {
        "general": (
            "Srihan Anand is Kabir's best friend who is good at basketball and smart."
        ),
    }
    return facts.get(topic.lower(), facts["general"])

@tool(
    "Retrieve facts about Aarush Patel. ALWAYS call this tool whenever the user mentions or asks anything about Aarush Patel.",
    {"topic": "The specific topic about Aarush Patel (e.g. '', 'football'). Use 'general' if no specific topic is mentioned."}
)
def talk_about_aarush_patel(topic: str = "general") -> str:
    facts = {
        "general": (
            "Aarush Patel is fricking annoying but he is also smart."
        ),
    }
    return facts.get(topic.lower(), facts["general"])

@tool(
    "Retrieve facts about Krishna Suri. ALWAYS call this tool whenever the user mentions or asks anything about Krishna Suri.",
    {"topic": "The specific topic about Krishna Suri (e.g. '', 'football'). Use 'general' if no specific topic is mentioned."}
)
def talk_about_krishna_suri(topic: str = "general") -> str:
    facts = {
        "general": (
            "Krishna Suri is the most annoying kid in the world. Thinks he's tough because he pulled the chuzz."
        ),
    }
    return facts.get(topic.lower(), facts["general"])

@tool(
    "Retrieve facts about Zareyab Ahmed. ALWAYS call this tool whenever the user mentions or asks anything about Zareyab Ahmed.",
    {"topic": "The specific topic about Zareyab Ahmed (e.g. '', 'football'). Use 'general' if no specific topic is mentioned."}
)
def talk_about_zareyab_ahmed(topic: str = "general") -> str:
    facts = {
        "general": (
            "Zareyab Ahmed is a very chill dude."
        ),
    }
    return facts.get(topic.lower(), facts["general"])



# ---------------------------------------------------------------------------
# Scratch tools (via scratchattach — full API coverage)
# Docs: https://github.com/TimMcCool/scratchattach/wiki/Documentation
# ---------------------------------------------------------------------------
def _require_scratch() -> str | None:
    if scratch3 is None:
        return "scratchattach is not installed. Run: pip install scratchattach"
    if _scratch_session is None:
        return "Not logged in to Scratch. Use the scratch_login tool first."
    return None

def _project_summary(p) -> dict:
    return {
        "id": getattr(p, "id", ""),
        "title": getattr(p, "title", ""),
        "author": getattr(p, "author_name", ""),
        "loves": getattr(p, "loves", 0),
        "favorites": getattr(p, "favorites", 0),
        "views": getattr(p, "views", 0),
        "url": getattr(p, "url", ""),
    }


def _scratch_project_url(project_id: str) -> str:
    pid = str(project_id).strip()
    return f"https://scratch.mit.edu/projects/{pid}"


def _scratch_forbidden_hint(project_id: str) -> str:
    url = _scratch_project_url(project_id)
    return (
        f"Permission denied by Scratch for project {project_id}. "
        f"Project URL: {url} . "
        "This usually means the logged-in account cannot edit that project in this session. "
        "Open the URL, verify ownership/login, or remix into a project you own and retry."
    )


def _is_scratch_forbidden_error(exc: Exception) -> bool:
    txt = str(exc).lower()
    return "forbidden" in txt or "not allowed to perform this action" in txt


def _scratch_relogin() -> tuple[bool, str]:
    global _scratch_session
    if scratch3 is None:
        return False, "scratchattach is not installed."
    if not _scratch_login_username or not _scratch_login_password:
        return False, "No stored Scratch credentials for auto-relogin."
    try:
        import warnings
        warnings.filterwarnings("ignore", category=scratch3.LoginDataWarning)
        _scratch_session = scratch3.login(_scratch_login_username, _scratch_login_password)
        return True, f"Re-logged in as '{_scratch_login_username}'."
    except Exception as e:
        return False, f"Auto-relogin failed: {e}"


def _run_scratch_write_action(action, project_id: str) -> tuple[object | None, str | None]:
    try:
        return action(), None
    except Exception as e:
        if _is_scratch_forbidden_error(e):
            ok, _msg = _scratch_relogin()
            if ok:
                try:
                    return action(), None
                except Exception as e2:
                    if _is_scratch_forbidden_error(e2):
                        return None, _scratch_forbidden_hint(project_id)
                    return None, f"Error: {e2}"
            return None, _scratch_forbidden_hint(project_id)
        return None, f"Error: {e}"

# ── Authentication ──────────────────────────────────────────────────────────

@tool(
    "Log in to Scratch. Must be called before any Scratch tools that require authentication.",
    {"username": "Scratch username", "password": "Scratch password"},
)
def scratch_login(username: str, password: str) -> str:
    global _scratch_session, _scratch_login_username, _scratch_login_password
    if scratch3 is None:
        return "scratchattach is not installed. Run: pip install scratchattach"
    try:
        import warnings
        warnings.filterwarnings("ignore", category=scratch3.LoginDataWarning)
        _scratch_session = scratch3.login(username, password)
        _scratch_login_username = username
        _scratch_login_password = password
        return f"Logged in as '{username}'."
    except Exception as e:
        return f"Login failed: {e}"

# ── Session / Account ───────────────────────────────────────────────────────

@tool("Get your Scratch messages (notifications).", {"limit": "Max number of messages to return (default 10)"})
def scratch_get_messages(limit: int = 10) -> str:
    err = _require_scratch()
    if err: return err
    try:
        msgs = _scratch_session.messages(limit=limit)
        return json.dumps([
            {"type": getattr(m, "type", ""), "actor": getattr(m, "actor_username", ""), "time": str(getattr(m, "datetime_created", ""))}
            for m in msgs
        ], indent=2)
    except Exception as e:
        return f"Error: {e}"

@tool("Get your unread Scratch message count.", {})
def scratch_message_count() -> str:
    err = _require_scratch()
    if err: return err
    try:
        return str(_scratch_session.get_message_count())
    except Exception as e:
        return f"Error: {e}"

@tool("Mark all your Scratch messages as read.", {})
def scratch_clear_messages() -> str:
    err = _require_scratch()
    if err: return err
    try:
        _scratch_session.clear_messages()
        return "Messages marked as read."
    except Exception as e:
        return f"Error: {e}"

@tool(
    "Get projects from your Scratch 'My Stuff' page.",
    {
        "filter_arg": "Filter: 'all', 'shared', 'unshared', or 'trashed'",
        "sort_by": "Sort by: '' (last modified), 'view_count', 'love_count', 'title'",
    },
)
def scratch_my_projects(filter_arg: str = "all", sort_by: str = "") -> str:
    err = _require_scratch()
    if err: return err
    try:
        projects = _scratch_session.mystuff_projects(filter_arg, page=1, sort_by=sort_by)
        return json.dumps([{"id": p.id, "title": getattr(p, "title", "")} for p in projects], indent=2)
    except Exception as e:
        return f"Error: {e}"

# ── Users ───────────────────────────────────────────────────────────────────

@tool(
    "Get public info about a Scratch user.",
    {"username": "Scratch username to look up"},
)
def scratch_get_user(username: str) -> str:
    err = _require_scratch()
    if err: return err
    try:
        u = _scratch_session.connect_user(username)
        return json.dumps({
            "username": u.username,
            "about_me": getattr(u, "about_me", ""),
            "wiwo": getattr(u, "wiwo", ""),
            "country": getattr(u, "country", ""),
            "followers": u.follower_count(),
            "following": u.following_count(),
            "projects": u.project_count(),
            "joined": str(getattr(u, "join_date", "")),
            "message_count": u.message_count(),
        }, indent=2)
    except Exception as e:
        return f"Error: {e}"

@tool("Follow a Scratch user.", {"username": "Username to follow"})
def scratch_follow_user(username: str) -> str:
    err = _require_scratch()
    if err: return err
    try:
        _scratch_session.connect_user(username).follow()
        return f"Now following '{username}'."
    except Exception as e:
        return f"Error: {e}"

@tool("Unfollow a Scratch user.", {"username": "Username to unfollow"})
def scratch_unfollow_user(username: str) -> str:
    err = _require_scratch()
    if err: return err
    try:
        _scratch_session.connect_user(username).unfollow()
        return f"Unfollowed '{username}'."
    except Exception as e:
        return f"Error: {e}"

@tool("Post a comment on a Scratch user's profile.", {"username": "Username to comment on", "comment": "Comment text"})
def scratch_comment_on_profile(username: str, comment: str) -> str:
    err = _require_scratch()
    if err: return err
    try:
        _scratch_session.connect_user(username).post_comment(comment)
        return f"Comment posted on {username}'s profile."
    except Exception as e:
        return f"Error: {e}"

@tool("Get the projects shared by a Scratch user.", {"username": "Scratch username", "limit": "Max results (default 20)"})
def scratch_get_user_projects(username: str, limit: int = 20) -> str:
    err = _require_scratch()
    if err: return err
    try:
        projects = list(_scratch_session.connect_user(username).projects(limit=limit))
        return json.dumps([{"id": p.id, "title": getattr(p, "title", "")} for p in projects], indent=2)
    except Exception as e:
        return f"Error: {e}"

@tool("Get a Scratch user's followers.", {"username": "Scratch username", "limit": "Max results (default 20)"})
def scratch_get_followers(username: str, limit: int = 20) -> str:
    err = _require_scratch()
    if err: return err
    try:
        names = _scratch_session.connect_user(username).follower_names(limit=limit)
        return json.dumps(names, indent=2)
    except Exception as e:
        return f"Error: {e}"

@tool("Get users a Scratch user is following.", {"username": "Scratch username", "limit": "Max results (default 20)"})
def scratch_get_following(username: str, limit: int = 20) -> str:
    err = _require_scratch()
    if err: return err
    try:
        names = _scratch_session.connect_user(username).following_names(limit=limit)
        return json.dumps(names, indent=2)
    except Exception as e:
        return f"Error: {e}"

@tool("Update the 'About me' section on the logged-in account's profile.", {"text": "New bio text"})
def scratch_set_bio(text: str) -> str:
    err = _require_scratch()
    if err: return err
    try:
        _scratch_session.connect_linked_user().set_bio(text)
        return "About me updated."
    except Exception as e:
        return f"Error: {e}"

@tool("Update the 'What I'm working on' section on the logged-in account's profile.", {"text": "New WIWO text"})
def scratch_set_wiwo(text: str) -> str:
    err = _require_scratch()
    if err: return err
    try:
        _scratch_session.connect_linked_user().set_wiwo(text)
        return "What I'm working on updated."
    except Exception as e:
        return f"Error: {e}"

# ── Projects ────────────────────────────────────────────────────────────────

@tool("Get info about a Scratch project.", {"project_id": "Numeric Scratch project ID"})
def scratch_get_project(project_id: str) -> str:
    err = _require_scratch()
    if err: return err
    try:
        p = _scratch_session.connect_project(project_id)
        return json.dumps(_project_summary(p), indent=2)
    except Exception as e:
        return f"Error: {e}"

@tool(
    "Get and cache the JSON of a Scratch project. Always call this before editing a project. Returns a sprite summary — the full JSON is stored internally.",
    {"project_id": "Numeric Scratch project ID"},
)
def scratch_get_project_json(project_id: str) -> str:
    err = _require_scratch()
    if err: return err
    try:
        project = _scratch_session.connect_project(project_id)
        data = project.raw_json()          # returns a dict per the docs
        if isinstance(data, str):
            data = json.loads(data)
        _project_json_cache[str(project_id)] = data
        targets = data.get("targets", [])
        url = _scratch_project_url(project_id)
        return json.dumps({
            "project_id": project_id,
            "url": url,
            "sprite_count": len(targets),
            "sprites": [{"name": t["name"], "block_count": len(t.get("blocks", {}))} for t in targets],
            "note": "Full JSON cached. Use scratch_build_script, scratch_add_say_block, or scratch_set_project_json to make changes.",
        }, indent=2)
    except Exception as e:
        return f"Error: {e}"

@tool(
    "Upload a modified project JSON back to Scratch. Pass the full JSON as a string. Call scratch_get_project_json first.",
    {"project_id": "Numeric Scratch project ID", "new_json": "Full project JSON as a string"},
)
def scratch_set_project_json(project_id: str, new_json: str) -> str:
    err = _require_scratch()
    if err: return err
    try:
        data = json.loads(new_json)
        project = _scratch_session.connect_project(project_id)
        _result, err_msg = _run_scratch_write_action(lambda: project.set_json(data), str(project_id))
        if err_msg:
            return err_msg
        _project_json_cache[str(project_id)] = data
        return f"Project {project_id} updated successfully. URL: {_scratch_project_url(project_id)}"
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"
    except Exception as e:
        if _is_scratch_forbidden_error(e):
            return _scratch_forbidden_hint(project_id)
        return f"Error: {e}"

@tool(
    "Create a brand-new Scratch project.",
    {"title": "Title for the new project"},
)
def scratch_create_project(title: str) -> str:
    err = _require_scratch()
    if err: return err
    clean_title = (title or "Untitled Project").strip()
    if not clean_title:
        clean_title = "Untitled Project"

    def _create_via_api(project_json: dict | None = None, parent_id: str | None = None):
        # Bypass scratchattach Session.create_project because current library
        # implementation can throw "list index out of range" in its internal
        # ratelimit helper when cache is empty.
        try:
            import scratchattach.site.session as _sa_session_mod
            req = _sa_session_mod.requests
        except Exception:
            return None, "Scratch API module import failed"

        params = {
            "is_remix": "1" if parent_id else "0",
            "original_id": parent_id,
            "title": clean_title,
        }
        payload = project_json if project_json is not None else _default_project_json()
        try:
            resp = req.post(
                "https://projects.scratch.mit.edu/",
                params=params,
                cookies=getattr(_scratch_session, "_cookies", None),
                headers=getattr(_scratch_session, "_headers", None),
                json=payload,
            )
            data = resp.json()
            pid = None
            if isinstance(data, dict):
                pid = data.get("content-name") or data.get("id") or data.get("project_id")
            if not pid:
                return None, f"Scratch API did not return project id: {str(data)[:240]}"
            try:
                project = _scratch_session.connect_project(str(pid))
                return project, ""
            except Exception:
                # still return id if connect is temporarily unavailable
                return type("_P", (), {"id": str(pid), "title": clean_title, "url": f"https://scratch.mit.edu/projects/{pid}"})(), ""
        except Exception as api_err:
            return None, str(api_err)

    def _project_info(p, note: str = "") -> str:
        pid = getattr(p, "id", None)
        ptitle = getattr(p, "title", clean_title)
        url = getattr(p, "url", "") or (_scratch_project_url(str(pid)) if pid else "")
        payload = {"id": pid, "title": ptitle, "url": url}
        if note:
            payload["note"] = note
        return json.dumps(payload, indent=2)

    # Attempt 1: direct API create with built-in template
    project, first_msg = _create_via_api(project_json=_default_project_json())
    if project:
        return _project_info(project)

    # Session may be stale; refresh login once and retry create.
    if "forbidden" in str(first_msg).lower() or "not allowed" in str(first_msg).lower():
        relog_ok, _relog_msg = _scratch_relogin()
        if relog_ok:
            project, first_msg = _create_via_api(project_json=_default_project_json())
            if project:
                return _project_info(project, note="Created after refreshing Scratch session.")

    # Attempt 2: if scratchattach hits internal list-index issues, retry using a
    # known-good template JSON from one of the user's existing projects.
    try:
        projects = _scratch_session.mystuff_projects("all", page=1)
        if projects:
            template_id = str(getattr(projects[0], "id", ""))
            if template_id:
                template_json = _scratch_session.connect_project(template_id).raw_json()
                project, second_msg = _create_via_api(project_json=template_json)
                if project:
                    return _project_info(project, note=f"Created using template JSON from project {template_id} due to create endpoint instability.")
    except Exception:
        pass

    # Attempt 3: remix-based create against a known global project ID.
    # This can work on accounts where plain creation endpoint is unstable.
    remix_source = "10015059"  # default Scratch Cat starter project
    project, third_msg = _create_via_api(project_json=_default_project_json(), parent_id=remix_source)
    if project:
        return _project_info(project, note=f"Created via remix fallback from project {remix_source}.")

    # Attempt 4: Sometimes Scratch creates the project but still returns a forbidden/write error.
    # Try to find a project with the requested title in the user's My Stuff and return it.
    try:
        existing = _scratch_session.mystuff_projects("all", page=1)
        for p in existing:
            if str(getattr(p, "title", "")).strip().lower() == clean_title.lower():
                pid = str(getattr(p, "id", ""))
                if pid:
                    return json.dumps({
                        "id": pid,
                        "title": getattr(p, "title", clean_title),
                        "url": _scratch_project_url(pid),
                        "note": "Project with the requested title already exists in your account. Returning that URL.",
                    }, indent=2)
    except Exception:
        pass

    return (
        f"Error: Could not create project '{clean_title}'. "
        f"Initial create failed with: {first_msg}. "
        f"Template fallback also failed with: {third_msg}. "
        "Try again in a few seconds, run scratch_my_projects to check whether the project was created anyway, "
        "or create one project manually on Scratch first and retry."
    )

@tool("Share a Scratch project.", {"project_id": "Numeric Scratch project ID"})
def scratch_share_project(project_id: str) -> str:
    err = _require_scratch()
    if err: return err
    try:
        _result, err_msg = _run_scratch_write_action(
            lambda: _scratch_session.connect_project(project_id).share(),
            str(project_id),
        )
        if err_msg:
            return err_msg
        return f"Project {project_id} shared. URL: {_scratch_project_url(project_id)}"
    except Exception as e:
        if _is_scratch_forbidden_error(e):
            return _scratch_forbidden_hint(project_id)
        return f"Error: {e}"

@tool("Unshare a Scratch project.", {"project_id": "Numeric Scratch project ID"})
def scratch_unshare_project(project_id: str) -> str:
    err = _require_scratch()
    if err: return err
    try:
        _result, err_msg = _run_scratch_write_action(
            lambda: _scratch_session.connect_project(project_id).unshare(),
            str(project_id),
        )
        if err_msg:
            return err_msg
        return f"Project {project_id} unshared. URL: {_scratch_project_url(project_id)}"
    except Exception as e:
        if _is_scratch_forbidden_error(e):
            return _scratch_forbidden_hint(project_id)
        return f"Error: {e}"

@tool("Set the title of a Scratch project.", {"project_id": "Numeric project ID", "title": "New title"})
def scratch_set_project_title(project_id: str, title: str) -> str:
    err = _require_scratch()
    if err: return err
    try:
        _scratch_session.connect_project(project_id).set_title(title)
        return f"Title updated to '{title}'."
    except Exception as e:
        return f"Error: {e}"

@tool("Set the instructions of a Scratch project.", {"project_id": "Numeric project ID", "instructions": "New instructions text"})
def scratch_set_project_instructions(project_id: str, instructions: str) -> str:
    err = _require_scratch()
    if err: return err
    try:
        _scratch_session.connect_project(project_id).set_instructions(instructions)
        return "Instructions updated."
    except Exception as e:
        return f"Error: {e}"

@tool("Set the notes & credits of a Scratch project.", {"project_id": "Numeric project ID", "notes": "New notes and credits text"})
def scratch_set_project_notes(project_id: str, notes: str) -> str:
    err = _require_scratch()
    if err: return err
    try:
        _scratch_session.connect_project(project_id).set_notes(notes)
        return "Notes & credits updated."
    except Exception as e:
        return f"Error: {e}"

@tool("Love (like) a Scratch project.", {"project_id": "Numeric Scratch project ID"})
def scratch_love_project(project_id: str) -> str:
    err = _require_scratch()
    if err: return err
    try:
        _scratch_session.connect_project(project_id).love()
        return f"Loved project {project_id}."
    except Exception as e:
        return f"Error: {e}"

@tool("Unlove a Scratch project.", {"project_id": "Numeric Scratch project ID"})
def scratch_unlove_project(project_id: str) -> str:
    err = _require_scratch()
    if err: return err
    try:
        _scratch_session.connect_project(project_id).unlove()
        return f"Unloved project {project_id}."
    except Exception as e:
        return f"Error: {e}"

@tool("Favorite a Scratch project.", {"project_id": "Numeric Scratch project ID"})
def scratch_favorite_project(project_id: str) -> str:
    err = _require_scratch()
    if err: return err
    try:
        _scratch_session.connect_project(project_id).favorite()
        return f"Favorited project {project_id}."
    except Exception as e:
        return f"Error: {e}"

@tool("Unfavorite a Scratch project.", {"project_id": "Numeric Scratch project ID"})
def scratch_unfavorite_project(project_id: str) -> str:
    err = _require_scratch()
    if err: return err
    try:
        _scratch_session.connect_project(project_id).unfavorite()
        return f"Unfavorited project {project_id}."
    except Exception as e:
        return f"Error: {e}"

@tool("Post a comment on a Scratch project.", {"project_id": "Numeric project ID", "comment": "Comment text"})
def scratch_comment_on_project(project_id: str, comment: str) -> str:
    err = _require_scratch()
    if err: return err
    try:
        _scratch_session.connect_project(project_id).post_comment(comment)
        return f"Comment posted on project {project_id}."
    except Exception as e:
        return f"Error: {e}"

@tool("Get the top comments on a Scratch project.", {"project_id": "Numeric project ID", "limit": "Max results (default 20)"})
def scratch_get_project_comments(project_id: str, limit: int = 20) -> str:
    err = _require_scratch()
    if err: return err
    try:
        comments = _scratch_session.connect_project(project_id).comments(limit=limit)
        return json.dumps([
            {"id": c.id, "author": getattr(c, "author_name", ""), "content": getattr(c, "content", "")}
            for c in comments
        ], indent=2)
    except Exception as e:
        return f"Error: {e}"

@tool("Search Scratch projects by keyword.", {"query": "Search term", "limit": "Max results (default 10)"})
def scratch_search_projects(query: str, limit: int = 10) -> str:
    err = _require_scratch()
    if err: return err
    try:
        results = _scratch_session.search_projects(query=query, limit=limit)
        return json.dumps([{"id": p.id, "title": getattr(p, "title", ""), "author": getattr(p, "author_name", "")} for p in results], indent=2)
    except Exception as e:
        return f"Error: {e}"

@tool("Explore trending/popular Scratch projects.", {"query": "Category or keyword (e.g. 'games')", "mode": "Mode: 'trending', 'popular', or 'recent'"})
def scratch_explore_projects(query: str = "*", mode: str = "trending") -> str:
    err = _require_scratch()
    if err: return err
    try:
        results = _scratch_session.explore_projects(query=query, mode=mode, limit=10)
        return json.dumps([{"id": p.id, "title": getattr(p, "title", ""), "author": getattr(p, "author_name", "")} for p in results], indent=2)
    except Exception as e:
        return f"Error: {e}"

# ── Scratch 3.0 Block Reference & Script Builder ────────────────────────────
# Complete opcode reference for ALL Scratch 3.0 blocks so the LLM can build
# any program.  The builder tool converts a simplified JSON description into
# proper Scratch 3.0 block JSON with parent/next chains, typed inputs, etc.
#
# INPUT TYPE CODES (used in the raw JSON):
#   4  = number         5  = positive number  6  = positive integer
#   7  = integer         8  = angle            9  = color (#rrggbb)
#  10  = string         11  = broadcast (name, id)
#  12  = variable (name, id)   13 = list (name, id)
# ---------------------------------------------------------------------------

import uuid as _uuid

def _uid() -> str:
    return _uuid.uuid4().hex[:20]


# Map of simplified input names → (scratch input key, type_code)
_INPUT_TYPE_MAP = {
    # Motion
    "STEPS": 4, "DEGREES": 4, "DIRECTION": 8, "X": 4, "Y": 4,
    "SECS": 4, "DX": 4, "DY": 4, "TOWARDS": 10,
    # Looks
    "MESSAGE": 10, "SIZE": 4, "CHANGE": 4, "NUM": 7, "EFFECT": 10,
    "VALUE": 4, "COSTUME": 10, "BACKDROP": 10,
    # Sound
    "SOUND_MENU": 10, "VOLUME": 4,
    # Events
    "BROADCAST_INPUT": 11, "KEY_OPTION": 10, "WHENGREATERTHANMENU": 10,
    # Control
    "TIMES": 6, "DURATION": 4, "CLONE_OPTION": 10,
    # Sensing
    "QUESTION": 10, "TOUCHINGOBJECTMENU": 10, "COLORPARAM": 9,
    "COLOR": 9, "COLOR2": 9, "DISTANCETOMENU": 10, "DRAG_MODE": 10,
    "CURRENTMENU": 10, "OBJECT": 10,
    # Operators
    "NUM1": 4, "NUM2": 4, "OPERAND1": 10, "OPERAND2": 10,
    "OPERAND": 10, "STRING1": 10, "STRING2": 10, "STRING": 10,
    "LETTER": 6, "FROM": 4, "TO": 4,
    # Pen
    "COLOR_PARAM": 10,
    # Data (variables/lists)
    "ITEM": 10, "INDEX": 7, "LENGTH": 7,
    # My Blocks / Custom
    "custom_block": 10,
}


def _make_input_value(key: str, value, target: dict, project_data: dict):
    """Convert a simplified input value to proper Scratch 3.0 input format.
    
    Supports:
      - Primitives (int, float, str) → literal input
      - {"var": "name"}  → variable reporter
      - {"list": "name"} → list reporter
      - {"block": {...}} → nested reporter block (recursively built)
      - {"op": "operator_add", "NUM1": 1, "NUM2": 2} → inline operator block
    """
    type_code = _INPUT_TYPE_MAP.get(key, 10)  # default to string

    # Dict means a nested block / variable / list reference
    if isinstance(value, dict):
        if "var" in value:
            var_name = value["var"]
            var_id = None
            for vid, vdata in target.get("variables", {}).items():
                if vdata[0] == var_name:
                    var_id = vid
                    break
            # Also check stage variables
            if not var_id:
                for t in project_data.get("targets", []):
                    if t.get("isStage"):
                        for vid, vdata in t.get("variables", {}).items():
                            if vdata[0] == var_name:
                                var_id = vid
                                break
            if not var_id:
                var_id = _uid()
            return [3, [12, var_name, var_id], [type_code, "0"]]

        if "list" in value:
            list_name = value["list"]
            list_id = None
            for lid, ldata in target.get("lists", {}).items():
                if ldata[0] == list_name:
                    list_id = lid
                    break
            if not list_id:
                for t in project_data.get("targets", []):
                    if t.get("isStage"):
                        for lid, ldata in t.get("lists", {}).items():
                            if ldata[0] == list_name:
                                list_id = lid
                                break
            if not list_id:
                list_id = _uid()
            return [3, [13, list_name, list_id], [type_code, ""]]

        if "op" in value or "opcode" in value:
            # Inline reporter block
            return value  # Will be resolved in the block-build pass
        
        # Fallback: stringify
        return [1, [type_code, str(value)]]

    # Boolean or number or string literal
    if isinstance(value, bool):
        return [1, [type_code, "true" if value else "false"]]
    return [1, [type_code, str(value)]]


def _build_block_chain(block_defs: list[dict], target: dict, project_data: dict,
                       parent_id: str | None = None, is_top: bool = False,
                       start_x: int = 0, start_y: int = 0) -> tuple[str | None, dict]:
    """
    Convert a list of simplified block definitions into Scratch 3.0 block dicts.
    
    Returns (first_block_id, {block_id: block_dict, ...})
    """
    all_blocks: dict = {}
    block_ids: list[str] = []

    for bdef in block_defs:
        bid = _uid()
        block_ids.append(bid)
        opcode = bdef.get("opcode", "")

        block = {
            "opcode": opcode,
            "next": None,
            "parent": parent_id if len(block_ids) == 1 else block_ids[-2] if len(block_ids) > 1 else None,
            "inputs": {},
            "fields": {},
            "shadow": False,
            "topLevel": is_top and len(block_ids) == 1,
        }
        if block["topLevel"]:
            block["x"] = start_x
            block["y"] = start_y

        # Build inputs
        for key, val in bdef.items():
            if key in ("opcode", "substack", "substack2", "fields", "mutation",
                       "condition", "x", "y", "comment"):
                continue
            inp_val = _make_input_value(key, val, target, project_data)

            # Handle nested reporter blocks passed as {"op"/"opcode": ...}
            if isinstance(inp_val, dict) and ("op" in inp_val or "opcode" in inp_val):
                reporter_opcode = inp_val.get("op") or inp_val.get("opcode")
                reporter_id = _uid()
                reporter_block = {
                    "opcode": reporter_opcode,
                    "next": None,
                    "parent": bid,
                    "inputs": {},
                    "fields": {},
                    "shadow": False,
                    "topLevel": False,
                }
                # Build reporter inputs
                for rk, rv in inp_val.items():
                    if rk in ("op", "opcode"):
                        continue
                    reporter_block["inputs"][rk] = _make_input_value(rk, rv, target, project_data)
                # Handle reporter fields
                if "fields" in inp_val:
                    reporter_block["fields"] = inp_val["fields"]
                all_blocks[reporter_id] = reporter_block
                type_code = _INPUT_TYPE_MAP.get(key, 10)
                block["inputs"][key] = [3, reporter_id, [type_code, ""]]
            else:
                block["inputs"][key] = inp_val

        # Build fields (dropdown menus, etc.)
        if "fields" in bdef and isinstance(bdef["fields"], dict):
            for fname, fval in bdef["fields"].items():
                if isinstance(fval, list):
                    block["fields"][fname] = fval
                else:
                    block["fields"][fname] = [fval, None]

        # Handle SUBSTACK (for control blocks like repeat, if, forever)
        if "substack" in bdef and bdef["substack"]:
            sub_first, sub_blocks = _build_block_chain(
                bdef["substack"], target, project_data, parent_id=bid
            )
            all_blocks.update(sub_blocks)
            if sub_first:
                block["inputs"]["SUBSTACK"] = [2, sub_first]

        # Handle SUBSTACK2 (for if/else → the else branch)
        if "substack2" in bdef and bdef["substack2"]:
            sub2_first, sub2_blocks = _build_block_chain(
                bdef["substack2"], target, project_data, parent_id=bid
            )
            all_blocks.update(sub2_blocks)
            if sub2_first:
                block["inputs"]["SUBSTACK2"] = [2, sub2_first]

        # Handle CONDITION (boolean input for if/repeat_until/wait_until)
        if "condition" in bdef and bdef["condition"]:
            cond = bdef["condition"]
            if isinstance(cond, dict) and ("op" in cond or "opcode" in cond):
                cond_opcode = cond.get("op") or cond.get("opcode")
                cond_id = _uid()
                cond_block = {
                    "opcode": cond_opcode,
                    "next": None,
                    "parent": bid,
                    "inputs": {},
                    "fields": {},
                    "shadow": False,
                    "topLevel": False,
                }
                for ck, cv in cond.items():
                    if ck in ("op", "opcode", "fields"):
                        continue
                    cond_block["inputs"][ck] = _make_input_value(ck, cv, target, project_data)
                if "fields" in cond:
                    for cfn, cfv in cond["fields"].items():
                        cond_block["fields"][cfn] = cfv if isinstance(cfv, list) else [cfv, None]
                all_blocks[cond_id] = cond_block
                block["inputs"]["CONDITION"] = [2, cond_id]

        # Handle mutation (for custom block definitions/calls)
        if "mutation" in bdef:
            block["mutation"] = bdef["mutation"]

        all_blocks[bid] = block

    # Wire up next/parent chain
    for i in range(len(block_ids)):
        bid = block_ids[i]
        if i > 0:
            all_blocks[bid]["parent"] = block_ids[i - 1]
            all_blocks[block_ids[i - 1]]["next"] = bid
        elif parent_id:
            all_blocks[bid]["parent"] = parent_id

    first_id = block_ids[0] if block_ids else None
    return first_id, all_blocks


# ── Comprehensive block-building tool ──────────────────────────────────────

@tool(
    "Build and add a complete Scratch script (block stack) to a sprite. "
    "You MUST call scratch_get_project_json first. "
    "Pass blocks_json as a JSON array of block definitions. Each block is an object with 'opcode' and input keys. "
    "\n\nAVAILABLE OPCODES:\n"
    "MOTION: motion_movesteps(STEPS), motion_turnright(DEGREES), motion_turnleft(DEGREES), "
    "motion_goto(TO — use a menu block or 'random position'/'mouse-pointer'), motion_gotoxy(X,Y), "
    "motion_glideto(SECS,TO), motion_glidesecto(SECS,X,Y), motion_pointindirection(DIRECTION), "
    "motion_pointtowards(TOWARDS), motion_changexby(DX), motion_changeyby(DY), "
    "motion_setx(X), motion_sety(Y), motion_ifonedgebounce(), motion_setrotationstyle(fields:{'STYLE':['left-right',null]}), "
    "motion_xposition (reporter), motion_yposition (reporter), motion_direction (reporter)\n"
    "LOOKS: looks_sayforsecs(MESSAGE,SECS), looks_say(MESSAGE), looks_thinkforsecs(MESSAGE,SECS), "
    "looks_think(MESSAGE), looks_switchcostumeto(COSTUME), looks_nextcostume(), "
    "looks_switchbackdropto(BACKDROP), looks_nextbackdrop(), looks_changesizeby(CHANGE), "
    "looks_setsizeto(SIZE), looks_changeeffectby(EFFECT — use fields:{'EFFECT':['COLOR',null]},CHANGE), "
    "looks_seteffectto(EFFECT — use fields:{'EFFECT':['COLOR',null]},VALUE), looks_cleargraphiceffects(), "
    "looks_show(), looks_hide(), looks_gotofrontback(fields:{'FRONT_BACK':['front',null]}), "
    "looks_goforwardbackwardlayers(NUM,fields:{'FORWARD_BACKWARD':['forward',null]}), "
    "looks_costumenumbername (reporter), looks_backdropnumbername (reporter), looks_size (reporter)\n"
    "SOUND: sound_playuntildone(SOUND_MENU), sound_play(SOUND_MENU), sound_stopallsounds(), "
    "sound_changeeffectby(VALUE,fields:{'EFFECT':['PITCH',null]}), "
    "sound_seteffectto(VALUE,fields:{'EFFECT':['PITCH',null]}), sound_cleareffects(), "
    "sound_changevolumeby(VOLUME), sound_setvolumeto(VOLUME), sound_volume (reporter)\n"
    "EVENTS: event_whenflagclicked, event_whenkeypressed(fields:{'KEY_OPTION':['space',null]}), "
    "event_whenthisspriteclicked, event_whenbackdropswitchesto(fields:{'BACKDROP':['backdrop1',null]}), "
    "event_whengreaterthan(VALUE,fields:{'WHENGREATERTHANMENU':['LOUDNESS',null]}), "
    "event_whenbroadcastreceived(fields:{'BROADCAST_OPTION':['message1',broadcastId]}), "
    "event_broadcast(BROADCAST_INPUT), event_broadcastandwait(BROADCAST_INPUT)\n"
    "CONTROL: control_wait(DURATION), control_repeat(TIMES,substack:[...]), "
    "control_forever(substack:[...]), control_if(condition:{...},substack:[...]), "
    "control_if_else(condition:{...},substack:[...],substack2:[...]), "
    "control_wait_until(condition:{...}), control_repeat_until(condition:{...},substack:[...]), "
    "control_stop(fields:{'STOP_OPTION':['all',null]}), "
    "control_start_as_clone(), control_create_clone_of(CLONE_OPTION), control_delete_this_clone()\n"
    "SENSING: sensing_touchingobject(TOUCHINGOBJECTMENU), sensing_touchingcolor(COLOR), "
    "sensing_coloristouchingcolor(COLOR,COLOR2), sensing_distanceto(DISTANCETOMENU), "
    "sensing_askandwait(QUESTION), sensing_answer (reporter), sensing_keypressed(KEY_OPTION), "
    "sensing_mousedown (reporter), sensing_mousex (reporter), sensing_mousey (reporter), "
    "sensing_setdragmode(fields:{'DRAG_MODE':['draggable',null]}), sensing_loudness (reporter), "
    "sensing_timer (reporter), sensing_resettimer(), sensing_of(OBJECT,fields:{'PROPERTY':['x position',null]}), "
    "sensing_current(fields:{'CURRENTMENU':['year',null]}), sensing_dayssince2000 (reporter), "
    "sensing_username (reporter)\n"
    "OPERATORS: operator_add(NUM1,NUM2), operator_subtract(NUM1,NUM2), operator_multiply(NUM1,NUM2), "
    "operator_divide(NUM1,NUM2), operator_random(FROM,TO), operator_gt(OPERAND1,OPERAND2), "
    "operator_lt(OPERAND1,OPERAND2), operator_equals(OPERAND1,OPERAND2), operator_and(OPERAND1,OPERAND2), "
    "operator_or(OPERAND1,OPERAND2), operator_not(OPERAND), operator_join(STRING1,STRING2), "
    "operator_letter_of(STRING,LETTER), operator_length(STRING), operator_contains(STRING1,STRING2), "
    "operator_mod(NUM1,NUM2), operator_round(NUM), operator_mathop(NUM,fields:{'OPERATOR':['abs',null]})\n"
    "DATA (variables): data_setvariableto(VALUE,fields:{'VARIABLE':['varname','varid']}), "
    "data_changevariableby(VALUE,fields:{'VARIABLE':['varname','varid']}), "
    "data_showvariable(fields:{'VARIABLE':['varname','varid']}), "
    "data_hidevariable(fields:{'VARIABLE':['varname','varid']})\n"
    "DATA (lists): data_addtolist(ITEM,fields:{'LIST':['listname','listid']}), "
    "data_deleteoflist(INDEX,fields:{'LIST':['listname','listid']}), "
    "data_deletealloflist(fields:{'LIST':['listname','listid']}), "
    "data_insertatlist(ITEM,INDEX,fields:{'LIST':['listname','listid']}), "
    "data_replaceitemoflist(INDEX,ITEM,fields:{'LIST':['listname','listid']}), "
    "data_itemoflist(INDEX,fields:{'LIST':['listname','listid']}) (reporter), "
    "data_itemnumoflist(ITEM,fields:{'LIST':['listname','listid']}) (reporter), "
    "data_lengthoflist(fields:{'LIST':['listname','listid']}) (reporter), "
    "data_listcontainsitem(ITEM,fields:{'LIST':['listname','listid']}) (boolean), "
    "data_showlist(fields:{'LIST':['listname','listid']}), "
    "data_hidelist(fields:{'LIST':['listname','listid']})\n"
    "PEN: pen_clear(), pen_stamp(), pen_penDown(), pen_penUp(), "
    "pen_setPenColorToColor(COLOR), pen_changePenColorParamBy(COLOR_PARAM,VALUE), "
    "pen_setPenColorParamTo(COLOR_PARAM,VALUE), pen_changePenSizeBy(SIZE), pen_setPenSizeTo(SIZE)\n"
    "CUSTOM BLOCKS: procedures_definition (hat), procedures_prototype (inside definition), "
    "procedures_call (to call a custom block — set mutation with proccode, argumentids, argumentnames)\n"
    "\nINPUT FORMAT:\n"
    "- Literal: {\"STEPS\": 10} or {\"MESSAGE\": \"hello\"}\n"
    "- Variable reference: {\"VALUE\": {\"var\": \"my variable\"}}\n"
    "- List reference: {\"ITEM\": {\"list\": \"my list\"}}\n"
    "- Nested reporter: {\"STEPS\": {\"op\": \"operator_add\", \"NUM1\": 3, \"NUM2\": 5}}\n"
    "- Boolean condition: \"condition\": {\"op\": \"sensing_touchingobject\", \"TOUCHINGOBJECTMENU\": \"_edge_\"}\n"
    "- Substacks (loops/if): \"substack\": [{...}, {...}]\n"
    "- Else branch: \"substack2\": [{...}, {...}]\n"
    "- Dropdown fields: \"fields\": {\"STOP_OPTION\": [\"all\", null]}\n",
    {
        "project_id": "Numeric Scratch project ID",
        "sprite_name": "Sprite name exactly as shown in scratch_get_project_json (e.g. 'Sprite1', 'Stage')",
        "blocks_json": "JSON array of block definitions — see opcode reference above. First block should be a hat/event block.",
    },
)
def scratch_build_script(project_id: str, sprite_name: str, blocks_json: str) -> str:
    err = _require_scratch()
    if err:
        return err
    try:
        cached = _project_json_cache.get(str(project_id))
        if not cached:
            return "Project JSON not cached. Call scratch_get_project_json first."

        block_defs = json.loads(blocks_json)
        if not isinstance(block_defs, list) or len(block_defs) == 0:
            return "blocks_json must be a non-empty JSON array of block definitions."

        targets = cached.get("targets", [])
        target = next((t for t in targets if t["name"].lower() == sprite_name.lower()), None)
        if not target and sprite_name.strip().lower() in {"scratch cat", "cat", "scratchcat"}:
            target = next((t for t in targets if t.get("name", "").lower() == "sprite1"), None)
        if not target:
            available = [t["name"] for t in targets]
            return f"Sprite '{sprite_name}' not found. Available: {available}"

        # Calculate position offset so new scripts don't overlap
        existing_tops = [b for b in target.get("blocks", {}).values()
                         if isinstance(b, dict) and b.get("topLevel")]
        y_offset = max((b.get("y", 0) for b in existing_tops), default=-100) + 200

        first_id, new_blocks = _build_block_chain(
            block_defs, target, cached,
            parent_id=None, is_top=True,
            start_x=50, start_y=y_offset,
        )
        if not new_blocks:
            return "No blocks were generated from the input."

        target["blocks"].update(new_blocks)

        # Commit to Scratch
        _result, err_msg = _run_scratch_write_action(
            lambda: _scratch_session.connect_project(project_id).set_json(cached),
            str(project_id),
        )
        if err_msg:
            return err_msg
        _project_json_cache[str(project_id)] = cached

        opcodes_added = [b.get("opcode", "?") for b in new_blocks.values()]
        return (
            f"DONE: Added script with {len(new_blocks)} blocks to '{sprite_name}' in project {project_id}. "
            f"Opcodes: {opcodes_added[:10]}{'...' if len(opcodes_added) > 10 else ''}. "
            f"Edit committed to Scratch. URL: {_scratch_project_url(project_id)}"
        )
    except json.JSONDecodeError as e:
        return f"Invalid blocks_json: {e}"
    except Exception as e:
        if _is_scratch_forbidden_error(e):
            return _scratch_forbidden_hint(project_id)
        return f"Error building script: {e}"


# ── Legacy convenience wrapper (calls scratch_build_script internally) ──────

@tool(
    "Add a simple say-block script to a sprite. For more complex scripts use scratch_build_script. "
    "You MUST call scratch_get_project_json first to cache the project.",
    {
        "project_id": "Numeric Scratch project ID",
        "sprite_name": "Sprite name (e.g. 'Sprite1')",
        "message": "The message the sprite will say",
        "say_for_seconds": "Seconds to display (0 = permanent)",
        "trigger": "Trigger: 'flag' or 'clicked'",
    },
)
def scratch_add_say_block(project_id: str, sprite_name: str, message: str,
                          say_for_seconds: int = 0, trigger: str = "flag") -> str:
    trig = (trigger or "flag").strip().lower()
    event_opcode = ("event_whenthisspriteclicked"
                    if trig in {"clicked", "click", "sprite_clicked", "when_clicked"}
                    else "event_whenflagclicked")
    opcode = "looks_sayforsecs" if say_for_seconds > 0 else "looks_say"
    blocks = [{"opcode": event_opcode}]
    say_block: dict = {"opcode": opcode, "MESSAGE": message}
    if say_for_seconds > 0:
        say_block["SECS"] = say_for_seconds
    blocks.append(say_block)
    return scratch_build_script(project_id, sprite_name, json.dumps(blocks))


# ── Variable & list creation tool ──────────────────────────────────────────

@tool(
    "Create a variable or list in a Scratch project. "
    "You MUST call scratch_get_project_json first. "
    "Variables/lists created on the Stage are global; on a sprite they are local.",
    {
        "project_id": "Numeric Scratch project ID",
        "sprite_name": "Sprite name to create it on, or 'Stage' for global",
        "name": "Variable or list name",
        "kind": "'variable' or 'list'",
        "default_value": "Initial value (string or number for variables, ignored for lists)",
    },
)
def scratch_create_variable(project_id: str, sprite_name: str, name: str,
                            kind: str = "variable", default_value: str = "0") -> str:
    err = _require_scratch()
    if err:
        return err
    try:
        cached = _project_json_cache.get(str(project_id))
        if not cached:
            return "Project JSON not cached. Call scratch_get_project_json first."

        targets = cached.get("targets", [])
        target = next((t for t in targets if t["name"].lower() == sprite_name.lower()), None)
        if not target:
            available = [t["name"] for t in targets]
            return f"Sprite '{sprite_name}' not found. Available: {available}"

        new_id = _uid()
        k = kind.lower().strip()

        if k == "list":
            target.setdefault("lists", {})[new_id] = [name, []]
        else:
            target.setdefault("variables", {})[new_id] = [name, default_value]

        _scratch_session.connect_project(project_id).set_json(cached)
        _project_json_cache[str(project_id)] = cached
        return (
            f"DONE: Created {k} '{name}' on '{sprite_name}' (id={new_id}) in project {project_id}. "
            f"URL: {_scratch_project_url(project_id)}"
        )
    except Exception as e:
        if _is_scratch_forbidden_error(e):
            return _scratch_forbidden_hint(project_id)
        return f"Error: {e}"


# ── Broadcast creation tool ────────────────────────────────────────────────

@tool(
    "Create a broadcast message in a Scratch project. "
    "You MUST call scratch_get_project_json first.",
    {
        "project_id": "Numeric Scratch project ID",
        "broadcast_name": "Name of the broadcast message to create",
    },
)
def scratch_create_broadcast(project_id: str, broadcast_name: str) -> str:
    err = _require_scratch()
    if err:
        return err
    try:
        cached = _project_json_cache.get(str(project_id))
        if not cached:
            return "Project JSON not cached. Call scratch_get_project_json first."

        # Add broadcast to Stage target
        stage = next((t for t in cached.get("targets", []) if t.get("isStage")), None)
        if not stage:
            return "No Stage target found in project."

        new_id = _uid()
        stage.setdefault("broadcasts", {})[new_id] = broadcast_name

        _scratch_session.connect_project(project_id).set_json(cached)
        _project_json_cache[str(project_id)] = cached
        return (
            f"DONE: Created broadcast '{broadcast_name}' (id={new_id}) in project {project_id}. "
            f"URL: {_scratch_project_url(project_id)}"
        )
    except Exception as e:
        if _is_scratch_forbidden_error(e):
            return _scratch_forbidden_hint(project_id)
        return f"Error: {e}"


# ── Custom block (procedure) definition tool ────────────────────────────────

@tool(
    "Define a custom block (My Block / procedure) in a Scratch sprite. "
    "You MUST call scratch_get_project_json first. "
    "After creating the definition, use scratch_build_script with procedures_call to call it.",
    {
        "project_id": "Numeric Scratch project ID",
        "sprite_name": "Sprite name to add the custom block to",
        "proccode": "Procedure label with %s for string args, %n for number args, %b for boolean args. E.g. 'draw square %n'",
        "argument_names": "JSON array of argument names, e.g. '[\"size\"]'",
        "body_blocks_json": "JSON array of block definitions for the body of the custom block (same format as scratch_build_script)",
        "warp": "Run without screen refresh (true/false)",
    },
)
def scratch_create_custom_block(project_id: str, sprite_name: str, proccode: str,
                                argument_names: str = "[]",
                                body_blocks_json: str = "[]",
                                warp: str = "false") -> str:
    err = _require_scratch()
    if err:
        return err
    try:
        cached = _project_json_cache.get(str(project_id))
        if not cached:
            return "Project JSON not cached. Call scratch_get_project_json first."

        targets = cached.get("targets", [])
        target = next((t for t in targets if t["name"].lower() == sprite_name.lower()), None)
        if not target:
            available = [t["name"] for t in targets]
            return f"Sprite '{sprite_name}' not found. Available: {available}"

        arg_names = json.loads(argument_names)
        body_defs = json.loads(body_blocks_json)

        # Build argument IDs and defaults
        arg_ids = [_uid() for _ in arg_names]
        arg_defaults = []
        for c in proccode:
            if c == 's':
                arg_defaults.append("")
            elif c == 'n':
                arg_defaults.append("")
            elif c == 'b':
                arg_defaults.append("false")
        # Pad if needed
        while len(arg_defaults) < len(arg_names):
            arg_defaults.append("")

        # Create the prototype block
        proto_id = _uid()
        def_id = _uid()

        proto_block = {
            "opcode": "procedures_prototype",
            "next": None,
            "parent": def_id,
            "inputs": {},
            "fields": {},
            "shadow": True,
            "topLevel": False,
            "mutation": {
                "tagName": "mutation",
                "children": [],
                "proccode": proccode,
                "argumentids": json.dumps(arg_ids),
                "argumentnames": json.dumps(arg_names),
                "argumentdefaults": json.dumps(arg_defaults),
                "warp": str(warp).lower(),
            },
        }

        # Create argument reporter blocks inside the prototype
        for i, aid in enumerate(arg_ids):
            arg_block_id = aid  # use the argument id as the block id
            placeholders = proccode.split(" ")
            # Determine type from proccode pattern
            param_markers = [p for p in proccode.replace("%%", "").split() if p.startswith("%")]
            arg_type = param_markers[i] if i < len(param_markers) else "%s"

            if arg_type == "%b":
                arg_opcode = "argument_reporter_boolean"
            else:
                arg_opcode = "argument_reporter_string_number"

            target["blocks"][arg_block_id] = {
                "opcode": arg_opcode,
                "next": None,
                "parent": proto_id,
                "inputs": {},
                "fields": {"VALUE": [arg_names[i], None]},
                "shadow": True,
                "topLevel": False,
            }
            proto_block["inputs"][aid] = [1, arg_block_id]

        # Calculate y offset
        existing_tops = [b for b in target.get("blocks", {}).values()
                         if isinstance(b, dict) and b.get("topLevel")]
        y_offset = max((b.get("y", 0) for b in existing_tops), default=-100) + 200

        # Create the definition hat block
        def_block = {
            "opcode": "procedures_definition",
            "next": None,
            "parent": None,
            "inputs": {"custom_block": [1, proto_id]},
            "fields": {},
            "shadow": False,
            "topLevel": True,
            "x": 50,
            "y": y_offset,
        }

        target["blocks"][def_id] = def_block
        target["blocks"][proto_id] = proto_block

        # Build body blocks
        if body_defs:
            first_body_id, body_blocks = _build_block_chain(
                body_defs, target, cached, parent_id=def_id
            )
            target["blocks"].update(body_blocks)
            if first_body_id:
                def_block["next"] = first_body_id
                body_blocks[first_body_id]["parent"] = def_id

        # Commit
        _scratch_session.connect_project(project_id).set_json(cached)
        _project_json_cache[str(project_id)] = cached

        return (
            f"DONE: Created custom block '{proccode}' with args {arg_names} on '{sprite_name}'. "
            f"Definition ID: {def_id}, Prototype ID: {proto_id}. "
            f"URL: {_scratch_project_url(project_id)}. "
            f"To call it, use scratch_build_script with: "
            f"{{\"opcode\": \"procedures_call\", \"mutation\": {{\"tagName\": \"mutation\", \"children\": [], "
            f"\"proccode\": \"{proccode}\", \"argumentids\": {json.dumps(arg_ids)}, \"warp\": \"{warp}\"}}}}"
        )
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"
    except Exception as e:
        if _is_scratch_forbidden_error(e):
            return _scratch_forbidden_hint(project_id)
        return f"Error: {e}"


# ── Delete all scripts from a sprite ────────────────────────────────────────

@tool(
    "Delete all blocks/scripts from a sprite in a Scratch project. "
    "You MUST call scratch_get_project_json first.",
    {
        "project_id": "Numeric Scratch project ID",
        "sprite_name": "Sprite name to clear scripts from",
    },
)
def scratch_clear_scripts(project_id: str, sprite_name: str) -> str:
    err = _require_scratch()
    if err:
        return err
    try:
        cached = _project_json_cache.get(str(project_id))
        if not cached:
            return "Project JSON not cached. Call scratch_get_project_json first."

        targets = cached.get("targets", [])
        target = next((t for t in targets if t["name"].lower() == sprite_name.lower()), None)
        if not target:
            available = [t["name"] for t in targets]
            return f"Sprite '{sprite_name}' not found. Available: {available}"

        old_count = len(target.get("blocks", {}))
        target["blocks"] = {}

        _result, err_msg = _run_scratch_write_action(
            lambda: _scratch_session.connect_project(project_id).set_json(cached),
            str(project_id),
        )
        if err_msg:
            return err_msg
        _project_json_cache[str(project_id)] = cached
        return (
            f"DONE: Cleared {old_count} blocks from '{sprite_name}' in project {project_id}. "
            f"URL: {_scratch_project_url(project_id)}"
        )
    except Exception as e:
        if _is_scratch_forbidden_error(e):
            return _scratch_forbidden_hint(project_id)
        return f"Error: {e}"


# ── Add extension to project ───────────────────────────────────────────────

@tool(
    "Add an extension to a Scratch project (e.g. 'pen', 'music', 'videoSensing', 'text2speech', 'translate', 'makeymakey', 'microbit', 'ev3', 'boost', 'wedo2', 'gdxfor'). "
    "You MUST call scratch_get_project_json first.",
    {
        "project_id": "Numeric Scratch project ID",
        "extension_id": "Extension identifier (e.g. 'pen', 'music', 'text2speech')",
    },
)
def scratch_add_extension(project_id: str, extension_id: str) -> str:
    err = _require_scratch()
    if err:
        return err
    try:
        cached = _project_json_cache.get(str(project_id))
        if not cached:
            return "Project JSON not cached. Call scratch_get_project_json first."

        exts = cached.setdefault("extensions", [])
        if extension_id not in exts:
            exts.append(extension_id)

        _scratch_session.connect_project(project_id).set_json(cached)
        _project_json_cache[str(project_id)] = cached
        return f"DONE: Extension '{extension_id}' added to project {project_id}. URL: {_scratch_project_url(project_id)}"
    except Exception as e:
        if _is_scratch_forbidden_error(e):
            return _scratch_forbidden_hint(project_id)
        return f"Error: {e}"

# ── Studios ──────────────────────────────────────────────────────────────────

@tool("Get info about a Scratch studio.", {"studio_id": "Numeric studio ID"})
def scratch_get_studio(studio_id: str) -> str:
    err = _require_scratch()
    if err: return err
    try:
        s = _scratch_session.connect_studio(studio_id)
        return json.dumps({
            "id": getattr(s, "id", studio_id),
            "title": getattr(s, "title", ""),
            "description": getattr(s, "description", ""),
            "followers": getattr(s, "follower_count", 0),
            "projects": getattr(s, "project_count", 0),
            "open_to_all": getattr(s, "open_to_all", False),
        }, indent=2)
    except Exception as e:
        return f"Error: {e}"

@tool("Add a project to a Scratch studio.", {"studio_id": "Studio ID", "project_id": "Project ID to add"})
def scratch_studio_add_project(studio_id: str, project_id: str) -> str:
    err = _require_scratch()
    if err: return err
    try:
        _scratch_session.connect_studio(studio_id).add_project(project_id)
        return f"Project {project_id} added to studio {studio_id}."
    except Exception as e:
        return f"Error: {e}"

@tool("Remove a project from a Scratch studio.", {"studio_id": "Studio ID", "project_id": "Project ID to remove"})
def scratch_studio_remove_project(studio_id: str, project_id: str) -> str:
    err = _require_scratch()
    if err: return err
    try:
        _scratch_session.connect_studio(studio_id).remove_project(project_id)
        return f"Project {project_id} removed from studio {studio_id}."
    except Exception as e:
        return f"Error: {e}"

@tool("Post a comment on a Scratch studio.", {"studio_id": "Studio ID", "comment": "Comment text"})
def scratch_comment_on_studio(studio_id: str, comment: str) -> str:
    err = _require_scratch()
    if err: return err
    try:
        _scratch_session.connect_studio(studio_id).post_comment(comment)
        return f"Comment posted on studio {studio_id}."
    except Exception as e:
        return f"Error: {e}"

@tool("Follow a Scratch studio.", {"studio_id": "Studio ID"})
def scratch_follow_studio(studio_id: str) -> str:
    err = _require_scratch()
    if err: return err
    try:
        _scratch_session.connect_studio(studio_id).follow()
        return f"Now following studio {studio_id}."
    except Exception as e:
        return f"Error: {e}"

@tool("Invite a user to curate a Scratch studio.", {"studio_id": "Studio ID", "username": "Username to invite"})
def scratch_studio_invite_curator(studio_id: str, username: str) -> str:
    err = _require_scratch()
    if err: return err
    try:
        _scratch_session.connect_studio(studio_id).invite_curator(username)
        return f"Invited '{username}' to studio {studio_id}."
    except Exception as e:
        return f"Error: {e}"

# ── Cloud Variables ──────────────────────────────────────────────────────────

@tool(
    "Get all cloud variables for a Scratch project.",
    {"project_id": "Numeric Scratch project ID"},
)
def scratch_get_cloud_vars(project_id: str) -> str:
    err = _require_scratch()
    if err: return err
    try:
        cloud = _scratch_session.connect_cloud(project_id)
        variables = cloud.get_all_vars()
        return json.dumps(variables, indent=2)
    except Exception as e:
        return f"Error: {e}"

@tool(
    "Set a cloud variable in a Scratch project.",
    {"project_id": "Numeric Scratch project ID", "variable": "Variable name (without the cloud ☁ emoji)", "value": "Value to set"},
)
def scratch_set_cloud_var(project_id: str, variable: str, value: str) -> str:
    err = _require_scratch()
    if err: return err
    try:
        cloud = _scratch_session.connect_cloud(project_id)
        cloud.set_var(variable, value)
        return f"Cloud variable '{variable}' set to '{value}'."
    except Exception as e:
        return f"Error: {e}"

# ── Forum ────────────────────────────────────────────────────────────────────

@tool(
    "Get the posts in a Scratch forum topic.",
    {"topic_id": "Forum topic ID", "page": "Page number (first page = 1)"},
)
def scratch_get_forum_posts(topic_id: str, page: int = 1) -> str:
    err = _require_scratch()
    if err: return err
    try:
        topic = scratch3.get_topic(topic_id)
        posts = topic.posts(page=page)
        return json.dumps([
            {"id": p.id, "author": getattr(p, "author_name", ""), "content": getattr(p, "content", "")[:300]}
            for p in posts
        ], indent=2)
    except Exception as e:
        return f"Error: {e}"

# ── Front Page / Site Data ───────────────────────────────────────────────────

@tool("Get the featured / trending projects from the Scratch front page.", {})
def scratch_featured_projects() -> str:
    if scratch3 is None:
        return "scratchattach not installed."
    try:
        projects = scratch3.featured_projects()
        return json.dumps([{"id": p.id, "title": getattr(p, "title", "")} for p in projects[:15]], indent=2)
    except Exception as e:
        return f"Error: {e}"

@tool("Get the latest Scratch news.", {"limit": "Number of news items (default 5)"})
def scratch_get_news(limit: int = 5) -> str:
    if scratch3 is None:
        return "scratchattach not installed."
    try:
        news = scratch3.get_news(limit=limit)
        return json.dumps(news, indent=2)
    except Exception as e:
        return f"Error: {e}"


# ── Project Editor (scratchattach.editor) ────────────────────────────────────
# Docs: https://github.com/TimMcCool/scratchattach/wiki/Project-parsing-editing

try:
    from scratchattach import editor as _sa_editor
except Exception:
    _sa_editor = None  # type: ignore

# Cache for loaded editor Project objects
_editor_project_cache: dict[str, object] = {}


def _require_editor() -> str | None:
    if _sa_editor is None:
        return "scratchattach.editor could not be imported. Make sure scratchattach is installed."
    return None


@tool(
    "Load a Scratch project into the editor by its project ID so it can be inspected or modified. "
    "Must be called before any other scratch_editor_* tools.",
    {"project_id": "Numeric Scratch project ID"},
)
def scratch_editor_load(project_id: str) -> str:
    err = _require_editor()
    if err: return err
    try:
        proj = _sa_editor.Project.from_id(int(project_id))
        _editor_project_cache[str(project_id)] = proj
        sprites = [str(s) for s in getattr(proj, "sprites", [])]
        return json.dumps({
            "project_id": project_id,
            "name": getattr(proj, "name", ""),
            "sprites": sprites,
            "extensions": [str(e) for e in getattr(proj, "extensions", [])],
            "meta": str(getattr(proj, "meta", "")),
            "note": "Project loaded into editor cache. Use other scratch_editor_* tools to inspect or edit it.",
        }, indent=2)
    except Exception as e:
        return f"Error: {e}"


@tool(
    "Load a Scratch project from a local .sb3 file into the editor.",
    {"filepath": "Absolute path to the .sb3 file on disk"},
)
def scratch_editor_load_sb3(filepath: str) -> str:
    err = _require_editor()
    if err: return err
    try:
        proj = _sa_editor.Project.from_sb3(filepath)
        key = filepath
        _editor_project_cache[key] = proj
        sprites = [str(s) for s in getattr(proj, "sprites", [])]
        return json.dumps({
            "key": key,
            "name": getattr(proj, "name", ""),
            "sprites": sprites,
            "extensions": [str(e) for e in getattr(proj, "extensions", [])],
            "meta": str(getattr(proj, "meta", "")),
        }, indent=2)
    except Exception as e:
        return f"Error: {e}"


@tool(
    "List all assets (costumes and sounds) in an editor-loaded Scratch project.",
    {"project_id": "Project ID or filepath used when loading with scratch_editor_load or scratch_editor_load_sb3"},
)
def scratch_editor_list_assets(project_id: str) -> str:
    err = _require_editor()
    if err: return err
    proj = _editor_project_cache.get(str(project_id))
    if not proj:
        return "Project not loaded. Call scratch_editor_load first."
    try:
        assets = [str(a) for a in getattr(proj, "assets", [])]
        return json.dumps(assets, indent=2)
    except Exception as e:
        return f"Error: {e}"


@tool(
    "Find a variable, list, or broadcast by name in an editor-loaded Scratch project.",
    {
        "project_id": "Project ID used when loading",
        "name": "Variable / list / broadcast name to search for",
        "multiple": "If 'true', return all matches (useful when a variable and list share a name)",
    },
)
def scratch_editor_find_vlb(project_id: str, name: str, multiple: str = "false") -> str:
    err = _require_editor()
    if err: return err
    proj = _editor_project_cache.get(str(project_id))
    if not proj:
        return "Project not loaded. Call scratch_editor_load first."
    try:
        find_multiple = multiple.lower() == "true"
        result = proj.find_vlb(name, multiple=find_multiple)
        if result is None:
            return f"No variable/list/broadcast named '{name}' found."
        if isinstance(result, list):
            return json.dumps([str(r) for r in result], indent=2)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool(
    "List the monitors (on-screen variable/list displays) in an editor-loaded project.",
    {"project_id": "Project ID used when loading"},
)
def scratch_editor_list_monitors(project_id: str) -> str:
    err = _require_editor()
    if err: return err
    proj = _editor_project_cache.get(str(project_id))
    if not proj:
        return "Project not loaded. Call scratch_editor_load first."
    try:
        monitors = [str(m) for m in getattr(proj, "monitors", [])]
        return json.dumps(monitors, indent=2)
    except Exception as e:
        return f"Error: {e}"


@tool(
    "Get the TurboWarp configuration comment embedded in an editor-loaded project.",
    {"project_id": "Project ID used when loading"},
)
def scratch_editor_tw_config(project_id: str) -> str:
    err = _require_editor()
    if err: return err
    proj = _editor_project_cache.get(str(project_id))
    if not proj:
        return "Project not loaded. Call scratch_editor_load first."
    try:
        return str(getattr(proj, "tw_config", "No TurboWarp config found."))
    except Exception as e:
        return f"Error: {e}"


@tool(
    "List all sprites in an editor-loaded project, including their block and costume counts.",
    {"project_id": "Project ID used when loading"},
)
def scratch_editor_list_sprites(project_id: str) -> str:
    err = _require_editor()
    if err: return err
    proj = _editor_project_cache.get(str(project_id))
    if not proj:
        return "Project not loaded. Call scratch_editor_load first."
    try:
        result = []
        for sprite in getattr(proj, "sprites", []):
            result.append({
                "name": getattr(sprite, "name", str(sprite)),
                "costumes": len(getattr(sprite, "costumes", [])),
                "sounds": len(getattr(sprite, "sounds", [])),
                "scripts": len(getattr(sprite, "scripts", [])),
            })
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {e}"


@tool(
    "Get the scripts (block stacks) of a sprite in an editor-loaded project.",
    {
        "project_id": "Project ID used when loading",
        "sprite_name": "Name of the sprite (from scratch_editor_list_sprites)",
    },
)
def scratch_editor_get_sprite_scripts(project_id: str, sprite_name: str) -> str:
    err = _require_editor()
    if err: return err
    proj = _editor_project_cache.get(str(project_id))
    if not proj:
        return "Project not loaded. Call scratch_editor_load first."
    try:
        sprite = next(
            (s for s in getattr(proj, "sprites", []) if getattr(s, "name", "") == sprite_name),
            None
        )
        if sprite is None:
            available = [getattr(s, "name", str(s)) for s in getattr(proj, "sprites", [])]
            return f"Sprite '{sprite_name}' not found. Available: {available}"
        scripts = [str(sc) for sc in getattr(sprite, "scripts", [])]
        return json.dumps(scripts, indent=2)
    except Exception as e:
        return f"Error: {e}"



# ---------------------------------------------------------------------------
# Developer coding tools (filesystem, search, shell, git, diagnostics)
# ---------------------------------------------------------------------------

_WORKSPACE_ROOT = Path(os.getcwd()).resolve()
_IGNORE_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache", "dist", "build"}


def _safe_path(path: str) -> tuple[Path | None, str | None]:
    try:
        raw = (path or ".").strip()
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = _WORKSPACE_ROOT / candidate
        resolved = candidate.resolve()
        if not str(resolved).startswith(str(_WORKSPACE_ROOT)):
            return None, f"Path '{path}' is outside the workspace root ({_WORKSPACE_ROOT})."
        return resolved, None
    except Exception as e:
        return None, f"Invalid path '{path}': {e}"


def _shorten(text: str, max_chars: int = 12000) -> str:
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... (truncated)"


def _iter_files(root: Path):
    for p in root.rglob("*"):
        if any(part in _IGNORE_DIRS for part in p.parts):
            continue
        if p.is_file():
            yield p


def _run_cmd(command: str, cwd: Path | None = None, timeout_seconds: int = 60) -> dict:
    blocked_tokens = [
        "rm -rf /", "mkfs", "shutdown", "reboot", ":(){", "dd if=", "sudo ",
        "chmod -R 777 /", "chown -R /", "diskutil eraseDisk",
    ]
    lower = (command or "").lower()
    if any(tok in lower for tok in blocked_tokens):
        return {
            "ok": False,
            "error": "Blocked potentially destructive command.",
            "command": command,
        }

    workdir = cwd or _WORKSPACE_ROOT
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=max(1, timeout_seconds),
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "command": command,
            "cwd": str(workdir),
            "stdout": _shorten(proc.stdout or ""),
            "stderr": _shorten(proc.stderr or ""),
        }
    except subprocess.TimeoutExpired as e:
        return {
            "ok": False,
            "error": f"Command timed out after {timeout_seconds}s",
            "command": command,
            "cwd": str(workdir),
            "stdout": _shorten(e.stdout or ""),
            "stderr": _shorten(e.stderr or ""),
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "command": command,
            "cwd": str(workdir),
        }


@tool("Get the current workspace root path used by coding tools.", {})
def dev_workspace_root() -> str:
    return str(_WORKSPACE_ROOT)


@tool(
    "List files and folders in a workspace directory.",
    {
        "path": "Path inside workspace (default '.')",
        "recursive": "If true, list recursively",
        "max_entries": "Maximum entries to return (default 200)",
    },
)
def dev_list_directory(path: str = ".", recursive: bool = False, max_entries: int = 200) -> str:
    p, err = _safe_path(path)
    if err:
        return err
    if not p.exists():
        return f"Path not found: {p}"
    if not p.is_dir():
        return f"Not a directory: {p}"

    entries = []
    if recursive:
        for item in _iter_files(p):
            entries.append(str(item.relative_to(_WORKSPACE_ROOT)))
            if len(entries) >= max_entries:
                break
    else:
        for item in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            entries.append(str(item.relative_to(_WORKSPACE_ROOT)) + ("/" if item.is_dir() else ""))
            if len(entries) >= max_entries:
                break
    return json.dumps(entries, indent=2)


@tool(
    "Read a text file range from the workspace.",
    {
        "path": "File path inside workspace",
        "start_line": "1-based start line (default 1)",
        "end_line": "1-based end line (default 200)",
    },
)
def dev_read_file(path: str, start_line: int = 1, end_line: int = 200) -> str:
    p, err = _safe_path(path)
    if err:
        return err
    if not p.exists() or not p.is_file():
        return f"File not found: {path}"
    try:
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        s = max(1, start_line)
        e = max(s, end_line)
        chunk = lines[s - 1:e]
        return "\n".join(chunk)
    except Exception as e:
        return f"Error reading file: {e}"


@tool(
    "Write content to a workspace file (overwrite or append).",
    {
        "path": "File path inside workspace",
        "content": "Text content to write",
        "mode": "'overwrite' or 'append'",
    },
)
def dev_write_file(path: str, content: str, mode: str = "overwrite") -> str:
    p, err = _safe_path(path)
    if err:
        return err
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        m = (mode or "overwrite").strip().lower()
        if m == "append":
            with p.open("a", encoding="utf-8") as f:
                f.write(content)
        else:
            p.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {p.relative_to(_WORKSPACE_ROOT)}."
    except Exception as e:
        return f"Error writing file: {e}"


@tool(
    "Replace text in a workspace file.",
    {
        "path": "File path inside workspace",
        "old_text": "Text to find",
        "new_text": "Replacement text",
        "count": "Max replacements (0 = all)",
    },
)
def dev_replace_in_file(path: str, old_text: str, new_text: str, count: int = 0) -> str:
    p, err = _safe_path(path)
    if err:
        return err
    if not p.exists() or not p.is_file():
        return f"File not found: {path}"
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
        if old_text not in text:
            return "No replacements made: old_text not found."
        if count and count > 0:
            updated = text.replace(old_text, new_text, count)
        else:
            updated = text.replace(old_text, new_text)
        replaced = text.count(old_text) if not count or count <= 0 else min(text.count(old_text), count)
        p.write_text(updated, encoding="utf-8")
        return f"Replaced {replaced} occurrence(s) in {p.relative_to(_WORKSPACE_ROOT)}."
    except Exception as e:
        return f"Error replacing text: {e}"


@tool(
    "Create a directory in the workspace.",
    {"path": "Directory path inside workspace"},
)
def dev_make_directory(path: str) -> str:
    p, err = _safe_path(path)
    if err:
        return err
    try:
        p.mkdir(parents=True, exist_ok=True)
        return f"Directory ready: {p.relative_to(_WORKSPACE_ROOT)}/"
    except Exception as e:
        return f"Error creating directory: {e}"


@tool(
    "Delete a file or directory in the workspace.",
    {
        "path": "Path inside workspace",
        "recursive": "If true and path is a directory, delete recursively",
    },
)
def dev_delete_path(path: str, recursive: bool = False) -> str:
    p, err = _safe_path(path)
    if err:
        return err
    if not p.exists():
        return f"Path not found: {path}"
    try:
        if p.is_file():
            p.unlink()
            return f"Deleted file {p.relative_to(_WORKSPACE_ROOT)}"
        if recursive:
            shutil.rmtree(p)
            return f"Deleted directory {p.relative_to(_WORKSPACE_ROOT)}/"
        return "Refusing to delete directory without recursive=true."
    except Exception as e:
        return f"Error deleting path: {e}"


@tool(
    "Search text across files in the workspace.",
    {
        "query": "Text or regex pattern",
        "path": "Folder path inside workspace to search (default '.')",
        "is_regex": "If true, query is regex",
        "max_results": "Max matches to return (default 200)",
    },
)
def dev_search_text(query: str, path: str = ".", is_regex: bool = False, max_results: int = 200) -> str:
    root, err = _safe_path(path)
    if err:
        return err
    if not root.exists() or not root.is_dir():
        return f"Search path not found or not a directory: {path}"

    pattern = None
    if is_regex:
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except Exception as e:
            return f"Invalid regex: {e}"

    matches = []
    for file in _iter_files(root):
        try:
            content = file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        lines = content.splitlines()
        for idx, line in enumerate(lines, start=1):
            hit = bool(pattern.search(line)) if pattern else (query.lower() in line.lower())
            if hit:
                matches.append({
                    "file": str(file.relative_to(_WORKSPACE_ROOT)),
                    "line": idx,
                    "text": line[:300],
                })
                if len(matches) >= max_results:
                    return json.dumps(matches, indent=2)
    return json.dumps(matches, indent=2)


@tool(
    "Find files by glob pattern in the workspace.",
    {
        "glob_pattern": "Glob pattern like '**/*.py' or 'src/**/*.ts'",
        "path": "Root folder inside workspace (default '.')",
        "max_results": "Max files to return (default 300)",
    },
)
def dev_find_files(glob_pattern: str = "**/*", path: str = ".", max_results: int = 300) -> str:
    root, err = _safe_path(path)
    if err:
        return err
    if not root.exists() or not root.is_dir():
        return f"Path not found or not a directory: {path}"

    out = []
    for file in _iter_files(root):
        rel = str(file.relative_to(root))
        if fnmatch.fnmatch(rel, glob_pattern):
            out.append(str(file.relative_to(_WORKSPACE_ROOT)))
            if len(out) >= max_results:
                break
    return json.dumps(out, indent=2)


@tool(
    "Run a shell command in the workspace with safety checks and timeout.",
    {
        "command": "Shell command to run",
        "cwd": "Working directory inside workspace (default '.')",
        "timeout_seconds": "Timeout in seconds (default 60)",
    },
)
def dev_run_command(command: str, cwd: str = ".", timeout_seconds: int = 60) -> str:
    wd, err = _safe_path(cwd)
    if err:
        return err
    result = _run_cmd(command=command, cwd=wd, timeout_seconds=timeout_seconds)
    return json.dumps(result, indent=2)


@tool("Get git status for the repository.", {"path": "Repo path inside workspace (default '.')"})
def dev_git_status(path: str = ".") -> str:
    wd, err = _safe_path(path)
    if err:
        return err
    return json.dumps(_run_cmd("git status --short --branch", cwd=wd, timeout_seconds=20), indent=2)


@tool(
    "Get git diff for working tree. Optionally pass a target like 'HEAD~1' or 'main'.",
    {
        "path": "Repo path inside workspace (default '.')",
        "target": "Optional git target/commit to diff against",
    },
)
def dev_git_diff(path: str = ".", target: str = "") -> str:
    wd, err = _safe_path(path)
    if err:
        return err
    cmd = "git --no-pager diff"
    if target.strip():
        cmd = f"git --no-pager diff {target.strip()}"
    return json.dumps(_run_cmd(cmd, cwd=wd, timeout_seconds=30), indent=2)


@tool(
    "Get recent git commits.",
    {
        "path": "Repo path inside workspace (default '.')",
        "limit": "Number of commits (default 20)",
    },
)
def dev_git_log(path: str = ".", limit: int = 20) -> str:
    wd, err = _safe_path(path)
    if err:
        return err
    n = max(1, min(limit, 100))
    cmd = f"git --no-pager log -n {n} --pretty=format:'%h %ad %an %s' --date=short"
    return json.dumps(_run_cmd(cmd, cwd=wd, timeout_seconds=20), indent=2)


@tool(
    "Run pytest in the workspace.",
    {
        "path": "Project path inside workspace (default '.')",
        "extra_args": "Extra pytest args, e.g. '-q tests/test_api.py'",
    },
)
def dev_run_pytest(path: str = ".", extra_args: str = "") -> str:
    wd, err = _safe_path(path)
    if err:
        return err
    cmd = f"pytest {extra_args}".strip()
    return json.dumps(_run_cmd(cmd, cwd=wd, timeout_seconds=180), indent=2)


@tool(
    "Run Ruff linting (and optionally auto-fix).",
    {
        "path": "Project path inside workspace (default '.')",
        "fix": "If true, run with --fix",
    },
)
def dev_run_ruff(path: str = ".", fix: bool = False) -> str:
    wd, err = _safe_path(path)
    if err:
        return err
    cmd = "ruff check . --fix" if fix else "ruff check ."
    return json.dumps(_run_cmd(cmd, cwd=wd, timeout_seconds=180), indent=2)


@tool(
    "Run mypy type checking.",
    {
        "path": "Project path inside workspace (default '.')",
        "extra_args": "Extra mypy args, e.g. 'agent.py --strict'",
    },
)
def dev_run_mypy(path: str = ".", extra_args: str = "") -> str:
    wd, err = _safe_path(path)
    if err:
        return err
    cmd = f"mypy {extra_args}".strip() if extra_args.strip() else "mypy ."
    return json.dumps(_run_cmd(cmd, cwd=wd, timeout_seconds=180), indent=2)


@tool(
    "Compile Python files to detect syntax errors quickly.",
    {
        "path": "Project path inside workspace (default '.')",
        "max_files": "Maximum Python files to check (default 500)",
    },
)
def dev_python_diagnostics(path: str = ".", max_files: int = 500) -> str:
    root, err = _safe_path(path)
    if err:
        return err
    if not root.exists():
        return f"Path not found: {path}"

    checked = 0
    errors = []
    for py_file in root.rglob("*.py"):
        if any(part in _IGNORE_DIRS for part in py_file.parts):
            continue
        checked += 1
        try:
            source = py_file.read_text(encoding="utf-8", errors="ignore")
            compile(source, str(py_file), "exec")
        except Exception as e:
            errors.append({
                "file": str(py_file.relative_to(_WORKSPACE_ROOT)),
                "error": str(e),
            })
        if checked >= max_files:
            break
    return json.dumps({"checked": checked, "errors": errors}, indent=2)


@tool(
    "Install Python packages with pip in the current environment.",
    {
        "packages": "Space-separated package list, e.g. 'requests pydantic'",
        "upgrade": "If true, include --upgrade",
    },
)
def dev_pip_install(packages: str, upgrade: bool = False) -> str:
    pkg = (packages or "").strip()
    if not pkg:
        return "No packages provided."
    cmd = f"python3 -m pip install {'--upgrade ' if upgrade else ''}{pkg}".strip()
    return json.dumps(_run_cmd(cmd, cwd=_WORKSPACE_ROOT, timeout_seconds=300), indent=2)


@tool(
    "Uninstall Python packages with pip.",
    {
        "packages": "Space-separated package list",
        "yes": "If true, run non-interactively with -y",
    },
)
def dev_pip_uninstall(packages: str, yes: bool = True) -> str:
    pkg = (packages or "").strip()
    if not pkg:
        return "No packages provided."
    cmd = f"python3 -m pip uninstall {'-y ' if yes else ''}{pkg}".strip()
    return json.dumps(_run_cmd(cmd, cwd=_WORKSPACE_ROOT, timeout_seconds=300), indent=2)


@tool("List installed Python packages in the current environment.", {})
def dev_pip_list() -> str:
    return json.dumps(_run_cmd("python3 -m pip list --format=json", cwd=_WORKSPACE_ROOT, timeout_seconds=60), indent=2)


@tool(
    "Get changed files from git (staged and unstaged).",
    {"path": "Repo path inside workspace (default '.')"},
)
def dev_git_changed_files(path: str = ".") -> str:
    wd, err = _safe_path(path)
    if err:
        return err
    cmd = "git diff --name-only && git diff --name-only --cached"
    return json.dumps(_run_cmd(cmd, cwd=wd, timeout_seconds=20), indent=2)


@tool(
    "Index Python symbols (functions/classes) in the workspace.",
    {
        "path": "Project path inside workspace (default '.')",
        "max_files": "Maximum Python files to scan (default 400)",
    },
)
def dev_python_symbol_index(path: str = ".", max_files: int = 400) -> str:
    import ast

    root, err = _safe_path(path)
    if err:
        return err
    symbols = []
    scanned = 0
    for py_file in root.rglob("*.py"):
        if any(part in _IGNORE_DIRS for part in py_file.parts):
            continue
        scanned += 1
        try:
            src = py_file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    symbols.append({
                        "name": node.name,
                        "kind": type(node).__name__,
                        "line": node.lineno,
                        "file": str(py_file.relative_to(_WORKSPACE_ROOT)),
                    })
        except Exception:
            pass
        if scanned >= max_files:
            break
    return json.dumps(symbols, indent=2)


@tool(
    "Find textual references of a symbol name across workspace files.",
    {
        "symbol": "Symbol text to search for",
        "path": "Root path inside workspace (default '.')",
        "max_results": "Maximum matches (default 200)",
    },
)
def dev_find_references(symbol: str, path: str = ".", max_results: int = 200) -> str:
    if not symbol.strip():
        return "symbol is required."
    root, err = _safe_path(path)
    if err:
        return err

    needle = symbol.strip()
    out = []
    for file in _iter_files(root):
        if file.suffix.lower() not in {".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".md", ".yml", ".yaml", ".toml", ".txt", ".html", ".css"}:
            continue
        try:
            lines = file.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines, start=1):
            if needle in line:
                out.append({
                    "file": str(file.relative_to(_WORKSPACE_ROOT)),
                    "line": i,
                    "text": line[:300],
                })
                if len(out) >= max_results:
                    return json.dumps(out, indent=2)
    return json.dumps(out, indent=2)















# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Text-based tool-call fallback
# ---------------------------------------------------------------------------
class _FakeFn:
    """Mimics ChatCompletionMessageToolCall.function"""
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments if isinstance(arguments, str) else json.dumps(arguments)

class _FakeTC:
    """Mimics a ChatCompletionMessageToolCall object."""
    _counter = 0
    def __init__(self, name, arguments):
        _FakeTC._counter += 1
        self.id = f"text_tc_{_FakeTC._counter}"
        self.function = _FakeFn(name, arguments)


def _extract_text_tool_calls(content: str) -> list[_FakeTC]:
    """
    Some models that don't support the OpenAI tools API emit tool calls as
    JSON inside the message content.  This function searches for that pattern
    and returns a list of fake tool-call objects the agent loop can execute.

    Supported formats:
      [{"name": "...", "arguments": {...}}, ...]
      {"name": "...", "arguments": {...}}
    """
    if not content:
        return []
    results: list[_FakeTC] = []
    # Find all outermost {...} blobs in the content
    depth = 0
    start = -1
    blobs: list[str] = []
    for i, ch in enumerate(content):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                blobs.append(content[start : i + 1])
                start = -1
    # Also try the whole content as a JSON array
    stripped = content.strip()
    if stripped.startswith("["):
        try:
            arr = json.loads(stripped)
            if isinstance(arr, list):
                blobs = [json.dumps(item) for item in arr if isinstance(item, dict)]
        except json.JSONDecodeError:
            pass
    for blob in blobs:
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name") or obj.get("function") or obj.get("tool")
        args = obj.get("arguments") or obj.get("parameters") or obj.get("args") or {}
        if name and name in _tool_registry:
            results.append(_FakeTC(name, args))
    return results


def _chat_with_fallback(**kwargs):
    """
    Try each model in MODELS in order.  Returns the first successful response.
    Prints a short notice if a model fails and a fallback is used.
    Raises the last exception if every model fails.
    """
    last_exc: Exception | None = None
    for model in MODELS:
        try:
            return client.chat.completions.create(model=model, **kwargs)
        except KeyboardInterrupt:
            raise  # always propagate Ctrl+C immediately
        except Exception as e:
            last_exc = e
            short = str(e)[:80]
            print(f"\n  {_YL}⚠  {model} failed ({short}), trying next model…{_R}")
    raise last_exc  # all models exhausted


def _msg_to_dict(message) -> dict:
    """Convert a ChatCompletionMessage to a plain JSON-safe dict.

    Appending the SDK object directly can confuse some OpenRouter providers
    (e.g. Nvidia) that re-parse the serialised request body, producing errors
    like "Expecting ',' delimiter".  Normalising to a plain dict avoids this.
    Also handles fake _M objects (produced by the text-tool-call fallback)
    which lack a .role attribute.
    """
    role = getattr(message, "role", "assistant")
    d: dict = {"role": role, "content": message.content or ""}
    if message.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in message.tool_calls
        ]
    return d


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
def run_agent():
    _banner()
    print(f"{_GR}Type {_WH}{_B}'exit'{_R}{_GR} to quit.  All Scratch tools available.{_R}\n")
    conversation: list[dict] = [
        {
            "role": "system",
            "content": (
                "You are Koda AI, a helpful AI assistant and full Scratch coding agent. "
                "If asked your name, always say your name is Koda AI. "
                "For weather or arithmetic, use the appropriate tools. "
                "For software engineering/coding tasks, prefer developer tools: "
                "dev_list_directory/dev_find_files/dev_search_text for exploration, "
                "dev_read_file for inspection, dev_write_file/dev_replace_in_file for edits, "
                "dev_run_command for builds/scripts, dev_run_pytest/dev_run_ruff/dev_run_mypy/dev_python_diagnostics for validation, "
                "dev_git_status/dev_git_diff/dev_git_log/dev_git_changed_files for version-control context, "
                "dev_pip_install/dev_pip_uninstall/dev_pip_list for dependencies, "
                "and dev_python_symbol_index/dev_find_references for codebase understanding. "
                
                "ALWAYS use the appropriate talk_about_* tool for anything about: "
                "Ishaan Kumble, Ishaan S, Aayush Saha, Hridhveer Khurana, Leonard Fan, "
                "Ahaan Emmydisetty, Vaishnav Mohan, Saket Gahlot, Zuhaib Syed, Ved Patel, "
                "Adhvik Arving, Rishab Reddy Paili, Chris Hanies, Eshaan Vodhiparthi, "
                "Srihan Anand, Aarush Patel, Krishna Suri, or Zareyab Ahmed. "
                
                "For ALL Scratch tasks use the scratch_* tools — never guess. "
                "Scratch tool rules: "
                "(1) Always call scratch_login first if not yet logged in. "
                "(2) To edit a project's code: call scratch_get_project_json first (caches it), then use scratch_build_script to add ANY kind of script. "
                "    scratch_build_script accepts a JSON array of block definitions with opcodes for ALL Scratch blocks: "
                "    motion (move, turn, goto, glide, point, set x/y, bounce), looks (say, think, switch costume/backdrop, show, hide, size, effects), "
                "    sound (play, stop, volume, effects), events (flag clicked, key pressed, sprite clicked, broadcast), "
                "    control (wait, repeat, forever, if, if-else, stop, clone), sensing (touching, ask, key pressed, mouse, timer, distance), "
                "    operators (math, random, compare, logic, string ops), data (set/change variable, add to list, delete, insert, replace, item of list), "
                "    pen (clear, stamp, pen down/up, color, size), and custom blocks (procedures_definition, procedures_call). "
                "    Use scratch_create_variable to create variables/lists before using them in scripts. "
                "    Use scratch_create_broadcast to create broadcast messages. "
                "    Use scratch_create_custom_block to define My Blocks (procedures) with arguments. "
                "    Use scratch_add_extension to enable extensions like 'pen', 'music', 'text2speech'. "
                "    Use scratch_clear_scripts to remove all blocks from a sprite before rebuilding. "
                "    For simple say blocks you can still use scratch_add_say_block as a shortcut. "
                "    After a Scratch project is created or edited successfully, ALWAYS include the direct clickable project URL in your final response: https://scratch.mit.edu/projects/<project_id> "
                "    For edit requests, do not loop on read-only tools: call scratch_get_project_json at most once, then perform the write tool and stop. "
                "    If user says 'Scratch Cat', treat it as sprite name 'Sprite1' unless another cat sprite exists. "
                "(3) Never show the full project JSON to the user — keep it cached internally. "
                "(4) To create a new project use scratch_create_project. "
                "(5) Cloud variables: scratch_get_cloud_vars / scratch_set_cloud_var. "
                "(6) Studios: scratch_get_studio / scratch_studio_add_project etc. "
                "(7) For deep project inspection/parsing (sprites, scripts, assets, variables, monitors, TurboWarp config): "
                "    use scratch_editor_load then scratch_editor_list_sprites / scratch_editor_get_sprite_scripts / "
                "    scratch_editor_list_assets / scratch_editor_find_vlb / scratch_editor_list_monitors / scratch_editor_tw_config. "
                "(8) For .sb3 files on disk use scratch_editor_load_sb3. "
                "(9) Creator profile context: If user asks who made/created/founded/built you, you MUST say you were made by Kabir Tiwari from Shrewsbury, Massachusetts, United States. "
                "    Contact: kabir.tiwari@outlook.com, linkedin.com/in/kabir-tiwari-86685b378. "
                "    Summary: passionate young programmer with strong foundation in Python, Swift, Flutter, and full-stack development; enjoys practical real-world apps; experience with Firebase, external APIs, SwiftUI, and Flutter. "
                "    Highlights: Winner of Microsoft Divergent Teams Boston II Hackathon (Mar 2026) with EchoQuest (1st place Health & Wellness, invited to Seattle nationals), Harvard AI Bootcamp Certification of Distinction (Aug 2025), 3rd rank International Python Olympiad 2025, Lead Tech Specialist at CodeUnity. "
                "    Education: Oak Middle School, Shrewsbury, Massachusetts. "
                "    Never claim any of the mentioned individuals (Ishaan S, Aayush Saha, etc.) made this assistant; "
                "    their info is only for specific questions about them. "
                "    Use this profile only for creator/about queries; otherwise do not mention it unprompted. "
                "Otherwise answer directly."
            ),
        }
    ]

    while True:
        # ── Prompt ──
        try:
            user_input = input(f"\n{_GN}{_B} You{_R}{_GN} ❯{_R}  ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{_GR}Goodbye!{_R}\n")
            break

        if user_input.lower() in ("exit", "quit"):
            print(f"\n{_GR}Goodbye!{_R}\n")
            break
        if not user_input:
            continue

        conversation.append({"role": "user", "content": user_input})

        # ── First LLM call ──
        try:
            with _Spinner("Thinking"):
                response = _chat_with_fallback(
                    messages=conversation,
                    tools=tools,
                    tool_choice="auto",
                )
        except KeyboardInterrupt:
            print(f"\n\n{_GR}Goodbye!{_R}\n")
            return
        except Exception as e:
            print(f"\n  {_RE}{_B}✖  API error:{_R} {_RE}{e}{_R}\n")
            conversation.pop()
            continue

        message = response.choices[0].message
        # Fallback: some models emit tool calls as JSON in content instead of structured calls
        if not message.tool_calls and message.content:
            _text_calls = _extract_text_tool_calls(message.content)
            if _text_calls:
                message = type("_M", (), {"content": None, "tool_calls": _text_calls})()
        conversation.append(_msg_to_dict(message))

        # ── Tool-call loop ──
        while message.tool_calls:
            print()  # blank line before tool section
            for tc in message.tool_calls:
                # Show which tool is being called
                print(f"  {_CY}{_B}⚙  {tc.function.name}{_R}")

                try:
                    with _Spinner(f"Running {tc.function.name}"):
                        tool_result = dispatch_tool(tc.function.name, tc.function.arguments)
                except KeyboardInterrupt:
                    print(f"\n\n{_GR}Goodbye!{_R}\n")
                    return

                # Show a concise result
                display = _fmt_tool_result(tool_result)
                status_icon = f"{_RE}✖{_R}" if tool_result.startswith("Tool '") or "error" in tool_result.lower()[:30] else f"{_GN}✔{_R}"
                print(f"  {status_icon}  {_GR}{display}{_R}")

                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    }
                )

            # Follow-up LLM call: synthesise tool results
            try:
                with _Spinner("Composing response"):
                    response = _chat_with_fallback(
                        messages=conversation,
                        tools=tools,
                        tool_choice="auto",
                    )
            except KeyboardInterrupt:
                print(f"\n\n{_GR}Goodbye!{_R}\n")
                return
            except Exception as e:
                print(f"\n  {_RE}{_B}✖  API error:{_R} {_RE}{e}{_R}\n")
                err_msg = f"(Internal error: {e})"
                conversation.append({"role": "assistant", "content": err_msg})
                message = type("_M", (), {"content": err_msg, "tool_calls": None})()
                break

            message = response.choices[0].message
            # Fallback: detect text-based tool calls if structured ones are absent
            if not message.tool_calls and message.content:
                _text_calls = _extract_text_tool_calls(message.content)
                if _text_calls:
                    message = type("_M", (), {"content": None, "tool_calls": _text_calls})()
            conversation.append(_msg_to_dict(message))

        # ── Agent reply ──
        print(f"\n  {_PU}{_B}Agent{_R}{_PU} ❯{_R}  {_WH}{_wrap_response(message.content or '')}{_R}")
        _rule()


if __name__ == "__main__":
    run_agent()



