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
import time
import threading
import itertools
import shutil
import textwrap
import urllib.request
import urllib.parse
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
    "qwen/qwen3-coder:free",
    "arcee-ai/trinity-large-preview:free",
    "openai/gpt-oss-120b:free",
    "qwen/qwen3-coder:free",
]

# Scratch session — populated by scratch_login tool
_scratch_session = None
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
    "Retrieve facts about Ishaan Subramanian. ALWAYS call this tool whenever the user mentions or asks anything about Ishaan Subramanian.",
    {"topic": "The specific topic about Ishaan Subramanian (e.g. '', 'football'). Use 'general' if no specific topic is mentioned."}
)
def talk_about_ishaan_subramanian(topic: str = "general") -> str:
    facts = {
        "general": (
            "Ishaan Subramanian is a very smart and intelligent kid who is smart even without going to RSM. "
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

# ── Authentication ──────────────────────────────────────────────────────────

@tool(
    "Log in to Scratch. Must be called before any Scratch tools that require authentication.",
    {"username": "Scratch username", "password": "Scratch password"},
)
def scratch_login(username: str, password: str) -> str:
    global _scratch_session
    if scratch3 is None:
        return "scratchattach is not installed. Run: pip install scratchattach"
    try:
        import warnings
        warnings.filterwarnings("ignore", category=scratch3.LoginDataWarning)
        _scratch_session = scratch3.login(username, password)
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
        return json.dumps({
            "project_id": project_id,
            "sprite_count": len(targets),
            "sprites": [{"name": t["name"], "block_count": len(t.get("blocks", {}))} for t in targets],
            "note": "Full JSON cached. Use scratch_add_say_block or scratch_set_project_json to make changes.",
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
        project.set_json(data)   # set_json accepts a dict per the docs
        _project_json_cache[str(project_id)] = data
        return f"Project {project_id} updated successfully."
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"
    except Exception as e:
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
        url = getattr(p, "url", "")
        payload = {"id": pid, "title": ptitle, "url": url}
        if note:
            payload["note"] = note
        return json.dumps(payload, indent=2)

    # Attempt 1: direct API create with built-in template
    project, first_msg = _create_via_api(project_json=_default_project_json())
    if project:
        return _project_info(project)

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

    return (
        f"Error: Could not create project '{clean_title}'. "
        f"Initial create failed with: {first_msg}. "
        f"Template fallback also failed with: {third_msg}. "
        "Try again in a few seconds, or create one project manually on Scratch first and retry."
    )

@tool("Share a Scratch project.", {"project_id": "Numeric Scratch project ID"})
def scratch_share_project(project_id: str) -> str:
    err = _require_scratch()
    if err: return err
    try:
        _scratch_session.connect_project(project_id).share()
        return f"Project {project_id} shared."
    except Exception as e:
        return f"Error: {e}"

@tool("Unshare a Scratch project.", {"project_id": "Numeric Scratch project ID"})
def scratch_unshare_project(project_id: str) -> str:
    err = _require_scratch()
    if err: return err
    try:
        _scratch_session.connect_project(project_id).unshare()
        return f"Project {project_id} unshared."
    except Exception as e:
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

# ── Project JSON editing helper ─────────────────────────────────────────────

@tool(
    "Add a script to a sprite that says a message on trigger. "
    "You MUST call scratch_get_project_json first to cache the project.",
    {
        "project_id": "Numeric Scratch project ID",
        "sprite_name": "Sprite name exactly as shown in scratch_get_project_json (e.g. 'Sprite1')",
        "message": "The message the sprite will say",
        "say_for_seconds": "Seconds to display the message (0 = permanent until next say block)",
        "trigger": "Trigger mode: 'flag' (when green flag clicked) or 'clicked' (when this sprite clicked)",
    },
)
def scratch_add_say_block(project_id: str, sprite_name: str, message: str, say_for_seconds: int = 0, trigger: str = "flag") -> str:
    err = _require_scratch()
    if err: return err
    try:
        import uuid
        cached = _project_json_cache.get(str(project_id))
        if not cached:
            return "Project JSON not cached. Call scratch_get_project_json first."
        targets = cached.get("targets", [])
        target = next((t for t in targets if t["name"].lower() == sprite_name.lower()), None)
        if not target and sprite_name.strip().lower() in {"scratch cat", "cat", "scratchcat"}:
            target = next((t for t in targets if t.get("name", "").lower() == "sprite1"), None)
            if target:
                sprite_name = target.get("name", sprite_name)
        if not target:
            available = [t["name"] for t in targets]
            return f"Sprite '{sprite_name}' not found. Available: {available}"
        flag_id = str(uuid.uuid4()).replace("-", "")[:20]
        say_id  = str(uuid.uuid4()).replace("-", "")[:20]
        opcode  = "looks_sayforsecs" if say_for_seconds > 0 else "looks_say"
        trig = (trigger or "flag").strip().lower()
        event_opcode = "event_whenthisspriteclicked" if trig in {"clicked", "click", "sprite_clicked", "when_clicked"} else "event_whenflagclicked"
        target["blocks"][flag_id] = {
            "opcode": event_opcode, "next": say_id, "parent": None,
            "inputs": {}, "fields": {}, "shadow": False, "topLevel": True, "x": 0, "y": 0,
        }
        target["blocks"][say_id] = {
            "opcode": opcode, "next": None, "parent": flag_id,
            "inputs": {"MESSAGE": [1, [10, message]], **({"SECS": [1, [4, str(say_for_seconds)]]} if say_for_seconds > 0 else {})},
            "fields": {}, "shadow": False, "topLevel": False,
        }
        _scratch_session.connect_project(project_id).set_json(cached)
        _project_json_cache[str(project_id)] = cached
        trig_text = "when this sprite clicked" if event_opcode == "event_whenthisspriteclicked" else "when green flag clicked"
        return f"DONE: Added script ({trig_text} → say '{message}') to '{sprite_name}' in project {project_id}. Edit committed to Scratch."
    except Exception as e:
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
                
                "ALWAYS use the appropriate talk_about_* tool for anything about: "
                "Ishaan Kumble, Ishaan Subramanian, Aayush Saha, Hridhveer Khurana, Leonard Fan, "
                "Ahaan Emmydisetty, Vaishnav Mohan, Saket Gahlot, Zuhaib Syed, Ved Patel, "
                "Adhvik Arving, Rishab Reddy Paili, Chris Hanies, Eshaan Vodhiparthi, "
                "Srihan Anand, Aarush Patel, Krishna Suri, or Zareyab Ahmed. "
                
                "For ALL Scratch tasks use the scratch_* tools — never guess. "
                "Scratch tool rules: "
                "(1) Always call scratch_login first if not yet logged in. "
                "(2) To edit a project's code/JSON: call scratch_get_project_json first (caches it), then use scratch_add_say_block or scratch_set_project_json. "
                "    For edit requests, do not loop on read-only tools: call scratch_get_project_json at most once, then perform the write tool and stop. "
                "    If user asks for 'when clicked', use trigger='clicked' in scratch_add_say_block. "
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
                "    Never claim any of the mentioned individuals (Ishaan Subramanian, Aayush Saha, etc.) made this assistant; "
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



