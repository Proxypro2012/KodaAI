"""
Web frontend for the AI Agent — Claude-AI themed UI.
Reuses the tool system from agent.py.
"""

import json
import os
import uuid
import time
import openai
from flask import Flask, request, jsonify, send_from_directory

# Import everything we need from the agent module
from agent import (
    client, MODELS, tools, dispatch_tool,
    _msg_to_dict, _extract_text_tool_calls,
    get_openrouter_key_diagnostics, OPENROUTER_API_KEY,
)

app = Flask(__name__, static_folder="static")


@app.errorhandler(Exception)
def _handle_unexpected_error(e):
    return jsonify({"error": f"Server error: {e}"}), 500


@app.after_request
def _add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    return response

# ── Per-session conversations (keyed by session id) ──────────────────────────
_conversations: dict[str, list[dict]] = {}

SYSTEM_PROMPT = (
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
    "ALWAYS use talk_about_ishaan for anything about Ishaan Kumble. "
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
    "    Never claim Ishaan Kumble made this assistant; Ishaan info is only for Ishaan-specific questions. "
    "    Use this profile only for creator/about queries; otherwise do not mention it unprompted. "
    "Otherwise answer directly."
)


def _get_conversation(session_id: str) -> list[dict]:
    if session_id not in _conversations:
        _conversations[session_id] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
    return _conversations[session_id]


def _is_auth_error(exc: Exception) -> bool:
    """Return True if *exc* is an authentication/authorisation failure."""
    return isinstance(exc, openai.AuthenticationError)


def _chat_with_fallback(**kwargs):
    """Try each model in order — mirrors agent.py's logic."""
    last_exc = None
    for model in MODELS:
        try:
            return client.chat.completions.create(model=model, **kwargs)
        except Exception as e:
            # Authentication errors affect all models — stop immediately
            if _is_auth_error(e):
                raise
            last_exc = e
    raise last_exc


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return ("", 204)

    data = request.get_json(silent=True) or {}
    user_msg = data.get("message", "").strip()
    session_id = data.get("session_id", str(uuid.uuid4()))

    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    if not OPENROUTER_API_KEY or not OPENROUTER_API_KEY.strip():
        return jsonify({
            "error": (
                "OpenRouter API key is not configured. "
                "Please set the OPENROUTER_API_KEY environment variable "
                "with a valid key from https://openrouter.ai/keys"
            ),
            "session_id": session_id,
        }), 503

    conversation = _get_conversation(session_id)
    conversation.append({"role": "user", "content": user_msg})

    tool_log: list[dict] = []  # collect tool calls for the frontend

    try:
        response = _chat_with_fallback(
            messages=conversation,
            tools=tools,
            tool_choice="auto",
        )
    except Exception as e:
        conversation.pop()
        if _is_auth_error(e):
            return jsonify({
                "error": (
                    "Authentication failed (401): your OPENROUTER_API_KEY is invalid or expired. "
                    "Please update it at https://openrouter.ai/keys"
                ),
                "session_id": session_id,
            }), 401
        return jsonify({"error": str(e), "session_id": session_id}), 502

    message = response.choices[0].message

    # Fallback: text-based tool calls
    if not message.tool_calls and message.content:
        _text_calls = _extract_text_tool_calls(message.content)
        if _text_calls:
            message = type("_M", (), {"content": None, "tool_calls": _text_calls})()

    conversation.append(_msg_to_dict(message))

    # ── Tool-call loop ──
    max_rounds = 10
    round_count = 0
    while message.tool_calls and round_count < max_rounds:
        round_count += 1
        for tc in message.tool_calls:
            t0 = time.time()
            result = dispatch_tool(tc.function.name, tc.function.arguments)
            elapsed = round(time.time() - t0, 2)
            is_error = result.startswith("Tool '") or "error" in result.lower()[:30]
            tool_log.append({
                "name": tc.function.name,
                "args": tc.function.arguments,
                "result": result[:500],
                "error": is_error,
                "time": elapsed,
            })
            conversation.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        try:
            response = _chat_with_fallback(
                messages=conversation,
                tools=tools,
                tool_choice="auto",
            )
        except Exception as e:
            err = f"(API error: {e})"
            conversation.append({"role": "assistant", "content": err})
            return jsonify({
                "reply": err,
                "tool_log": tool_log,
                "session_id": session_id,
            })

        message = response.choices[0].message
        if not message.tool_calls and message.content:
            _text_calls = _extract_text_tool_calls(message.content)
            if _text_calls:
                message = type("_M", (), {"content": None, "tool_calls": _text_calls})()
        conversation.append(_msg_to_dict(message))

    return jsonify({
        "reply": message.content or "",
        "tool_log": tool_log,
        "session_id": session_id,
        "model": getattr(response, "model", MODELS[0]),
    })


@app.route("/api/models", methods=["GET"])
def get_models():
    return jsonify({"models": MODELS})


@app.route("/api/tools", methods=["GET"])
def get_tools():
    tool_names = [t["function"]["name"] for t in tools]
    return jsonify({"count": len(tool_names), "tools": tool_names})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


@app.route("/api/diag", methods=["GET"])
def diag():
    return jsonify({
        "ok": True,
        "models": MODELS,
        "openrouter": get_openrouter_key_diagnostics(),
    })


@app.route("/api/reset", methods=["POST", "OPTIONS"])
def reset():
    if request.method == "OPTIONS":
        return ("", 204)

    data = request.json or {}
    sid = data.get("session_id")
    if sid and sid in _conversations:
        del _conversations[sid]
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    print(f"\n  🌐  Koda AI Web UI → http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
