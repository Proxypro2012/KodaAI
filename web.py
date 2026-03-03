"""
Web frontend for the AI Agent — Claude-AI themed UI.
Reuses the tool system from agent.py.
"""

import json
import os
import uuid
import time
from flask import Flask, request, jsonify, send_from_directory

# Import everything we need from the agent module
from agent import (
    client, MODELS, tools, dispatch_tool,
    _msg_to_dict, _extract_text_tool_calls,
    get_openrouter_key_diagnostics,
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
    "ALWAYS use talk_about_ishaan for anything about Ishaan Kumble. "
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


def _chat_with_fallback(**kwargs):
    """Try each model in order — mirrors agent.py's logic."""
    last_exc = None
    for model in MODELS:
        try:
            return client.chat.completions.create(model=model, **kwargs)
        except Exception as e:
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
