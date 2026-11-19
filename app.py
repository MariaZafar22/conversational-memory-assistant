import os
import re
import json
import time
import sqlite3
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL_OVERRIDE = os.getenv("GEMINI_MODEL", "")  # optional: force one model name in .env
DB_PATH = "memory.db"
MAX_HISTORY = 10             # short-term memory: only the last 10 messages are sent to the LLM
TOP_K = 5                    # retrieval: at most 5 memories

STOPWORDS = {
    "the", "a", "an", "is", "are", "am", "i", "my", "me", "you", "your", "to", "of",
    "and", "in", "on", "for", "do", "does", "what", "which", "who", "how", "can",
    "user", "it", "that",
}


# ---------------- Long-term memory (SQLite) ----------------
def get_con():
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """CREATE TABLE IF NOT EXISTS memories (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               content TEXT UNIQUE,
               created_at TEXT)"""
    )
    return con


def all_memories():
    with get_con() as con:
        return con.execute(
            "SELECT id, content, created_at FROM memories ORDER BY id DESC"
        ).fetchall()


def add_memory(text):
    text = text.strip()
    if not text:
        return False
    try:
        with get_con() as con:
            con.execute(
                "INSERT INTO memories (content, created_at) VALUES (?, ?)",
                (text, datetime.now().strftime("%Y-%m-%d %H:%M")),
            )
        return True
    except sqlite3.IntegrityError:  # duplicate memory
        return False


def delete_memory(mem_id):
    with get_con() as con:
        con.execute("DELETE FROM memories WHERE id = ?", (mem_id,))


def clear_memories():
    with get_con() as con:
        con.execute("DELETE FROM memories")


# ---------------- Retrieval ----------------
def tokens(text):
    return {w for w in re.findall(r"[a-zA-Z0-9]+", text.lower()) if w not in STOPWORDS}


def retrieve(query):
    """Find relevant memories by keyword overlap.
    If there are few memories (<= TOP_K), return all of them."""
    mems = all_memories()
    q = tokens(query)
    scored = [(len(q & tokens(m[1])), m) for m in mems]
    if len(mems) <= TOP_K:
        return sorted(scored, key=lambda x: -x[0])
    scored = [s for s in scored if s[0] > 0]
    return sorted(scored, key=lambda x: -x[0])[:TOP_K]


# ---------------- LLM helpers ----------------
def get_client():
    key = st.session_state.get("api_key") or os.getenv("GEMINI_API_KEY")
    return genai.Client(api_key=key) if key else None


def pick_models(names):
    """Choose usable text models from the names your API key can access.
    Prefers newest 'flash' models; skips image/audio/embedding/live variants."""
    skip = ("image", "tts", "audio", "live", "embedding", "robotics", "computer", "veo", "imagen")
    scored = []
    for n in names:
        n = n.replace("models/", "")
        if "gemini" not in n or "flash" not in n or any(k in n for k in skip):
            continue
        m = re.search(r"gemini-(\d+(?:\.\d+)?)", n)
        version = float(m.group(1)) if m else 0.0
        stable = 0 if ("preview" in n or "exp" in n) else 1
        lite = 0 if "lite" in n else 1
        scored.append((stable, version, lite, n))
    scored.sort(reverse=True)
    return [x[3] for x in scored[:4]]


def available_models(client):
    """Ask the API which models this key can use (cached for the session)."""
    if MODEL_OVERRIDE:
        return [MODEL_OVERRIDE]
    if st.session_state.get("models"):
        return st.session_state.models
    names = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None)
        if actions is None or "generateContent" in actions:
            names.append(m.name)
    models = pick_models(names)
    if not models:
        raise RuntimeError("No usable Gemini model found for this API key.")
    st.session_state.models = models
    return models


def call_gemini(client, contents, config):
    """Call Gemini. Models are discovered automatically.
    - busy / network error: wait and retry
    - model not available (404): move to the next model"""
    last_error = None
    for model in list(available_models(client)):
        for attempt in range(3):
            try:
                resp = client.models.generate_content(
                    model=model, contents=contents, config=config
                )
                st.session_state.active_model = model
                return resp
            except Exception as e:
                last_error = e
                msg = str(e)
                if any(c in msg for c in ("404", "NOT_FOUND", "no longer available")):
                    # remember this model is unusable so we skip it next time
                    if st.session_state.get("models") and model in st.session_state.models:
                        st.session_state.models.remove(model)
                    break  # try next model
                transient = any(
                    c in msg
                    for c in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED",
                              "getaddrinfo", "11001", "Connection", "timed out")
                )
                if not transient:
                    raise
                time.sleep(2 * (attempt + 1))
    raise last_error


def chat_reply(client, history, memories):
    if memories:
        mem_text = "\n".join(f"- {m[1][1]}" for m in memories)
    else:
        mem_text = "(no long-term memories yet)"
    system = (
        "You are a friendly AI assistant with memory. Reply in the same language "
        "as the user (English / Urdu / Roman Urdu), keep it short.\n"
        "Known facts about the user from past conversations:\n"
        f"{mem_text}\n"
        "Use these facts naturally when relevant. Do not invent facts."
    )
    contents = [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[types.Part(text=m["content"])],
        )
        for m in history
    ]
    resp = call_gemini(
        client,
        contents,
        types.GenerateContentConfig(system_instruction=system),
    )
    return resp.text or "(empty reply)"


def extract_memories(client, user_msg):
    """Decide what is worth remembering."""
    prompt = (
        "Extract long-term useful facts about the user from this message "
        "(name, city, studies, job, preferences, goals, projects). "
        "NEVER store passwords, API keys, card numbers, ID numbers, health or "
        "other sensitive data. Ignore questions and small talk. "
        'Return ONLY a JSON list of short strings like ["User\'s name is Ali"]. '
        "If nothing is worth saving return [].\n\n"
        f"Message: {user_msg}"
    )
    try:
        resp = call_gemini(
            client,
            prompt,
            types.GenerateContentConfig(response_mime_type="application/json"),
        )
        facts = json.loads(resp.text)
        return [f for f in facts if isinstance(f, str)]
    except Exception:
        return []


# ---------------- UI ----------------
st.set_page_config(page_title="AI Agent with Memory", page_icon="🧠", layout="wide")
st.title("🧠 AI Agent with Memory")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("⚙️ Settings")
    if not os.getenv("GEMINI_API_KEY"):
        st.text_input("Gemini API Key", type="password", key="api_key")
    st.metric("Short-term messages", len(st.session_state.messages))
    st.metric("Long-term memories", len(all_memories()))
    if st.session_state.get("active_model"):
        st.caption(f"Model in use: {st.session_state.active_model}")
    st.caption(f"Only the last {MAX_HISTORY} messages are sent to the LLM (context window limit).")
    if st.button("🆕 New Chat (short-term clear)", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption("New Chat clears only the current conversation. Long-term memory stays safe.")
    st.info("🔒 Privacy: passwords, cards and ID numbers are never saved. You can delete any memory at any time.")

tab_chat, tab_mem = st.tabs(["💬 Chat", "🗂️ Memory Manager"])

with tab_chat:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("used"):
                with st.expander("🔍 Retrieved memories used"):
                    for score, mem in m["used"]:
                        st.write(f"• {mem[1]}  _(match score: {score})_")
            if m.get("saved"):
                st.success("💾 Saved to long-term memory: " + " | ".join(m["saved"]))

    if prompt := st.chat_input("Type a message... (e.g. My name is Maria)"):
        client = get_client()
        if not client:
            st.error("Please enter your Gemini API key in the sidebar first.")
            st.stop()

        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        used = retrieve(prompt)  # 1) fetch relevant long-term memories
        history = st.session_state.messages[-MAX_HISTORY:]  # 2) short-term memory (recent messages)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    reply = chat_reply(client, history, used)
                except Exception as e:
                    if "getaddrinfo" in str(e) or "11001" in str(e):
                        st.error("Network problem: cannot reach Gemini. Check your internet / DNS or VPN and try again.")
                    else:
                        st.error(f"Gemini error: {e}")
                    st.stop()
                facts = extract_memories(client, prompt)  # 3) decide what to remember
                saved = [f for f in facts if add_memory(f)]
            st.markdown(reply)
            if used:
                with st.expander("🔍 Retrieved memories used"):
                    for score, mem in used:
                        st.write(f"• {mem[1]}  _(match score: {score})_")
            if saved:
                st.success("💾 Saved to long-term memory: " + " | ".join(saved))

        st.session_state.messages.append(
            {"role": "assistant", "content": reply, "used": used, "saved": saved}
        )
        st.rerun()

with tab_mem:
    st.subheader("Stored long-term memories")
    new_mem = st.text_input("➕ Add a memory manually", key="manual_mem")
    if st.button("Add memory"):
        if add_memory(new_mem):
            st.rerun()
        else:
            st.warning("Empty or duplicate memory.")

    mems = all_memories()
    if not mems:
        st.info("No memories saved yet.")
    for mem_id, content, created in mems:
        c1, c2 = st.columns([8, 1])
        c1.write(f"**{content}**  \n<small>🕒 {created}</small>", unsafe_allow_html=True)
        if c2.button("🗑️", key=f"del_{mem_id}", help="Delete this memory"):
            delete_memory(mem_id)
            st.rerun()
    if mems:
        st.divider()
        if st.button("⚠️ Delete ALL memories"):
            clear_memories()
            st.rerun()