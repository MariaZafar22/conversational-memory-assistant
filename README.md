# AI Agent with Memory

**A Streamlit chat assistant that remembers. It keeps the current conversation in short-term memory and saves useful facts about the user in long-term memory, so it can bring them up in a brand-new chat.**

Built with Python, Streamlit, Google Gemini and SQLite.

---

## **Features**

- **Short-term memory:** the last 10 messages of the current chat are sent to the LLM on every turn.
- **Long-term memory:** useful facts (name, city, studies, goals, projects) are extracted by Gemini and stored in a local SQLite database.
- **Memory retrieval:** before each reply, the most relevant memories are looked up and added to the prompt. The chat shows which ones were used.
- **Memory manager:** view all stored memories, add one manually, delete one, or wipe everything.
- **New chat without forgetting:** clearing the conversation keeps long-term memory intact.
- **Privacy aware:** passwords, API keys, card numbers, ID numbers and other sensitive data are never saved.

---

## **How It Works**

```
User message
   |
   v
Retrieve relevant memories (SQLite)
   |
   v
Prompt = memories + last 10 messages  -->  Gemini  -->  Reply
   |
   v
Extract new facts from the message  -->  Save to SQLite
```

| Memory type | Where it lives | Lifetime |
|---|---|---|
| Short-term | `st.session_state` | Current chat only |
| Long-term | `memory.db` (SQLite) | Until you delete it |

**Retrieval:** memories are ranked by keyword overlap with the new message. When there are 5 or fewer memories, all of them are included.

**Context window:** only the last 10 messages are sent to the model, which keeps prompts small and shows why long-term memory is needed.

---

## **Project Structure**

```
memory-ai-agent/
├── app.py              # Full app: UI, memory DB, retrieval, Gemini calls
├── requirements.txt
├── .env.example        # Copy to .env and add your key
├── .gitignore
└── README.md
```

---

## **Setup**

**1. Clone and install**

```bash
git clone https://github.com/MariaZafar22/ai-agent-memory.git
cd ai-agent-memory
pip install -r requirements.txt
```

**2. Add your Gemini API key**

```bash
cp .env.example .env
```

Then open `.env` and set:

```
GEMINI_API_KEY=your_key_here
```

You can also skip this and paste the key into the app sidebar.

**3. Run**

```bash
streamlit run app.py
```

If your model name is not available, change `MODEL` at the top of `app.py`.

---

## **Try It**

1. Send: `My name is Maria and I study Computer Science in Faisalabad`
2. Send: `What is my name?` (short-term memory)
3. Open the **Memory Manager** tab and see the saved facts.
4. Click **New Chat**, then ask: `Where do I study?` (long-term memory)
5. Delete a memory with the trash icon and ask again to confirm it is forgotten.

---

## **Privacy**

- Memories are stored only in a local `memory.db` file and are never uploaded.
- `memory.db` and `.env` are in `.gitignore`.
- Every memory can be deleted individually, or all at once.

---

## **Ideas to Extend**

- Embedding-based (semantic) memory retrieval
- Memory categories and importance scores
- Auto-summarising old messages instead of dropping them
- Per-user memory with login

---

Built by **Maria Zafar**. Small memories, big difference.
