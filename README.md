# Basic RAG — HR Policy Assistant

A small Retrieval-Augmented Generation (RAG) demo that turns a company HR policy document into a searchable assistant. It uses chunking + embeddings + a FAISS vector store to retrieve relevant policy passages, then answers user questions with a language model.

This repository includes:

- hr_assistant/: core modules that load and split documents, build a FAISS vector store, and wire up a LangGraph/LangChain-based agent.
- data/hr_policy.txt: sample HR policy document used as the knowledge source.
- data/faiss_index/: saved FAISS index created from the HR policy (auto-created when the index is built).
- main.py: simple CLI demo that builds the assistant and asks a few example questions.
- app.py: Streamlit chat UI to interact with the assistant in the browser.

Key files to inspect:

- [hr_assistant/pipeline.py] — wires components and exposes build_hr_assistant() and ask().
- [hr_assistant/vector_store.py] — functions to build/load/save FAISS vector store.
- [hr_assistant/config.py] — project configuration and required environment variables.

---

Why this project exists

This project demonstrates how to build a practical RAG assistant for a domain document (HR policy) so users can ask factual questions and get answers grounded in the source text.

---

Requirements

- Python 3.10+ (recommended)
- A virtual environment (venv)
- The following Python packages (also listed in requirements.txt):
  - python-dotenv
  - langchain
  - langchain-core
  - langchain-community
  - langchain-groq (or other LLM provider integration you're using)
  - langchain-text-splitters
  - faiss-cpu
  - streamlit
  - jupyter, ipykernel (optional)

Install steps (Windows example)

1. Create and activate a virtualenv

```powershell
python -m venv basicragenv
basicragenv\Scripts\activate
```

2. Install dependencies

```powershell
pip install -r requirements.txt
```

3. Create a .env file in the repo root with the required API keys (see below).

Environment variables

Create a .env file in the repository root with the following variables (the names match [hr_assistant/config.py]).

```
GROQ_API_KEY=your_groq_api_key_here
JINA_API_KEY=your_jina_api_key_here
```

Note: config.py currently checks for GROQ_API_KEY and JINA_API_KEY. If you use a different LLM/embedding provider (e.g., OpenAI), update config.py and the environment variables accordingly.

Data

- Place your HR policy text at data/hr_policy.txt. The repo includes a sample file path — the loader will read and split this file into chunks for embedding.
- The FAISS index is saved under data/faiss_index by default. To force rebuilding the index, remove that directory and re-run the script.

Usage

- CLI demo

```powershell
python main.py
```

This builds the assistant (or loads the saved FAISS index if present) and runs a few demo questions.

- Streamlit UI

```powershell
streamlit run app.py
```

Opens a chat UI in your browser where you can ask questions about the HR policy.

Development notes

- The pipeline wires components in [hr_assistant/pipeline.py]. Review it to customize chunk size, model names, or retrieval k (TOP_K_RESULTS).
- Embeddings are created with the model configured in [hr_assistant/config.py].

Contributing

Contributions welcome. Suggested workflow:

1. Create an issue describing the change.
2. Open a branch, implement changes, run tests (if any), and submit a PR.

---
