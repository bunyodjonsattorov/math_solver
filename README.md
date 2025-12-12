# Math Solver (A-Level) – Setup Notes

## Prerequisites
- Python 3.9+
- An OpenAI API key (`OPENAI_API_KEY`)

## Local Setup
1. Create/activate your venv.
2. Install deps (already in venv): `pip install -r requirements.txt` (or reuse existing venv).
3. Set your key:
   ```bash
   export OPENAI_API_KEY=sk-...
   ```
4. Run the UI:
   ```bash
   streamlit run app.py
   ```

## Streamlit Cloud
1. In `Secrets`, add:
   ```
   OPENAI_API_KEY=sk-...
   ```
2. Deploy `streamlit run app.py`.

## Files
- `agent.py` — builds the LangChain agent (Python tool, GPT-4o-mini).
- `app.py` — Streamlit premium UI (chat, vision upload, shows Python steps).
- `main.py` — CLI entrypoint (optional).
- `config.py` — reads `OPENAI_API_KEY` from env only.

