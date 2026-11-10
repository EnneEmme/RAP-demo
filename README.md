# RAP Demo

Implementazione di **Reasoning via Planning (RAP)** con **MCTS** su Blocksworld, supporto **Hybrid / Paper mode**, modelli locali (`llama-server`), API esterne (`litellm`) e dashboard Web in tempo reale.

---

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

*(Su Python 3.15, se necessario per compilare `fastuuid`: `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 uv pip install -e ".[dev]"`)*

---

## Configurazione API (`.env`)

Per usare provider esterni con `--backend litellm`, copia `.env.example` in `.env` e inserisci la tua API key:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
# GEMINI_API_KEY=...
# OPENROUTER_API_KEY=...
# GROQ_API_KEY=...
```

Il file `.env` viene caricato automaticamente all'avvio.

---

## Utilizzo CLI

### 1. Esecuzione da terminale (`rap run`)

```bash
# Test rapido offline (MockLLM)
rap run --iterations 10 -v

# API esterne via LiteLLM (OpenAI, Anthropic, Gemini, ecc.)
rap run --backend litellm --model openai/gpt-4o-mini --iterations 10 -v
rap run --backend litellm --model anthropic/claude-3-5-sonnet-20241022 --iterations 10 -v

# Endpoint OpenAI-compatible custom
rap run --backend litellm --model openai/my-model --api-base https://api.example.com/v1

# Modello locale con llama.cpp
llama-server -hf saidutta69/LFM3-VL-2B-heretic --port 8080 -ngl 99
rap run --backend llama-server --iterations 10 -v
```

### 2. Dashboard Web (`rap ui`)

Avvia il server e apri `http://127.0.0.1:8000` nel browser:

```bash
rap ui --backend mock
rap ui --backend litellm --model gpt-4o-mini --port 8000
rap ui --backend llama-server --port 8000
```

### 3. Confronto Hybrid vs Paper (`rap compare`)

```bash
rap compare --backend mock --iterations 10
rap compare --backend litellm --model gpt-4o-mini --iterations 10
```

---

## Flag Principali

| Flag | Valori | Default | Descrizione |
| :--- | :--- | :--- | :--- |
| `--backend` | `mock`, `litellm`, `llama-server` | `mock` | Backend LLM |
| `--model` | es. `gpt-4o-mini`, `gemini/gemini-2.0-flash` | `None` | Modello per LiteLLM |
| `--api-base` | URL | `None` | Endpoint API custom |
| `--mode` | `hybrid`, `paper` | `hybrid` | Modalità di ragionamento |
| `--iterations` | intero | `10` (`50` su `ui`) | Iterazioni MCTS |
| `-v`, `--verbose` | flag | `False` | Stampa l'albero MCTS completo |

---

## Test

```bash
.venv/bin/pytest tests/
```
