# RAP Demo

Implementazione di **Reasoning via Planning (RAP)** con **MCTS** su Blocksworld, supporto **Hybrid / Paper mode**, modelli locali (`llama.cpp`), API esterne (`litellm`) con vincoli TPM/RPM e dashboard Web in tempo reale.

---

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

*(Su Python 3.15, se necessario per compilare `fastuuid`: `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 uv pip install -e ".[dev]"`)*

---

## Configurazione `.env` (API Esterne & Limiti)

Per usare API esterne con `--backend litellm`, copia `.env.example` in `.env` e inserisci la tua API key:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
# GEMINI_API_KEY=...
# GROQ_API_KEY=...

# Limiti opzionali (Token/Richieste al minuto)
# RAP_TPM=30000
# RAP_RPM=60
```

Il file `.env` viene caricato automaticamente all'avvio.

---

## Attivare `llama.cpp` (Modello Locale)

1. **Installa `llama.cpp`** (macOS con Homebrew):
   ```bash
   brew install llama.cpp
   ```
2. **Avvia il server locale** in un terminale dedicato scaricando automaticamente un modello GGUF da Hugging Face (`-hf`) con GPU Metal (`-ngl 99`):
   ```bash
   llama-server -hf saidutta69/LFM3-VL-2B-heretic --port 8080 -ngl 99
   ```
   *(Puoi sostituire il repo con qualsiasi modello GGUF, es. `LiquidAI/LFM3-1.7B-GGUF`.)*
3. Lascia aperto quel terminale e usa `--backend llama-server` nei comandi `rap` qui sotto.

---

## Utilizzo CLI

### 1. Esecuzione da terminale (`rap run`)

```bash
# Mock offline (nessun modello richiesto)
rap run --iterations 10 -v

# API esterne via LiteLLM + vincoli token/minuto
rap run --backend litellm --model openai/gpt-4o-mini --tpm 30000 --rpm 60 --iterations 10 -v
rap run --backend litellm --model anthropic/claude-3-5-sonnet-20241022 --iterations 10 -v

# Modello locale via llama-server (avviato sopra)
rap run --backend llama-server --iterations 10 -v
```

### 2. Dashboard Web (`rap ui`)

Il modello e il backend si scelgono **direttamente dal comando nel terminale** (nessun selettore nella UI web). Avvia il comando e apri `http://127.0.0.1:8000`:

```bash
rap ui --backend mock
rap ui --backend litellm --model openai/gpt-4o-mini --tpm 30000 --port 8000
rap ui --backend llama-server --port 8000
```

### 3. Confronto Hybrid vs Paper (`rap compare`)

```bash
rap compare --backend mock --iterations 10
rap compare --backend litellm --model openai/gpt-4o-mini --tpm 30000 --iterations 10
```

### 4. Cache & Replay delle Run (`rap history` / `rap replay`)

Ogni esecuzione (`rap run` e `rap ui` al termine) salva automaticamente un file JSON in `runs/` con **data, modello, backend, iterazioni, metriche e albero MCTS completo** (usa `--no-save` per disabilitare). Puoi rivedere i risultati salvati senza consumare token API:

```bash
# Elenca le run salvate
rap history

# Rivedi l'ultima run (o l'indice #1, #2...) da terminale
rap replay latest -v
rap replay 1 -v

# Riapri una run salvata nella Web UI (zero chiamate LLM)
rap replay latest --ui --port 8000
```

---

## Flag Principali

| Flag | Valori | Default | Descrizione |
| :--- | :--- | :--- | :--- |
| `--backend` | `mock`, `litellm`, `llama-server` | `mock` | Backend LLM |
| `--model` | es. `gpt-4o-mini`, `gemini/gemini-2.0-flash` | `None` | Modello per LiteLLM |
| `--api-base` | URL | `None` | Endpoint API custom |
| `--tpm` | intero | `None` | Limite massimo **Token al Minuto** (o `RAP_TPM` in `.env`) |
| `--rpm` | intero | `None` | Limite massimo **Richieste al Minuto** (o `RAP_RPM` in `.env`) |
| `--mode` | `hybrid`, `paper` | `hybrid` | Modalità di ragionamento |
| `--iterations` | intero | `10` (`50` su `ui`) | Iterazioni MCTS |
| `-v`, `--verbose` | flag | `False` | Stampa l'albero MCTS completo |

---

## Test

```bash
.venv/bin/pytest tests/
```
