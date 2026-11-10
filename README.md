# RAP Demo

Implementazione di **Reasoning via Planning (RAP)** con **MCTS** su Blocksworld, supporto **Hybrid / Paper mode**, modelli locali (`llama.cpp`), API esterne (`litellm`) con vincoli TPM/RPM per modello, salvataggio/replay cache delle run e dashboard Web in tempo reale.

---

## Setup

```bash
uv sync
source .venv/bin/activate
```

*(Su Python 3.15, se necessario per compilare `fastuuid`: `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 uv sync`)*

---

## Configurazione: `.env` (API Key) + `models.json` (Modelli & Limiti)

- **`.env`** contiene solo i token API:
  ```bash
  cp .env.example .env
  ```
  ```env
  GEMINI_API_KEY=AIza...
  # OPENAI_API_KEY=sk-...
  # ANTHROPIC_API_KEY=sk-ant-...
  ```

- **`models.json`** contiene i profili dei modelli, backend, `api_base`, timeout e limiti **`tpm` / `rpm`** per singolo modello (`gemma4-26b`, `gemma4-31b`, `gpt-4o-mini`, `local-llama`, ecc.).
  ```bash
  # Elenca i modelli configurati in models.json
  rap models
  ```

---

## Attivare `llama.cpp` (Modello Locale)

1. **Installa `llama.cpp`** (macOS con Homebrew):
   ```bash
   brew install llama.cpp
   ```
2. **Avvia il server locale** scaricando un GGUF da Hugging Face (`-hf`) con GPU Metal (`-ngl 99`):
   ```bash
   llama-server -hf saidutta69/LFM3-VL-2B-heretic --port 8080 -ngl 99
   ```
3. In un altro terminale usa `--model local-llama` (configurato in `models.json`) oppure `--backend llama-server`.

---

## Utilizzo CLI

### 1. Esecuzione da terminale (`rap run`)

```bash
# Mock offline (nessun modello richiesto)
rap run --iterations 10 -v

# Usando un alias definito in models.json (risolve automaticamente backend, model, tpm, rpm)
rap run --model gemma4-26b -v
rap run --model gemma4-31b -v

# Modello ad-hoc non presente in models.json
rap run --backend litellm --model openai/gpt-4o-mini --tpm 30000 --rpm 60 -v

# Modello locale llama.cpp
rap run --model local-llama -v
```

### 2. Dashboard Web (`rap ui`)

Il modello si sceglie **solo dal comando nel terminale** (nessun selettore nella pagina web). Avvia e apri `http://127.0.0.1:8000`:

```bash
rap ui --backend mock
rap ui --model gemma4-26b --port 8000
rap ui --model local-llama --port 8000
```

### 3. Confronto Hybrid vs Paper (`rap compare`)

```bash
rap compare --backend mock --iterations 10
rap compare --model gemma4-26b --iterations 10
```

### 4. Cache & Replay delle Run (`rap history` / `rap replay`)

Ogni esecuzione (`rap run` e `rap ui` al completamento) salva automaticamente un JSON in `runs/` con data, modello, iterazioni e albero MCTS completo (`--no-save` per disabilitare):

```bash
# Elenca le run salvate
rap history

# Rivedi una run salvata da terminale (zero chiamate API)
rap replay latest -v
rap replay 1 -v

# Riaprila nella Web UI (zero chiamate API)
rap replay latest --ui --port 8000
```

---

## Flag Principali

| Flag | Valori | Default | Descrizione |
| :--- | :--- | :--- | :--- |
| `--model` | alias in `models.json` o ID modello | `None` | Seleziona modello (risolve anche backend/limiti da `models.json`) |
| `--config` | percorso `.json` | `models.json` | File JSON custom dei modelli e limiti |
| `--backend` | `mock`, `litellm`, `llama-server` | `mock` | Override esplicito del backend |
| `--api-base` | URL | `None` | Endpoint API custom |
| `--tpm` / `--rpm` | intero | da `models.json` | Override limiti **Token/Richieste al Minuto** |
| `--mode` | `hybrid`, `paper` | `hybrid` | Modalità di ragionamento |
| `--iterations` | intero | `10` (`50` su `ui`) | Iterazioni MCTS |
| `-v`, `--verbose` | flag | `False` | Stampa l'albero MCTS completo |

---

## Test

```bash
.venv/bin/pytest tests/
```
