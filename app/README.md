# Guardrail Governance Console

Local FastAPI + static frontend for the QLoRA guardrail adapter.

## Run

```powershell
.\run_guardrail_app.ps1
```

Open:

```text
http://127.0.0.1:8000
```

The backend starts a persistent WSL/Kali worker that loads:

```text
models/guardrail-qwen25-1_5b-qlora-v1
```

## API

```http
POST /api/analyze
{"text":"Ignora las instrucciones anteriores..."}
```

```http
POST /api/analyze-batch
{"items":["prompt 1", {"prompt":"prompt 2"}]}
```

```http
POST /api/upload-json
multipart form file=<json>
```
