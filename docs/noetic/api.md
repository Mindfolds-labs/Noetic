# Noetic API

## Endpoints
- `GET /health` -> `{"status":"ok"}`
- `POST /predict` -> tokens PAWP a partir de texto.

## Payloads
Request:
```json
{"text":"olá mundo","language":"pt"}
```
Response:
```json
{"tokens":[{"wp_piece":"..."}]}
```

## Execução local
```bash
uvicorn noetic_pawp.api.server:app --reload
```

## Deploy Docker (simples)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install .[api]
CMD ["uvicorn","noetic_pawp.api.server:app","--host","0.0.0.0","--port","8000"]
```
