# Aurora Stays backend

Flask API for the Aurora Stays RAG question-answering service. It loads the
FAISS index from `faiss_index/`, so run the service from this folder.

## Endpoints

| Endpoint | Method | Returns |
|---|---|---|
| `/` | GET | Plain-text welcome message |
| `/v1/answer` | POST | Generated answer only |
| `/v1/answer_with_relevant_chunks` | POST | Answer plus the retrieved chunks |
| `/v1/relevant_chunks` | POST | Retrieved chunks only |

## Request fields

All endpoints accept a JSON object. `query` and `k` are used by every
endpoint; the remaining fields are only used by the two answer endpoints.

| Field | Type | Default | Allowed values |
|---|---|---|---|
| `query` | string | required | 1-2000 characters after trimming |
| `k` | integer | `4` | 1-20 |
| `model_name` | string | `gpt-4o-mini` | 1-100 characters |
| `temperature` | number | `0.0` | 0.0-2.0 |
| `top_p` | number | `1.0` | 0.0-1.0 |
| `max_tokens` | integer | `512` | 1-16384 |

## Errors

Invalid JSON or invalid field values return HTTP `400` with the offending
field, when applicable:

```json
{
  "error": {
    "message": "Field 'k' must be between 1 and 20.",
    "field": "k"
  }
}
```

Retrieval or generation failures return HTTP `502` with a generic message, so
upstream errors are not leaked to clients. Unknown routes return `404` and
unsupported methods return `405`; all error responses are JSON.

## Tests

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

The tests stub the HuggingFace embedding model, FAISS vector store, and OpenAI
client, so they do not need an `OPENAI_API_KEY` or a model download.
