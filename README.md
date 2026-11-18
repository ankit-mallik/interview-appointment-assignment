# RAG Backend – Document Ingestion + Conversational RAG with Interview Booking

A FastAPI backend with two main APIs:

1. **Document Ingestion API** – upload `.pdf` / `.txt` files, extract text, chunk it (2 strategies), create embeddings and store them in **Qdrant**. File metadata is saved in **SQLite** (SQLAlchemy).
2. **Conversational RAG API** – a custom RAG pipeline (no LangChain / RetrievalQAChain) that uses **Redis** for chat memory, supports multi-turn questions, and can **book interviews** (name, email, date, time) using the LLM's tool calling. Bookings are stored in the SQL database.

## Tech Stack

| Part | Choice |
|------|--------|
| Web framework | FastAPI |
| Vector DB | Qdrant (Docker) |
| Metadata / bookings DB | SQLite with SQLAlchemy 2.0 (can be swapped to Postgres by changing `DATABASE_URL`) |
| Chat memory | Redis (Docker) |
| Embeddings | `sentence-transformers` – `all-MiniLM-L6-v2` (runs locally, 384 dims) |
| LLM | Open-source model running locally with **Ollama** (`qwen2.5:7b` by default, configurable) – free, no API key |
| PDF parsing | `pypdf` |

## Project Structure

```
app/
├── main.py                 # FastAPI app + router registration
├── config.py               # settings loaded from .env
├── database.py             # SQLAlchemy engine / session
├── models.py               # ORM models: Document, Booking
├── schemas.py              # Pydantic request/response models
├── routers/
│   ├── ingestion.py        # /documents endpoints
│   └── chat.py             # /chat and /bookings endpoints
└── services/
    ├── text_extractor.py   # PDF / TXT -> text
    ├── chunking.py         # fixed-size and sentence-based chunking
    ├── embeddings.py       # sentence-transformers wrapper
    ├── vector_store.py     # Qdrant operations
    ├── chat_memory.py      # Redis chat history
    ├── llm.py              # Ollama wrapper
    ├── booking.py          # booking tool definition + validation + saving
    └── rag.py              # the RAG pipeline
tests/                      # pytest tests (external services are mocked)
sample_docs/                # sample file to try the API with
```

## Setup

**Requirements:** Python 3.11+, Docker, [Ollama](https://ollama.com/download)

```bash
# 1. start Qdrant and Redis
docker compose up -d

# 2. install Ollama from https://ollama.com/download,
#    make sure it's running, then download the LLM (one time, ~4.7 GB)
ollama pull qwen2.5:7b

# 3. create virtual env and install dependencies
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 4. configure environment
cp .env.example .env            # Windows: copy .env.example .env
# defaults work out of the box, no API keys needed

# 5. run the server
uvicorn app.main:app --reload
```

Swagger docs: http://localhost:8000/docs

### About the LLM (Ollama)

The LLM runs locally with Ollama (installed natively, not in Docker, so it can use the GPU),
so it's free and no data leaves your machine. The app talks to it at `OLLAMA_URL` (default `http://localhost:11434`).
Any Ollama model that supports **tool calling** works – change `LLM_MODEL` in `.env` and pull it first:

| Model | Size | Notes |
|-------|------|-------|
| `qwen2.5:7b` (default) | ~4.7 GB | Good balance, reliable tool calling |
| `llama3.1:8b` | ~4.9 GB | Also good at tool calling |
| `qwen2.5:3b` | ~1.9 GB | Faster on slow machines, but less reliable for bookings |

> The 7B model needs about 6 GB of free RAM. On a machine with less memory use `qwen2.5:3b`.

## API Endpoints

### Document Ingestion

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/documents/upload` | Upload a `.pdf` or `.txt` file (multipart). Form field `chunking_strategy` = `fixed` or `sentence` (default `sentence`) |
| GET | `/documents` | List uploaded documents with metadata |
| GET | `/documents/{id}` | Get one document's metadata |
| DELETE | `/documents/{id}` | Delete a document and its vectors |

```bash
curl -F "file=@sample_docs/company_handbook.txt" -F "chunking_strategy=sentence" \
  http://localhost:8000/documents/upload
```

### Conversational RAG

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat` | Ask a question / book an interview. Body: `{"session_id": "optional", "message": "..."}` |
| GET | `/chat/{session_id}/history` | Get chat history stored in Redis |
| DELETE | `/chat/{session_id}` | Clear chat history |
| GET | `/bookings` | List all interview bookings |

If you don't send a `session_id`, a new one is created and returned. Send it back on the next request to continue the conversation.

```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"message": "How many paid leave days do employees get?"}'

# follow-up question in the same session
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"session_id": "<id from previous response>", "message": "Can they be carried over?"}'
```

Booking example (multi-turn):

```
User: I want to book an interview
Bot:  Sure! Please share your full name, email, preferred date and time.
User: I'm John Doe, john@example.com
Bot:  Thanks John. What date and time works for you?
User: tomorrow at 3pm
Bot:  Your interview is booked for 2026-11-18 at 15:00. ...
```

The response includes a `booking` object when a booking was created, and `sources` with the chunks used for the answer.

## How It Works

### Chunking strategies

- **`fixed`** – splits text into 800-character windows with a 100-character overlap. Simple and predictable, but can cut sentences in half.
- **`sentence`** – splits text into sentences (regex-based) and packs whole sentences into chunks of up to 800 characters, repeating the last sentence in the next chunk as overlap. Produces cleaner chunks for the LLM. Very long sentences fall back to fixed-size splitting.

### RAG pipeline (`app/services/rag.py`)

1. Load the session's chat history from Redis.
2. If there is history, the LLM rewrites the latest message into a **standalone question** (so "can they be carried over?" becomes "Can unused leave days be carried over?"). This makes retrieval work for follow-up questions.
3. Embed the question and fetch the top-k chunks from Qdrant (cosine similarity).
4. Send the system prompt (with retrieved context + today's date) + chat history + question to the LLM, with a `book_interview` tool (Ollama tool calling).
5. If the LLM calls the tool, the input is validated with Pydantic (valid email, date not in the past, no double booking for the same slot) and saved in the `bookings` table. The result (success or error) goes back to the LLM so it can reply or ask the user to fix the details.
6. The question and final answer are saved to Redis (last 20 messages are kept, with a 24h TTL).

### Why LLM tool calling for bookings?

Instead of writing regexes to find names/emails/dates in the message, the LLM collects the details over multiple turns and only calls the tool when it has all four. Our code still validates everything before saving, so the LLM can't save bad data.

## Running Tests

```bash
pytest
```

Tests cover chunking, text extraction, booking validation and the RAG flow (Redis, Qdrant, embeddings and the LLM are mocked, so no Docker needed).

## Things I Would Improve With More Time

- Async database/Redis clients and background jobs for large file ingestion
- Alembic migrations instead of `create_all`
- Hybrid search (BM25 + vectors) and a reranker
- Authentication and rate limiting
- OCR for scanned PDFs
- Dockerfile for the API itself
