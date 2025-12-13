# Backend - FastAPI Server

This is the FastAPI backend server for the ApplianceCare AI Assistant.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the backend directory (or use the one from the root directory):
```env
PINECONE_API_KEY=your_pinecone_api_key_here
```

3. Run the server:
```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

### GET `/`
Health check endpoint

### GET `/health`
Health check endpoint

### POST `/query`
Query the Pinecone database

**Request Body:**
```json
{
  "query": "How to fix a washing machine?",
  "top_k": 5
}
```

**Response:**
```json
{
  "results": [
    {
      "score": 0.85,
      "text": "To fix a washing machine...",
      "source": "All about repairing major household appliances.pdf",
      "chunk_index": 1
    }
  ],
  "query": "How to fix a washing machine?",
  "total_results": 5
}
```

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

