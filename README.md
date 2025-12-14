# 🔧 ApplianceCare AI Assistant

An AI-powered assistant for appliance repair and maintenance, built with FastAPI backend and Next.js frontend, using Pinecone vector database for semantic search and OpenAI for answer generation.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [API Documentation](#api-documentation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Environment Variables](#environment-variables)
- [Data Processing](#data-processing)

## 🎯 Overview

ApplianceCare AI Assistant is a RAG (Retrieval-Augmented Generation) system that helps users find accurate appliance repair information. It uses:

- **Semantic Search**: Pinecone vector database for finding relevant repair information
- **AI Generation**: OpenAI GPT-3.5-turbo for generating contextual answers
- **Modern UI**: Next.js frontend with responsive design

## ✨ Features

- 🔍 **Semantic Search**: Find relevant appliance repair information using natural language queries
- 🤖 **AI-Powered Answers**: Get contextual, accurate answers based on retrieved information
- 📊 **Confidence Scoring**: See relevance scores for each query
- 🎨 **Modern UI**: Beautiful, responsive web interface
- 📚 **RAG Architecture**: Retrieval-Augmented Generation for accurate responses
- 🚀 **Fast API**: FastAPI backend with automatic API documentation

## 🏗️ Architecture

```
┌─────────────────┐
│   Frontend      │
│   (Next.js)     │
│   Port: 3000    │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐
│   Backend       │
│   (FastAPI)     │
│   Port: 8000    │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌──────────┐
│Pinecone│ │  OpenAI  │
│Vector  │ │   GPT    │
│Database│ │   API    │
└────────┘ └──────────┘
```

### Workflow

1. **User Query** → Frontend sends query to backend
2. **Embedding Generation** → Query converted to 384-dimensional vector
3. **Vector Search** → Pinecone finds top 3 most similar documents
4. **Context Creation** → Relevant text chunks combined
5. **Score Check** → If score < 25%, return fallback message
6. **AI Generation** → OpenAI generates answer from context
7. **Response** → Answer + confidence score returned to user

## 📁 Project Structure

```
ApplianceCare_AI_Assistant/
├── backend/                 # FastAPI backend server
│   ├── main.py             # Main API server
│   ├── requirements.txt    # Python dependencies
│   └── README.md          # Backend documentation
├── frontend/               # Next.js frontend application
│   ├── app/               # Next.js app directory
│   │   ├── page.jsx       # Main page component
│   │   ├── layout.jsx     # Root layout
│   │   └── globals.css    # Global styles
│   ├── package.json       # Node.js dependencies
│   └── README.md         # Frontend documentation
├── PDFs/                  # Source PDF documents
│   └── Extracted_text/   # Extracted text files
├── DataExtraction.ipynb   # Jupyter notebook for data processing
└── README.md             # This file
```

## 🔧 Prerequisites

- **Python 3.11+** (or 3.12)
- **Node.js 18+** and npm
- **Pinecone API Key** ([Get one here](https://www.pinecone.io/))
- **OpenAI API Key** ([Get one here](https://platform.openai.com/))

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ApplianceCare_AI_Assistant
```

### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file in backend directory or root directory
# Add your API keys:
# PINECONE_API_KEY=your_pinecone_api_key_here
# OPENAI_API_KEY=your_openai_api_key_here

# Run the server
python main.py
```

The backend will be available at `http://localhost:8000`

### 3. Frontend Setup

```bash
# Navigate to frontend directory (in a new terminal)
cd frontend

# Install dependencies
npm install

# Run the development server
npm run dev
```

The frontend will be available at `http://localhost:3000`

## 📚 API Documentation

### Base URL

```
http://localhost:8000
```

### Endpoints

#### `GET /`
Health check endpoint

**Response:**
```json
{
  "message": "ApplianceCare AI Assistant API",
  "status": "running",
  "docs": "/docs",
  "redoc": "/redoc"
}
```

#### `GET /health`
Health check endpoint

**Response:**
```json
{
  "status": "healthy"
}
```

#### `POST /query`
Query the Pinecone database and get AI-generated answer

**Request Body:**
```json
{
  "query": "How to fix a washing machine that won't drain?"
}
```

**Response:**
```json
{
  "answer": "Based on the provided context, here's how to fix...",
  "total_score": 53.3
}
```

**Response Fields:**
- `answer` (string | null): AI-generated answer or fallback message if score < 25%
- `total_score` (float): Average relevance score as percentage (0-100)

**Fallback Message:**
If the relevance score is less than 25%, the API returns:
```json
{
  "answer": "I could not find enough information about this issue in the dataset.",
  "total_score": 18.5
}
```

### Interactive API Documentation

Once the backend is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 💻 Usage

### Using the Web Interface

1. Start both backend and frontend servers
2. Open `http://localhost:3000` in your browser
3. Enter your appliance repair question
4. Click "Search" to get AI-generated answer with confidence score

### Using the API Directly

```bash
# Example query using curl
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "How to fix a washing machine that won'\''t drain?"}'
```

### Example Queries

- "How to fix a washing machine that won't drain?"
- "Iron burns cloth, what to do?"
- "Toaster not heating properly"
- "Refrigerator making loud noise"

## 🛠️ Technologies Used

### Backend
- **FastAPI** - Modern Python web framework
- **Pinecone** - Vector database for embeddings
- **LangChain** - LLM framework and utilities
- **HuggingFace** - Embedding model (all-MiniLM-L6-v2)
- **OpenAI** - GPT-3.5-turbo for answer generation
- **Uvicorn** - ASGI server

### Frontend
- **Next.js 14** - React framework with App Router
- **React** - UI library
- **Axios** - HTTP client
- **JavaScript** - Programming language (no TypeScript)

### Data Processing
- **PyMuPDF (fitz)** - PDF text extraction
- **Python** - Data processing scripts
- **Jupyter Notebook** - Interactive data processing

## 🔐 Environment Variables

Create a `.env` file in the root directory or backend directory:

```env
# Required
PINECONE_API_KEY=your_pinecone_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

**Note**: The `load_dotenv()` function looks for `.env` files in the current directory and parent directories.

## 📊 Data Processing

The `DataExtraction.ipynb` notebook contains the complete workflow for:

1. **PDF Extraction**: Extract text from PDF documents
2. **Text Cleaning**: Remove OCR artifacts and noise
3. **Chunking**: Split documents into smaller chunks (800 chars, 150 overlap)
4. **Embedding Generation**: Create 384-dimensional vectors using HuggingFace
5. **Vector Storage**: Store embeddings in Pinecone with metadata

### Database Statistics

- **Total Vectors**: 593
- **Embedding Dimension**: 384
- **Similarity Metric**: Cosine
- **Index Name**: `appliance-care-data`
- **Top K Results**: 3 (default)

### Embedding Model

- **Model**: `all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Provider**: HuggingFace Transformers
- **Advantages**: Fast, efficient, good semantic understanding

## 🎯 Key Features Explained

### RAG (Retrieval-Augmented Generation)

The system uses RAG to provide accurate answers:

1. **Retrieval**: Semantic search finds relevant context from Pinecone
2. **Augmentation**: Context is combined with the user query
3. **Generation**: OpenAI generates an answer based on the augmented prompt

### Confidence Scoring

- Scores are calculated as: `(average of top 3 results) * 100`
- Scores < 25% trigger a fallback message
- Higher scores indicate better relevance

### Top K Results

The system uses the top 3 most relevant results by default to:
- Provide comprehensive context
- Balance accuracy and speed
- Generate more complete answers

## 📝 Notes

- The Pinecone database must be populated before using the API
- Use the `DataExtraction.ipynb` notebook to populate the database
- The backend automatically loads the RAG prompt from LangChain Hub
- CORS is configured to allow requests from `http://localhost:3000`

## 🔗 Useful Links

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Next.js Documentation](https://nextjs.org/docs)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [LangChain Documentation](https://python.langchain.com/)

## 📄 License

This project is for demonstration purposes.

---


