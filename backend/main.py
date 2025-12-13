from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import logging
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables (looks in current directory and parent directories)
load_dotenv()

# Initialize FastAPI app with Swagger documentation
app = FastAPI(
    title="ApplianceCare AI Assistant API",
    description="AI-powered appliance repair assistant API using Pinecone vector database for semantic search",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI endpoint
    redoc_url="/redoc",  # ReDoc endpoint
    openapi_url="/openapi.json"  # OpenAPI schema endpoint
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

prompt = hub.pull("rlm/rag-prompt")
# Log the prompt template - try multiple ways to display it
try:
    prompt_str = str(prompt)
    prompt_template = getattr(prompt, 'template', None)
    prompt_messages = getattr(prompt, 'messages', None)
    
    logger.info("=" * 80)
    logger.info("Loaded RAG prompt template:")
    logger.info("=" * 80)
    if prompt_template:
        logger.info(f"Template content:\n{prompt_template}")
    elif prompt_messages:
        logger.info(f"Messages:\n{prompt_messages}")
    else:
        logger.info(f"Prompt object:\n{prompt_str}")
    logger.info("=" * 80)
    # Also print to ensure it shows up
    print("=" * 80)
    print("Loaded RAG prompt template:")
    print("=" * 80)
    if prompt_template:
        print(f"Template content:\n{prompt_template}")
    elif prompt_messages:
        print(f"Messages:\n{prompt_messages}")
    else:
        print(f"Prompt object:\n{prompt_str}")
    print("=" * 80)
except Exception as e:
    logger.error(f"Error logging prompt: {e}")
    print(f"Error logging prompt: {e}")

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("appliance-care-data")


# Request/Response models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "How to fix a washing machine that won't drain?",
                "top_k": 5
            }
        }


class SearchResult(BaseModel):
    score: float
    text: str
    source: str
    chunk_index: Optional[int] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "score": 0.85,
                "text": "To fix a washing machine that won't drain, first check the drain hose...",
                "source": "All about repairing major household appliances.pdf",
                "chunk_index": 1
            }
        }


class QueryResponse(BaseModel):
    results: List[SearchResult]
    query: str
    total_results: int
    context: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "How to fix a washing machine that won't drain?",
                "total_results": 5,
                "context": "To fix a washing machine that won't drain... [combined text from all results]",
                "results": [
                    {
                        "score": 0.85,
                        "text": "To fix a washing machine that won't drain...",
                        "source": "All about repairing major household appliances.pdf",
                        "chunk_index": 1
                    }
                ]
            }
        }


@app.get("/", tags=["General"])
async def root():
    """
    Root endpoint - API information and status.
    
    Returns basic API information and status.
    """
    return {
        "message": "ApplianceCare AI Assistant API",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health", tags=["General"])
async def health():
    """
    Health check endpoint.
    
    Returns the health status of the API.
    """
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse, tags=["Search"])
async def query_pinecone(request: QueryRequest):
    """
    Query the Pinecone vector database with a text query.
    
    This endpoint performs semantic search on the appliance repair knowledge base.
    It converts the query text to an embedding vector and searches for similar content
    in the Pinecone database.
    
    **Parameters:**
    - **query**: The search query text (e.g., "How to fix a washing machine?")
    - **top_k**: Number of results to return (default: 5, max recommended: 10)
    
    **Returns:**
    - List of search results with relevance scores, text content, and source documents
    
    **Example:**
    ```json
    {
        "query": "How to fix a washing machine that won't drain?",
        "top_k": 5
    }
    ```
    """
    try:
        # Generate embedding for the query
        query_embedding = embedding_model.embed_query(request.query)
        
        # Search Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=request.top_k,
            include_metadata=True
        )
        
        # Format results
        search_results = []
        for match in results['matches']:
            search_results.append(SearchResult(
                score=float(match['score']),
                text=match['metadata'].get('text', ''),
                source=match['metadata'].get('source', ''),
                chunk_index=match['metadata'].get('chunk_index')
            ))
        
        # Combine all text from results into a single context
        context_parts = []
        for result in search_results:
            if result.text.strip():
                context_parts.append(result.text.strip())
        
        context = "\n\n".join(context_parts)
        # logger.info(f"Generated context for query '{request.query}':\n{context}")
        
        return QueryResponse(
            results=search_results,
            query=request.query,
            total_results=len(search_results),
            context=context
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying database: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

