from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub
from langchain_openai import ChatOpenAI

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

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("appliance-care-data")

# Initialize OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Initialize OpenAI Chat model
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",  # You can change to "gpt-4" or "gpt-4-turbo" if needed
    temperature=0.7,
    api_key=OPENAI_API_KEY
)


# Request/Response models
class QueryRequest(BaseModel):
    query: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "How to fix a washing machine that won't drain?"
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
    answer: Optional[str] = None
    total_score: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Based on the provided context, here's how to fix a washing machine that won't drain...",
                "total_score": 53.3
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
    Query the Pinecone vector database with a text query and get AI-generated answer.
    
    This endpoint performs semantic search on the appliance repair knowledge base.
    It converts the query text to an embedding vector, searches for similar content
    in the Pinecone database (top 3 results), formats a RAG prompt, and generates an AI answer using OpenAI.
    
    **Parameters:**
    - **query**: The search query text (e.g., "How to fix a washing machine?")
    
    **Returns:**
    - **answer**: AI-generated answer based on the context and query
    - **total_score**: Average relevance score as a percentage (average of top 3 results * 100)
    
    **Example:**
    ```json
    {
        "query": "How to fix a washing machine that won't drain?"
    }
    ```
    """
    try:
        # Default top_k is 3
        TOP_K = 3
        
        # Generate embedding for the query
        query_embedding = embedding_model.embed_query(request.query)
        
        # Search Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=TOP_K,
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
        
        # Format the prompt with query and context
        ai_answer = None
        try:
            messages = prompt.format_messages(question=request.query, context=context)
            
            # Call OpenAI API with the formatted messages
            response = llm.invoke(messages)
            
            # Extract the answer from the response
            if hasattr(response, 'content'):
                ai_answer = response.content
            else:
                ai_answer = str(response)
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            print(f"Error calling OpenAI API: {e}")
            # Continue without AI answer if OpenAI fails
        
        # Calculate total score as percentage: (average of scores) * 100
        if search_results:
            avg_score = sum(result.score for result in search_results) / len(search_results)
            total_score = avg_score * 100
        else:
            total_score = 0.0
        
        logger.info(f"Total score calculated (percentage): {total_score:.2f}%")
        print(f"Total score calculated (percentage): {total_score:.2f}%")
        
        return QueryResponse(
            answer=ai_answer,
            total_score=total_score
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying database: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

