# src/open_deep_research/api/app.py
# src/open_deep_research/api/app.py

import os
import sys
import time
import asyncio
import random
from typing import Dict, List, Any, Optional
import uuid
from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import json
import pickle
import shutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import LangGraph components
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.types import interrupt

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../../.."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import open_deep_research components
try:
    from ..graph import builder
    from ..document_processing.loaders import PDFLoader, ExcelLoader
    from ..document_processing.vector_store import FAISSVectorStore
    from ..document_processing.rag import HybridRetrievalSystem
    from ..document_processing.graph_integration import LocalDocumentConfig, process_local_documents
except ImportError as e:
    print(f"Error importing open_deep_research components: {e}")
    raise

# ===============================
# Rate Limiting Infrastructure
# ===============================

class TokenBucket:
    """Token bucket algorithm for rate limiting"""
    
    def __init__(self, tokens_per_minute=7500, max_tokens=7500):
        self.tokens_per_minute = tokens_per_minute
        self.max_tokens = max_tokens
        self.tokens = max_tokens  # Start with a full bucket
        self.last_refill = time.time()
        self.lock = asyncio.Lock()
    
    async def take(self, tokens: int) -> bool:
        """
        Try to take tokens from the bucket
        Return True if successful, False if not enough tokens
        """
        async with self.lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    async def wait_for_tokens(self, tokens: int, max_wait: float = 60.0) -> bool:
        """
        Wait until enough tokens are available or max_wait is reached
        Returns True if got tokens, False if timed out
        """
        start_time = time.time()
        while time.time() - start_time < max_wait:
            async with self.lock:
                self._refill()
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True
            
            # Wait a bit before checking again (with randomized jitter)
            wait_time = 0.5 + random.random() * 0.5
            await asyncio.sleep(wait_time)
        
        return False
        
    def _refill(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Convert rate per minute to rate per second
        tokens_per_second = self.tokens_per_minute / 60.0
        new_tokens = int(elapsed * tokens_per_second)
        
        if new_tokens > 0:
            self.tokens = min(self.tokens + new_tokens, self.max_tokens)
            self.last_refill = now

# ===============================
# Model Fallback System
# ===============================

class ModelConfig:
    """Configuration for a model with fallbacks"""
    def __init__(
        self, 
        primary_model: str,
        fallback_models: List[str],
        retry_attempts: int = 3,
        backoff_base: float = 2.0
    ):
        self.primary_model = primary_model
        self.fallback_models = fallback_models
        self.retry_attempts = retry_attempts
        self.backoff_base = backoff_base

class ModelManager:
    """Manager for handling model calls with retry and fallback logic"""
    
    def __init__(self):
        self.model_configs = {
            "planner": ModelConfig(
                primary_model="claude-3-7-sonnet-latest",
                fallback_models=["claude-3-5-sonnet-latest", "claude-3-haiku-latest"]
            ),
            "writer": ModelConfig(
                primary_model="claude-3-5-sonnet-latest",
                fallback_models=["claude-3-haiku-latest"]
            )
        }
    
    async def call_with_fallback(
        self, 
        role: str, 
        call_function, 
        *args, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call a model function with retry and fallback logic
        """
        config = self.model_configs.get(role)
        if not config:
            raise ValueError(f"Unknown model role: {role}")
        
        models_to_try = [config.primary_model] + config.fallback_models
        last_error = None
        
        for model in models_to_try:
            for retry in range(config.retry_attempts):
                try:
                    # Update the model in kwargs
                    if "model" in kwargs:
                        kwargs["model"] = model
                    
                    # Make the API call
                    return await call_function(*args, **kwargs)
                
                except Exception as e:
                    last_error = e
                    error_str = str(e).lower()
                    
                    # If it's a rate limit error, back off and retry
                    if "rate_limit" in error_str or "429" in error_str:
                        if retry < config.retry_attempts - 1:
                            backoff_time = (config.backoff_base ** retry) + random.uniform(0.1, 1.0)
                            print(f"Rate limit hit with {model}. Backing off for {backoff_time:.2f}s")
                            await asyncio.sleep(backoff_time)
                            continue
                    
                    # For other errors or if we've exhausted retries, try the next model
                    print(f"Error with {model}: {e}. Trying next model.")
                    break
        
        # If we get here, all models failed
        raise Exception(f"All models failed for role {role}. Last error: {last_error}")

# ===============================
# Initialize components (globals)
# ===============================

# Initialize rate limiting components
token_bucket = TokenBucket(tokens_per_minute=7000)  # Give some buffer below 8000
model_manager = ModelManager()

app = FastAPI(title="Enhanced Open Deep Research API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize memory saver for graph checkpoints
memory = MemorySaver()

# Compile graph
graph = builder.compile(checkpointer=memory)

class ChatMessage(BaseModel):
    """Chat message model"""
    role: str = Field(..., description="Role of the message sender (user or assistant)")
    content: str = Field(..., description="Content of the message")

class ReportRequest(BaseModel):
    """Report request model"""
    topic: str = Field(..., description="Topic for the report")
    search_api: str = Field(default="tavily", description="Search API to use")
    planner_provider: str = Field(default="anthropic", description="Provider for planner model")
    planner_model: str = Field(default="claude-3-7-sonnet-latest", description="Model for planning")
    writer_provider: str = Field(default="anthropic", description="Provider for writer model")
    writer_model: str = Field(default="claude-3-5-sonnet-latest", description="Model for writing")
    max_search_depth: int = Field(default=1, description="Maximum search depth")
    report_structure: str = Field(default=None, description="Custom report structure")
    local_documents: LocalDocumentConfig = Field(
        default_factory=LocalDocumentConfig,
        description="Configuration for local documents"
    )

class ChatRequest(BaseModel):
    """Chat request model"""
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    stream: bool = Field(default=True, description="Whether to stream the response")
    local_documents: LocalDocumentConfig = Field(
        default_factory=LocalDocumentConfig,
        description="Configuration for local documents"
    )

async def create_thread() -> Dict[str, Any]:
    """Create a new thread ID"""
    return {"configurable": {"thread_id": str(uuid.uuid4())}}

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Open Deep Research API!"}

@app.post("/api/report")
async def generate_report(request: ReportRequest):
    """Generate a report on a given topic with streaming and rate limiting"""
    
    # Create an async generator function for streaming
    async def generate():
        try:
            print("DEBUG: Starting report generation process")
            # Reserve tokens for this request
            tokens_needed = 6000  # Estimate for report generation
            if not await token_bucket.wait_for_tokens(tokens_needed, max_wait=30.0):
                print("DEBUG: Rate limit exceeded at token reservation")
                error_msg = "Rate limit exceeded. Please try again in a minute."
                print(f"ERROR: {error_msg}")
                yield "data: " + json.dumps({
                    "type": "error", 
                    "error": error_msg,
                    "retry_after": 60,
                    "suggestion": "You can also try:\n1. Using a different model (e.g., gpt-4)\n2. Reducing the scope of your request\n3. Breaking your request into smaller parts"
                }, ensure_ascii=False) + "\n\n"
                yield "data: " + json.dumps({"type": "stream_end"}, ensure_ascii=False) + "\n\n"
                return
            
            print("DEBUG: Creating thread configuration")
            # Create thread configuration with GPT-4 models
            thread = {"configurable": {
                "thread_id": str(uuid.uuid4()),
                "search_api": "local",  # Use local search only
                "planner_provider": "openai",  # Use OpenAI instead of Anthropic
                "planner_model": "gpt-4",  # Use GPT-4 for planning
                "writer_provider": "openai",  # Use OpenAI instead of Anthropic
                "writer_model": "gpt-4",  # Use GPT-4 for writing
                "max_search_depth": 1,  # Simplified search depth
                "local_documents": {
                    "enabled": True,  # Always enable local documents
                    "paths": request.local_documents.paths,
                    "vector_store_dir": request.local_documents.vector_store_dir,
                    "embedding_model": request.local_documents.embedding_model,
                    "reranker_model": request.local_documents.reranker_model,
                    "max_results": request.local_documents.max_results,
                    "recursive": request.local_documents.recursive,
                    "supported_extensions": request.local_documents.supported_extensions
                }
            }}
            
            # Process local documents if provided
            if request.local_documents.paths:
                try:
                    print("DEBUG: Processing local documents")
                    print(f"DEBUG: Local document paths: {request.local_documents.paths}")
                    # Use absolute paths for local documents
                    abs_paths = [os.path.abspath(path) for path in request.local_documents.paths]
                    request.local_documents.paths = abs_paths
                    print(f"DEBUG: Absolute paths: {abs_paths}")
                    
                    process_local_documents(request.local_documents)
                    status_msg = "Processing local documents for search..."
                    print(f"STATUS: {status_msg}")
                    yield "data: " + json.dumps({
                        "type": "text", 
                        "text": f"{status_msg}\n\n"
                    }, ensure_ascii=False) + "\n\n"
                except Exception as e:
                    error_msg = f"Error processing local documents: {str(e)}"
                    print(f"ERROR: {error_msg}")
                    yield "data: " + json.dumps({
                        "type": "error", 
                        "error": error_msg
                    }, ensure_ascii=False) + "\n\n"
                    yield "data: " + json.dumps({"type": "stream_end"}, ensure_ascii=False) + "\n\n"
                    return
            
            try:
                print("DEBUG: Starting report generation stream")
                # Send initial data
                yield "data: " + json.dumps({"type": "stream_start"}, ensure_ascii=False) + "\n\n"
                
                # First, perform a direct search to get the information
                print("DEBUG: Performing direct search for information")
                status_msg = "\n\nSearching for information...\n\n"
                print(f"STATUS: {status_msg}")
                yield "data: " + json.dumps({
                    "type": "text", 
                    "text": status_msg
                }, ensure_ascii=False) + "\n\n"
                
                # Import the necessary components for direct search
                from ..document_processing.vector_store import FAISSVectorStore
                from ..document_processing.rag import HybridRetrievalSystem
                
                # Initialize vector store
                print("DEBUG: Initializing vector store")
                vector_store = FAISSVectorStore(
                    embedding_model=request.local_documents.embedding_model
                )
                
                # Load vector store
                try:
                    vector_store.load(request.local_documents.vector_store_dir)
                    print("DEBUG: Loaded vector store from", request.local_documents.vector_store_dir)
                except Exception as e:
                    print(f"DEBUG: Error loading vector store: {str(e)}")
                    print("DEBUG: Creating new vector store")
                
                # Initialize hybrid retrieval system
                print("DEBUG: Initializing hybrid retrieval system")
                retrieval_system = HybridRetrievalSystem(
                    vector_store=vector_store,
                    reranker_model=request.local_documents.reranker_model,
                    max_results=request.local_documents.max_results
                )
                
                # Perform search
                print("DEBUG: Performing search for", request.topic)
                search_results = []
                
                # Create search queries based on the topic
                search_queries = [
                    f"Information about {request.topic}",
                    f"Details about {request.topic}",
                    f"Analysis of {request.topic}",
                    f"Comparison of {request.topic}",
                    f"History of {request.topic}",
                    f"Business model of {request.topic}",
                    f"Products and services of {request.topic}",
                    f"Market position of {request.topic}",
                    f"Financial performance of {request.topic}",
                    f"Future outlook for {request.topic}"
                ]
                
                # Search for each query
                for query in search_queries:
                    print(f"DEBUG: Searching for query: {query}")
                    # Use await with the async search function
                    results = await retrieval_system.search(query)
                    # Limit to top 5 most relevant results per query to avoid token limits
                    results = results[:5]
                    print(f"DEBUG: Found {len(results)} results for query: {query}")
                    search_results.extend(results)
                
                # Remove duplicates while preserving order
                seen = set()
                unique_results = []
                for result in search_results:
                    result_key = (result.page_content, result.metadata.get('source', ''), result.metadata.get('page', 0))
                    if result_key not in seen:
                        seen.add(result_key)
                        unique_results.append(result)
                
                search_results = unique_results
                
                # Format search results
                formatted_results = []
                for result in search_results:
                    formatted_results.append({
                        "content": result.page_content,
                        "metadata": result.metadata
                    })
                    print(f"DEBUG: Including result from {result.metadata.get('source', 'unknown')} page {result.metadata.get('page', 'unknown')}")
                    print(f"DEBUG: Content preview: {result.page_content[:200]}...")
                
                # Create report prompt with limited context
                report_prompt = f"""Based on the following information about Apple and Amazon, generate a comprehensive comparison report.
Focus on key aspects like business models, financial performance, market position, and future outlook.

Available information:
{json.dumps(formatted_results, indent=2)}

Please generate a detailed report that:
1. Compares their business models and strategies
2. Analyzes their financial performance
3. Evaluates their market positions
4. Discusses their future outlook
5. Highlights key differentiators

Format the report with clear sections and use specific data points from the provided information."""

                print("DEBUG: Starting report generation with prompt")
                print("=" * 80)
                print(report_prompt)
                print("=" * 80)
                
                # Generate the report directly
                from langchain_openai import ChatOpenAI
                model = ChatOpenAI(model="gpt-4", streaming=True)
                
                print("DEBUG: Streaming report generation")
                report_content = ""
                
                # Stream the report generation
                async for chunk in model.astream(report_prompt):
                    if hasattr(chunk, 'content'):
                        content = chunk.content
                        if content:
                            report_content += content
                            print("DEBUG: Yielding report content chunk")
                            print("\n" + "-"*80)
                            print("REPORT CONTENT CHUNK:")
                            print(content)
                            print("-"*80 + "\n")
                            
                            chunk_data = {"type": "text", "text": content}
                            yield "data: " + json.dumps(chunk_data, ensure_ascii=False) + "\n\n"
                
                print("DEBUG: Report generation completed, sending stream end")
                # Send completion message
                yield "data: " + json.dumps({"type": "stream_end"}, ensure_ascii=False) + "\n\n"
                
            except Exception as e:
                error_str = str(e)
                print(f"DEBUG: Error in report generation: {error_str}")
                print(f"ERROR: {error_str}")
                error_msg = {"type": "error", "error": error_str}
                
                if "rate_limit" in error_str.lower() or "429" in error_str:
                    suggestion = (
                        "Rate limit exceeded. You can:\n"
                        "1. Wait a minute and try again\n"
                        "2. Use a different model (e.g., gpt-4)\n"
                        "3. Reduce the scope of your request\n"
                        "4. Break your request into smaller parts"
                    )
                    error_msg["suggestion"] = suggestion
                    print(f"SUGGESTION: {suggestion}")
                
                yield "data: " + json.dumps(error_msg, ensure_ascii=False) + "\n\n"
                yield "data: " + json.dumps({"type": "stream_end"}, ensure_ascii=False) + "\n\n"
                
        except Exception as e:
            print(f"DEBUG: Unexpected error in generate: {str(e)}")
            print(f"ERROR: Unexpected error: {str(e)}")
            yield "data: " + json.dumps({
                "type": "error",
                "error": f"Unexpected error: {str(e)}",
                "suggestion": "Please try again or contact support if the issue persists."
            }, ensure_ascii=False) + "\n\n"
            yield "data: " + json.dumps({"type": "stream_end"}, ensure_ascii=False) + "\n\n"

    # Return streaming response
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat endpoint with local document RAG support and rate limiting"""
    
    try:
        # Reserve tokens for this request
        tokens_needed = 2000  # Estimate for chat
        if not await token_bucket.take(tokens_needed):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded. Please try again later.",
                    "retry_after": 60  # Suggest retrying after 1 minute
                }
            )
        
        # Get the user's latest message
        latest_message = next((m for m in reversed(request.messages) if m.role == "user"), None)
        if not latest_message:
            raise HTTPException(status_code=400, detail="No user message found")
        
        # Prepare query for RAG
        query = latest_message.content
        
        # Process local documents if enabled
        vector_store = None
        if request.local_documents.enabled:
            try:
                vector_store = process_local_documents(request.local_documents)
            except Exception as e:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Error processing local documents: {str(e)}"}
                )
        
        # Stream function to convert events to Vercel AI SDK compatible format
        async def stream_events():
            try:
                # Send initial data to conform to Vercel AI SDK protocol
                yield "data: " + json.dumps({"type": "stream_start"}, ensure_ascii=False) + "\n\n"
                
                # Perform local search if vector store is available
                local_docs = []
                if vector_store:
                    local_docs = vector_store.search(query, k=5)
                
                # Get web search results using the open_deep_research search
                # We'll need to adapt this part based on the actual search implementation
                web_docs = []
                
                # For now, we'll just use local docs if available
                docs = local_docs + web_docs
                
                # Generate context from documents
                context = "\n\n".join([doc.page_content for doc in docs[:5]]) if docs else ""
                
                # Prepare final prompt with context and query
                prompt = f"""Use the following information to answer the user's question:

Context:
{context}

User Question: {query}

Answer:"""
                
                # Generate response with an LLM
                # We'll use the same model as specified in the request
                from langchain_openai import ChatOpenAI
                model = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)
                
                # Stream the response
                response = ""
                try:
                    async for chunk in model.astream(prompt):
                        if hasattr(chunk, 'content'):
                            content = chunk.content
                            if content:
                                response += content
                                chunk_data = {"type": "text", "text": content}
                                yield "data: " + json.dumps(chunk_data, ensure_ascii=False) + "\n\n"
                except Exception as e:
                    error_str = str(e)
                    if "rate_limit" in error_str or "429" in error_str:
                        # Handle rate limit error specifically
                        error_msg = {
                            "type": "error", 
                            "error": "Rate limit exceeded. Please try again later or use a different model.",
                            "details": error_str,
                            "suggestion": "Try using a different model provider or reduce the scope of your request."
                        }
                        yield "data: " + json.dumps(error_msg, ensure_ascii=False) + "\n\n"
                        yield "data: " + json.dumps({"type": "stream_end"}, ensure_ascii=False) + "\n\n"
                        return
                    else:
                        # Re-raise other exceptions
                        raise
                
                # Send final completion message
                yield "data: " + json.dumps({"type": "stream_end"}, ensure_ascii=False) + "\n\n"
            except Exception as e:
                # Handle errors
                error_str = str(e)
                error_msg = {"type": "error", "error": error_str}
                
                # Add more helpful information for rate limit errors
                if "rate_limit" in error_str or "429" in error_str:
                    error_msg["suggestion"] = "Try using a different model provider or reduce the scope of your request."
                
                yield "data: " + json.dumps(error_msg, ensure_ascii=False) + "\n\n"
                yield "data: " + json.dumps({"type": "stream_end"}, ensure_ascii=False) + "\n\n"
        
        # Return streaming response
        return StreamingResponse(
            stream_events(),
            media_type="text/event-stream; charset=utf-8"
        )
    except Exception as e:
        # Handle any unexpected errors
        return JSONResponse(
            status_code=500,
            content={"error": f"An unexpected error occurred: {str(e)}"}
        )

@app.post("/api/documents/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload multiple documents for processing"""
    try:
        # Create uploads directory if it doesn't exist
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        processed_files = []
        errors = []
        
        for file in files:
            if not file.filename.endswith(('.pdf', '.xlsx', '.xls')):
                errors.append(f"Unsupported file type: {file.filename}")
                continue
                
            try:
                # Save file
                file_path = os.path.join(upload_dir, file.filename)
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                processed_files.append(file_path)
            except Exception as e:
                errors.append(f"Error saving {file.filename}: {str(e)}")
        
        if not processed_files:
            raise HTTPException(
                status_code=400,
                detail="No valid files were uploaded. Supported formats: PDF, XLSX, XLS"
            )
        
        # Process all uploaded documents
        config = LocalDocumentConfig(
            enabled=True,
            paths=processed_files,
            vector_store_dir="vector_store"
        )
        
        try:
            process_local_documents(config)
            return {
                "status": "success",
                "message": f"Successfully processed {len(processed_files)} documents",
                "processed_files": [os.path.basename(f) for f in processed_files],
                "errors": errors if errors else None
            }
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": f"Error processing documents: {str(e)}",
                    "processed_files": [os.path.basename(f) for f in processed_files],
                    "errors": errors if errors else None
                }
            )
    except Exception as e:
        # Handle any unexpected errors
        return JSONResponse(
            status_code=500,
            content={"error": f"An unexpected error occurred: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

