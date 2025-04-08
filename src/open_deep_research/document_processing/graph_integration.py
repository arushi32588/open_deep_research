# src/open_deep_research/document_processing/graph_integration.py

from typing import Dict, List, Any, Optional, Callable
from pydantic import BaseModel, Field
from langchain_core.documents import Document
import os
from .loaders import PDFLoader, ExcelLoader
from .vector_store import FAISSVectorStore
from .rag import HybridRetrievalSystem

class LocalDocumentConfig(BaseModel):
    """Configuration for local document processing"""
    
    enabled: bool = Field(default=True, description="Whether to use local documents")
    paths: List[str] = Field(default_factory=list, description="Paths to local documents or directories")
    vector_store_dir: str = Field(default="vector_store", description="Directory for vector store")
    embedding_model: str = Field(default="text-embedding-3-large", description="Embedding model")
    reranker_model: str = Field(default="gpt-3.5-turbo", description="Reranker model")
    max_results: int = Field(default=10, description="Maximum number of results")
    recursive: bool = Field(default=True, description="Whether to recursively search directories")
    supported_extensions: List[str] = Field(
        default_factory=lambda: ['.pdf', '.xlsx', '.xls'],
        description="List of supported file extensions"
    )

def find_documents(paths: List[str], recursive: bool = True, supported_extensions: List[str] = None) -> List[str]:
    """Find all supported documents in the given paths"""
    if supported_extensions is None:
        supported_extensions = ['.pdf', '.xlsx', '.xls']
    
    documents = []
    print(f"Looking for documents in paths: {paths}")
    print(f"Supported extensions: {supported_extensions}")
    
    for path in paths:
        # Convert to absolute path if it's not already
        abs_path = os.path.abspath(path)
        print(f"Checking path: {abs_path}")
        
        if os.path.isfile(abs_path):
            print(f"Found file: {abs_path}")
            if any(abs_path.lower().endswith(ext) for ext in supported_extensions):
                print(f"File has supported extension: {abs_path}")
                documents.append(abs_path)
            else:
                print(f"File does not have supported extension: {abs_path}")
        elif os.path.isdir(abs_path):
            print(f"Found directory: {abs_path}")
            if recursive:
                for root, _, files in os.walk(abs_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        print(f"Found file in directory: {file_path}")
                        if any(file.lower().endswith(ext) for ext in supported_extensions):
                            print(f"File has supported extension: {file_path}")
                            documents.append(file_path)
                        else:
                            print(f"File does not have supported extension: {file_path}")
            else:
                for file in os.listdir(abs_path):
                    file_path = os.path.join(abs_path, file)
                    print(f"Found file in directory (non-recursive): {file_path}")
                    if any(file.lower().endswith(ext) for ext in supported_extensions):
                        print(f"File has supported extension: {file_path}")
                        documents.append(file_path)
                    else:
                        print(f"File does not have supported extension: {file_path}")
        else:
            print(f"Path does not exist: {abs_path}")
    
    print(f"Total documents found: {len(documents)}")
    return documents

def process_local_documents(config: LocalDocumentConfig) -> FAISSVectorStore:
    """Process local documents and return vector store"""
    # Initialize vector store
    vector_store = FAISSVectorStore(embedding_model=config.embedding_model)
    
    # Try to load existing vector store
    try:
        vector_store.load(config.vector_store_dir)
        print(f"Loaded vector store from {config.vector_store_dir}")
    except Exception as e:
        print(f"Creating new vector store: {e}")
    
    # Find all supported documents
    print(f"Looking for documents in paths: {config.paths}")
    document_paths = find_documents(
        config.paths,
        recursive=config.recursive,
        supported_extensions=config.supported_extensions
    )
    
    if not document_paths:
        print("No supported documents found in the specified paths")
        return vector_store
    
    print(f"Found {len(document_paths)} documents to process")
    
    # Process documents
    for path in document_paths:
        print(f"Processing document: {path}")
        if path.lower().endswith('.pdf'):
            loader = PDFLoader()
        elif path.lower().endswith(('.xlsx', '.xls')):
            loader = ExcelLoader()
        else:
            print(f"Unsupported file type: {path}")
            continue
            
        try:
            documents = loader.load(path)
            print(f"Loaded {len(documents)} chunks from {path}")
            vector_store.add_documents(documents)
            print(f"Added {len(documents)} chunks to vector store")
        except Exception as e:
            print(f"Error processing {path}: {e}")
    
    # Save vector store
    try:
        vector_store.save(config.vector_store_dir)
        print(f"Saved vector store to {config.vector_store_dir}")
    except Exception as e:
        print(f"Error saving vector store: {e}")
    
    return vector_store