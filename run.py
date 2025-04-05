import os
from dotenv import load_dotenv
import uvicorn
from src.open_deep_research.api.app import app

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    uvicorn.run(
        "src.open_deep_research.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Enable auto-reload during development
    ) 