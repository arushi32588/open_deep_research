from setuptools import setup, find_packages

setup(
    name="open_deep_research",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pydantic>=2.0.0",
        "langchain>=0.1.0",
        "langchain-openai>=0.0.2",
        "langchain-community>=0.0.10",
        "faiss-cpu>=1.7.4",
        "pypdf>=3.0.0",
        "openpyxl>=3.1.0",
        "python-multipart>=0.0.5",
        "tavily-python>=0.1.9",
        "langgraph>=0.0.15",
        "requests>=2.31.0",
        "python-dotenv>=0.19.0",
    ],
    python_requires=">=3.8",
) 