from setuptools import setup, find_packages

setup(
    name="open_deep_research",
    version="0.0.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "fastapi",
        "uvicorn",
        "pydantic",
        "python-dotenv",
        "langgraph",
        "langchain",
        "langchain-openai",
        "langchain-anthropic",
        "faiss-cpu",
        "pypdf",
        "openpyxl",
    ],
    python_requires=">=3.9",
) 