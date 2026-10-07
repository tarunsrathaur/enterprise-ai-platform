import os

from dotenv import load_dotenv


load_dotenv()


RAG_MODEL = os.getenv(
    "RAG_MODEL",
    "llama3.2:3b",
)