from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings("ignore", message="Using default key encoder")
load_dotenv()

CHROMA_PATH = "./chroma_db"
FILE_PATH = os.getenv("FILE_PATH")

def get_retriever():
    underlying_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    store = LocalFileStore("./cache/")
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings,
        store,
        namespace=underlying_embeddings.model
    )

    if os.path.exists(CHROMA_PATH):
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=cached_embedder,
            collection_name="rag_collection"
        )
    else:
        loader = PyMuPDFLoader(FILE_PATH)
        docs = loader.load()
        result = RecursiveCharacterTextSplitter(
            chunk_size=700, chunk_overlap=100
        ).split_documents(docs)
        db = Chroma.from_documents(
            documents=result,
            embedding=cached_embedder,
            persist_directory=CHROMA_PATH,
            collection_name="rag_collection",
            collection_metadata={"hnsw:space": "cosine"}
        )

    return db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.7}
    )