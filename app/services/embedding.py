import json
import os
import warnings

from langchain_chroma import Chroma
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

warnings.filterwarnings("ignore", message="Using default key encoder")
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

CHROMA_PATH = "./chroma_db"
FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "flooding_rag.json")


def _build_rich_content(chunk: dict) -> str:
    parts = []

    if chunk.get("location"):
        parts.append(f"위치: {chunk['location']}")

    title = chunk.get("title", "")
    if title:
        parts.append(f"[{title}]")

    if "content" in chunk:
        parts.append(chunk["content"])

    if "apply_time" in chunk:
        parts.append(f"신청 시간: {chunk['apply_time']}")

    if "steps" in chunk:
        steps = "\n".join(f"{i+1}. {s}" for i, s in enumerate(chunk["steps"]))
        parts.append(f"사용 방법:\n{steps}")

    # dormitory_music: 신청 방법이 methods 하위에 중첩
    if "methods" in chunk:
        for method in chunk["methods"]:
            method_steps = "\n".join(f"{i+1}. {s}" for i, s in enumerate(method.get("steps", [])))
            parts.append(f"{method['method']} 방법:\n{method_steps}")

    # club: 신청 절차가 apply 하위에 중첩
    if "apply" in chunk:
        apply = chunk["apply"]
        if "steps" in apply:
            steps = "\n".join(f"{i+1}. {s}" for i, s in enumerate(apply["steps"]))
            parts.append(f"동아리 신청 방법:\n{steps}")
        if "notes" in apply:
            notes = "\n".join(f"- {n}" for n in apply["notes"])
            parts.append(f"동아리 신청 주의사항:\n{notes}")

    # club: 개설 절차
    if "establishment" in chunk:
        est = chunk["establishment"]
        parts.append(f"동아리 개설: {est.get('description', '')}")
        if "notes" in est:
            notes = "\n".join(f"- {n}" for n in est["notes"])
            parts.append(f"동아리 개설 주의사항:\n{notes}")

    # settings: 다크모드 및 로그아웃 상세
    if "dark_mode" in chunk:
        dm = chunk["dark_mode"]
        parts.append(f"다크모드 전환 방법: {dm.get('how_to', '')}\n{dm.get('notes', '')}")

    if "logout" in chunk:
        lo = chunk["logout"]
        parts.append(f"로그아웃 방법: {lo.get('how_to', '')} {lo.get('result', '')}")

    if "notes" in chunk:
        notes = "\n".join(f"- {n}" for n in chunk["notes"])
        parts.append(f"주의사항:\n{notes}")

    if "keywords" in chunk:
        parts.append(f"키워드: {', '.join(chunk['keywords'])}")

    return "\n\n".join(parts)


def _make_dormitory_overview(documents: list[dict]) -> Document:
    dorm_docs = [d for d in documents if d.get("location") == "기숙사 화면"]
    lines = ["[기숙사 신청 방법 개요]\n기숙사 화면에서 신청할 수 있는 기능과 신청 방법입니다.\n"]
    for d in dorm_docs:
        lines.append(f"■ {d['title']}")
        if "apply_time" in d:
            lines.append(f"  신청 시간: {d['apply_time']}")
        steps = d.get("steps", [])
        if steps:
            for i, s in enumerate(steps, 1):
                lines.append(f"  {i}. {s}")
        else:
            lines.append(f"  {d.get('content', '')}")
        lines.append("")
    return Document(
        page_content="\n".join(lines),
        metadata={"id": "dormitory_overview", "section": "기숙사 신청 개요"},
    )


def load_documents_from_json(path: str) -> list[Document]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw = data["documents"]
    docs = []

    for chunk in raw:
        base_meta = {"id": chunk["id"], "section": chunk.get("title", "")}

        if chunk["id"] == "login":
            content = _build_rich_content(chunk)
            # 회원가입 질문에 fallback 대신 올바른 안내를 하도록 Q&A 형식으로 추가
            content += "\n\nQ. 회원가입은 어떻게 하나요?\nA. Flooding은 별도 회원가입이 없습니다. DataGSM OAuth 학교 계정으로 바로 로그인하시면 됩니다."
            docs.append(Document(page_content=content, metadata=base_meta))
        elif "items" in chunk:
            for item in chunk["items"]:
                docs.append(Document(
                    page_content=f"Q: {item['question']}\nA: {item['answer']}",
                    metadata=base_meta,
                ))
        elif "principles" in chunk:
            docs.append(Document(
                page_content="\n".join(chunk["principles"]),
                metadata=base_meta,
            ))
        else:
            docs.append(Document(
                page_content=_build_rich_content(chunk),
                metadata=base_meta,
            ))

    docs.append(_make_dormitory_overview(raw))
    return docs


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
        docs = load_documents_from_json(FILE_PATH)
        db = Chroma.from_documents(
            documents=docs,
            embedding=cached_embedder,
            persist_directory=CHROMA_PATH,
            collection_name="rag_collection",
            collection_metadata={"hnsw:space": "cosine"}
        )

    return db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 7, "fetch_k": 20, "lambda_mult": 0.8}
    )
