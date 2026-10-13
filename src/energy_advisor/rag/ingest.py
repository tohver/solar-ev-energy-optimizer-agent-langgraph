"""Build the vector index of the knowledge base: load documents, split into chunks, embed, persist."""

import logging
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from energy_advisor.config import RAGSettings

logger = logging.getLogger(__name__)
CHROMA_DB_FILE = "chroma.sqlite3"


class IndexExistsError(RuntimeError):
    pass


@dataclass(frozen=True)
class IndexResult:
    documents: int
    chunks: int
    persist_dir: Path


def make_chroma_client(persist_dir: Path) -> ClientAPI:
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
    )


def collection_exists(client: ClientAPI, name: str) -> bool:
    return any(getattr(c, "name", c) == name for c in client.list_collections())


def load_documents(knowledge_base_dir: Path) -> list[Document]:
    """Every ``*.txt`` file becomes one document; ``source`` is the file name the agent cites."""
    paths = sorted(knowledge_base_dir.glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"No .txt documents found in {knowledge_base_dir}")
    return [Document(page_content=p.read_text(encoding="utf-8"), metadata={"source": p.name}) for p in paths]


def split_documents(documents: list[Document], rag: RAGSettings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=rag.chunk_size, chunk_overlap=rag.chunk_overlap)
    return splitter.split_documents(documents)


def build_index(
    knowledge_base_dir: Path,
    persist_dir: Path,
    embeddings: Embeddings,
    rag: RAGSettings,
    rebuild: bool = False,
) -> IndexResult:
    """Create the Chroma collection. With ``rebuild`` an existing collection is replaced, never appended to."""
    documents = load_documents(knowledge_base_dir)
    chunks = split_documents(documents, rag)

    client = make_chroma_client(persist_dir)
    if collection_exists(client, rag.collection_name):
        if not rebuild:
            raise IndexExistsError(
                f"Index already exists in {persist_dir}. Rebuild it with `energy-advisor build-index --rebuild`."
            )
        client.delete_collection(rag.collection_name)

    Chroma.from_documents(
        documents=chunks, embedding=embeddings, client=client, collection_name=rag.collection_name
    )
    _remove_orphaned_segments(persist_dir)
    logger.info("Indexed %d documents as %d chunks in %s", len(documents), len(chunks), persist_dir)
    return IndexResult(len(documents), len(chunks), persist_dir)


def _remove_orphaned_segments(persist_dir: Path) -> None:
    """Chroma keeps the folder of a deleted collection on disk; remove folders no longer referenced."""
    db_file = persist_dir / CHROMA_DB_FILE
    if not db_file.exists():
        return
    try:
        with sqlite3.connect(db_file) as con:
            active = {row[0] for row in con.execute("SELECT id FROM segments")}
    except sqlite3.Error as exc:
        logger.debug("Could not read Chroma segments: %s", exc)
        return
    for path in persist_dir.iterdir():
        if path.is_dir() and path.name not in active:
            shutil.rmtree(path, ignore_errors=True)  # a still-locked folder is removed next time
