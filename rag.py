"""Coeur du mini-RAG : lecture des PDF, découpage, indexation et réponse.

Tout tourne en local :
- embeddings avec sentence-transformers
- base vectorielle Chroma (stockée dans ./chroma_db)
- génération avec un modèle servi par Ollama
"""

import hashlib
import os
from dataclasses import dataclass

import chromadb
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DB_DIR = os.getenv("RAG_DB_DIR", "chroma_db")
COLLECTION = "documents"
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "llama3.2")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


@dataclass
class Chunk:
    text: str
    source: str
    page: int


def read_pdf(path, name=None):
    """Retourne une liste de (numéro de page, texte)."""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        if text:
            pages.append((i, text))
    return pages


def split_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Découpe un texte en morceaux qui se chevauchent un peu.

    On essaie de couper sur un espace pour ne pas casser les mots.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start + size // 2:
                end = space
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return [c for c in chunks if c]


def make_chunks(path, name=None):
    name = name or os.path.basename(path)
    result = []
    for page_num, text in read_pdf(path):
        for piece in split_text(text):
            result.append(Chunk(text=piece, source=name, page=page_num))
    return result


class MiniRAG:
    def __init__(self, db_dir=DB_DIR, embed_model=EMBED_MODEL, llm_model=LLM_MODEL):
        self.embedder = SentenceTransformer(embed_model)
        self.client = chromadb.PersistentClient(path=db_dir)
        self.collection = self.client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        self.llm_model = llm_model

    # --- indexation -------------------------------------------------------

    def add_pdf(self, path, name=None):
        """Indexe un PDF. Renvoie le nombre de morceaux ajoutés."""
        chunks = make_chunks(path, name)
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        vectors = self.embedder.encode(texts, show_progress_bar=False).tolist()
        ids = [
            hashlib.md5(f"{c.source}-{c.page}-{i}-{c.text[:50]}".encode()).hexdigest()
            for i, c in enumerate(chunks)
        ]
        metas = [{"source": c.source, "page": c.page} for c in chunks]
        self.collection.upsert(ids=ids, documents=texts, embeddings=vectors, metadatas=metas)
        return len(chunks)

    def sources(self):
        data = self.collection.get(include=["metadatas"])
        return sorted({m["source"] for m in data["metadatas"]})

    def reset(self):
        self.client.delete_collection(COLLECTION)
        self.collection = self.client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    # --- recherche --------------------------------------------------------

    def search(self, question, k=4):
        if self.collection.count() == 0:
            return []
        vector = self.embedder.encode([question]).tolist()
        res = self.collection.query(
            query_embeddings=vector, n_results=min(k, self.collection.count())
        )
        hits = []
        for text, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            hits.append({"text": text, "source": meta["source"], "page": meta["page"],
                         "score": round(1 - dist, 3)})
        return hits

    # --- génération -------------------------------------------------------

    def build_prompt(self, question, hits):
        context = "\n\n".join(
            f"[{i}] ({h['source']}, page {h['page']})\n{h['text']}"
            for i, h in enumerate(hits, start=1)
        )
        return (
            "Tu es un assistant qui répond uniquement à partir des extraits fournis.\n"
            "Si la réponse n'est pas dans les extraits, dis que tu ne sais pas.\n"
            "Cite tes sources avec leur numéro entre crochets, par exemple [1].\n"
            "Réponds dans la langue de la question.\n\n"
            f"Extraits :\n{context}\n\n"
            f"Question : {question}\n"
            "Réponse :"
        )

    def ask(self, question, k=4):
        hits = self.search(question, k)
        if not hits:
            return "Aucun document indexé pour l'instant.", []
        prompt = self.build_prompt(question, hits)
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": self.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
                timeout=300,
            )
            r.raise_for_status()
        except requests.ConnectionError:
            raise RuntimeError(
                f"Impossible de joindre Ollama sur {OLLAMA_URL}. Lance `ollama serve`."
            )
        answer = r.json()["message"]["content"].strip()
        return answer, hits
