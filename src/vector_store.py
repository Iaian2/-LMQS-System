"""
向量存储模块 - 负责文本嵌入和向量检索
"""
import os
import uuid
from typing import List, Dict, Optional
from . import config
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class VectorStore:
    """向量存储：使用 ChromaDB + sentence-transformers 进行文档索引和检索"""

    def __init__(self, collection_name: str = "knowledge_base"):
        self.collection_name = collection_name
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self._initialized = False

    def initialize(self):
        """初始化嵌入模型和向量数据库"""
        if self._initialized:
            return

        print(f"正在加载嵌入模型: {config.EMBEDDING_MODEL_NAME} ...")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

        os.makedirs(config.CHROMA_DB_DIR, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(
            path=config.CHROMA_DB_DIR,
            settings=Settings(anonymized_telemetry=False),
        )

        # 获取或创建集合
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            print(f"已加载现有集合 '{self.collection_name}'，包含 {self.collection.count()} 个文档块")
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            print(f"已创建新集合 '{self.collection_name}'")

        self._initialized = True

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """将文本列表转换为嵌入向量"""
        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return embeddings.tolist()

    def add_documents(self, chunks: List[Dict]) -> int:
        """将文档块添加到向量数据库"""
        if not chunks:
            return 0

        if not self._initialized:
            self.initialize()

        ids = []
        documents = []
        metadatas = []
        embeddings = []

        texts_to_embed = [chunk["content"] for chunk in chunks]
        embeddings = self.embed_texts(texts_to_embed)

        for i, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            ids.append(chunk_id)
            documents.append(chunk["content"])
            metadatas.append(chunk.get("metadata", {}))

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        print(f"已添加 {len(chunks)} 个文档块到向量数据库")
        return len(chunks)

    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """检索与查询最相关的文档块"""
        if not self._initialized:
            self.initialize()

        top_k = top_k or config.TOP_K_RETRIEVAL

        if self.collection.count() == 0:
            return []

        query_embedding = self.embed_texts([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                retrieved.append({
                    "id": doc_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": 1.0 - results["distances"][0][i],  # cosine distance -> similarity
                })

        return retrieved

    def get_document_count(self) -> int:
        """获取已索引的文档数量"""
        if not self._initialized:
            self.initialize()
        return self.collection.count()

    def clear(self):
        """清空向量数据库"""
        if self._initialized and self.collection:
            self.chroma_client.delete_collection(name=self.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            print(f"已清空集合 '{self.collection_name}'")

    def get_stats(self) -> Dict:
        """获取向量数据库统计信息"""
        if not self._initialized:
            self.initialize()
        return {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "embedding_model": config.EMBEDDING_MODEL_NAME,
        }
