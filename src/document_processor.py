"""
文档处理模块 - 负责加载、解析和分块各类文档
"""
import os
import re
from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from . import config


class DocumentProcessor:
    """文档处理器：加载文档并分割为适合检索的文本块"""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""],
            length_function=len,
        )

    def load_file(self, file_path: str) -> str:
        """根据文件类型加载并提取文本内容"""
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".txt":
            return self._load_txt(file_path)
        elif ext == ".pdf":
            return self._load_pdf(file_path)
        elif ext == ".docx":
            return self._load_docx(file_path)
        elif ext == ".md":
            return self._load_txt(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}。支持的格式: .txt, .pdf, .docx, .md")

    def _load_txt(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _load_pdf(self, file_path: str) -> str:
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        except ImportError:
            raise ImportError("请安装 pypdf: pip install pypdf")

    def _load_docx(self, file_path: str) -> str:
        try:
            from docx import Document
            doc = Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        except ImportError:
            raise ImportError("请安装 python-docx: pip install python-docx")

    def load_directory(self, dir_path: str) -> List[Dict]:
        """加载目录下所有支持的文档，返回文档列表"""
        documents = []
        supported_exts = {".txt", ".pdf", ".docx", ".md"}

        for root, _, files in os.walk(dir_path):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext in supported_exts:
                    file_path = os.path.join(root, filename)
                    try:
                        text = self.load_file(file_path)
                        documents.append({
                            "content": text,
                            "source": file_path,
                            "filename": filename,
                        })
                    except Exception as e:
                        print(f"警告: 加载文件 {file_path} 失败: {e}")

        return documents

    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """将文本分割为重叠的文本块"""
        chunks = self.text_splitter.split_text(text)
        result = []
        for i, chunk in enumerate(chunks):
            chunk_meta = {
                "chunk_index": i,
                "chunk_size": len(chunk),
            }
            if metadata:
                chunk_meta.update(metadata)
            result.append({
                "content": chunk,
                "metadata": chunk_meta,
            })
        return result

    def process_documents(self, dir_path: str = None, file_paths: List[str] = None) -> List[Dict]:
        """
        处理文档：加载 -> 清洗 -> 分块
        返回: [{"content": "...", "metadata": {...}}, ...]
        """
        all_chunks = []

        if dir_path:
            documents = self.load_directory(dir_path)
            for doc in documents:
                chunks = self.chunk_text(
                    doc["content"],
                    metadata={"source": doc["source"], "filename": doc["filename"]}
                )
                all_chunks.extend(chunks)

        if file_paths:
            for file_path in file_paths:
                try:
                    text = self.load_file(file_path)
                    filename = os.path.basename(file_path)
                    chunks = self.chunk_text(
                        text,
                        metadata={"source": file_path, "filename": filename}
                    )
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"警告: 处理文件 {file_path} 失败: {e}")

        return all_chunks

    def clean_text(self, text: str) -> str:
        """清洗文本：去除多余空白、统一标点等"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\t\r]', ' ', text)
        text = text.strip()
        return text
