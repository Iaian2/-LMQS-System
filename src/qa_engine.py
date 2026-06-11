"""
问答引擎模块 - 实现 RAG 检索增强生成的核心逻辑
支持: 云端 API (DeepSeek/阿里百炼/OpenAI等) > Ollama 本地模型 > 纯检索
"""
import json
from typing import List, Dict, Optional, Generator
import requests
from . import config
from .vector_store import VectorStore


class QAEngine:
    """RAG 问答引擎：检索 + 生成"""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history = 10

    def _build_context(self, retrieved_docs: List[Dict]) -> str:
        """构建上下文文本，将检索到的文档拼接起来"""
        if not retrieved_docs:
            return ""
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            source = doc.get("metadata", {}).get("filename", "未知来源")
            context_parts.append(f"[参考资料 {i + 1}] (来源: {source})\n{doc['content']}")
        return "\n\n".join(context_parts)

    # ============================================================
    # 云端 API 调用（OpenAI 兼容接口）
    # ============================================================

    def _check_cloud_api_available(self) -> bool:
        """检查云端 API 是否已配置"""
        return bool(config.LLM_API_KEY)

    def _call_cloud_api(self, messages: List[Dict]) -> str:
        """调用 OpenAI 兼容的云端 API（非流式）"""
        url = f"{config.LLM_API_BASE}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {config.LLM_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": config.LLM_API_MODEL,
            "messages": messages,
            "temperature": config.LLM_API_TEMPERATURE,
            "max_tokens": config.LLM_API_MAX_TOKENS,
            "stream": False,
        }
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    def _call_cloud_api_stream(self, messages: List[Dict]) -> Generator:
        """调用 OpenAI 兼容的云端 API（流式）"""
        url = f"{config.LLM_API_BASE}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {config.LLM_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": config.LLM_API_MODEL,
            "messages": messages,
            "temperature": config.LLM_API_TEMPERATURE,
            "max_tokens": config.LLM_API_MAX_TOKENS,
            "stream": True,
        }
        response = requests.post(url, json=payload, headers=headers, stream=True, timeout=120)
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    # ============================================================
    # Ollama 本地模型调用
    # ============================================================

    def _check_ollama_available(self) -> bool:
        """检查 Ollama 服务是否可用"""
        try:
            response = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3)
            return response.status_code == 200
        except Exception:
            return False

    def _call_ollama(self, messages: List[Dict]) -> str:
        """调用 Ollama API（非流式）"""
        url = f"{config.OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model": config.OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": config.OLLAMA_TEMPERATURE,
                "num_predict": config.OLLAMA_MAX_TOKENS,
            },
        }
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result.get("message", {}).get("content", "")

    def _call_ollama_stream(self, messages: List[Dict]) -> Generator:
        """调用 Ollama API（流式）"""
        url = f"{config.OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model": config.OLLAMA_MODEL,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": config.OLLAMA_TEMPERATURE,
                "num_predict": config.OLLAMA_MAX_TOKENS,
            },
        }
        response = requests.post(url, json=payload, stream=True, timeout=120)
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]
                except json.JSONDecodeError:
                    continue

    # ============================================================
    # LLM 后端选择
    # ============================================================

    def _get_active_backend(self) -> str:
        """获取当前可用的最佳 LLM 后端: 'cloud' | 'ollama' | 'none'"""
        if self._check_cloud_api_available():
            return "cloud"
        if self._check_ollama_available():
            return "ollama"
        return "none"

    def _call_llm(self, messages: List[Dict]) -> str:
        """统一 LLM 调用入口（非流式），自动选择最佳后端"""
        backend = self._get_active_backend()
        if backend == "cloud":
            return self._call_cloud_api(messages)
        elif backend == "ollama":
            return self._call_ollama(messages)
        else:
            raise RuntimeError("没有可用的 LLM 后端")

    def _call_llm_stream(self, messages: List[Dict]) -> Generator:
        """统一 LLM 调用入口（流式），自动选择最佳后端"""
        backend = self._get_active_backend()
        if backend == "cloud":
            return self._call_cloud_api_stream(messages)
        elif backend == "ollama":
            return self._call_ollama_stream(messages)
        else:
            raise RuntimeError("没有可用的 LLM 后端")

    # ============================================================
    # 问答接口
    # ============================================================

    def search_only(self, query: str, top_k: int = None) -> Dict:
        """纯检索模式（不依赖 LLM）：返回相关文档"""
        retrieved_docs = self.vector_store.search(query, top_k=top_k)

        if not retrieved_docs:
            return {
                "answer": "未在知识库中找到与您问题相关的内容。请尝试上传更多相关文档或换个问题。",
                "sources": [],
                "mode": "search_only",
            }

        context_parts = []
        sources = []
        for i, doc in enumerate(retrieved_docs):
            source = doc.get("metadata", {}).get("filename", "未知来源")
            sources.append({"filename": source, "score": round(doc["score"], 3), "index": i + 1})
            context_parts.append(
                f"【相关内容 {i + 1}】(来源: {source}, 相关度: {doc['score']:.2f})\n{doc['content']}"
            )

        answer = "根据知识库检索，找到以下相关内容：\n\n" + "\n\n---\n\n".join(context_parts)
        answer += "\n\n---\n提示: 配置 LLM_API_KEY 环境变量后可使用 AI 智能生成回答。"

        return {"answer": answer, "sources": sources, "mode": "search_only"}

    def rag_query(self, query: str, top_k: int = None, stream: bool = False):
        """RAG 问答：检索 + LLM 生成"""
        retrieved_docs = self.vector_store.search(query, top_k=top_k)
        context = self._build_context(retrieved_docs)

        if context:
            system_prompt = config.RAG_SYSTEM_PROMPT.format(context=context)
        else:
            system_prompt = config.NO_CONTEXT_PROMPT

        messages = [{"role": "system", "content": system_prompt}]
        for hist in self.conversation_history[-self.max_history:]:
            messages.append(hist)
        messages.append({"role": "user", "content": query})

        if stream:
            return self._stream_response(messages, retrieved_docs)
        else:
            answer = self._call_llm(messages)
            self._add_to_history(query, answer)
            return self._format_response(answer, retrieved_docs)

    def _stream_response(self, messages: List[Dict], retrieved_docs: List[Dict]) -> Generator:
        """流式响应生成器"""
        full_answer = ""
        try:
            for chunk in self._call_llm_stream(messages):
                full_answer += chunk
                yield chunk
        finally:
            if full_answer:
                self._add_to_history(messages[-1]["content"], full_answer)

    def _add_to_history(self, query: str, answer: str):
        """添加到对话历史"""
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": answer})
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]

    def _format_response(self, answer: str, retrieved_docs: List[Dict]) -> Dict:
        """格式化响应"""
        sources = []
        for i, doc in enumerate(retrieved_docs):
            source = doc.get("metadata", {}).get("filename", "未知来源")
            sources.append({
                "filename": source,
                "score": round(doc["score"], 3),
                "index": i + 1,
            })
        return {"answer": answer, "sources": sources, "mode": "rag"}

    def query(self, query: str, use_llm: bool = True, top_k: int = None) -> Dict:
        """统一查询接口：云端 API > Ollama > 纯检索"""
        if use_llm:
            backend = self._get_active_backend()
            if backend != "none":
                try:
                    return self.rag_query(query, top_k=top_k)
                except Exception as e:
                    print(f"LLM 调用失败: {e}，回退到纯检索模式。")
            else:
                print("提示: 未配置 LLM（设置 LLM_API_KEY 或安装 Ollama），使用纯检索模式。")
        return self.search_only(query, top_k=top_k)

    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []

    def get_llm_status(self) -> Dict:
        """获取 LLM 状态信息"""
        backend = self._get_active_backend()
        status = {
            "backend": backend,
            "cloud_api": {
                "configured": self._check_cloud_api_available(),
                "base_url": config.LLM_API_BASE,
                "model": config.LLM_API_MODEL,
            },
            "ollama": {
                "available": self._check_ollama_available(),
                "base_url": config.OLLAMA_BASE_URL,
                "model": config.OLLAMA_MODEL,
            },
        }
        return status

    def get_stats(self) -> Dict:
        """获取系统统计信息"""
        return {
            "vector_store": self.vector_store.get_stats(),
            "llm": self.get_llm_status(),
            "conversation_turns": len(self.conversation_history) // 2,
        }
