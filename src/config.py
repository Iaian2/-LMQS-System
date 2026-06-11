"""
系统配置模块
"""
import os

# HuggingFace 镜像设置（国内用户使用）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 向量数据库存储路径
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# 知识库文档存储路径
DATA_DIR = os.path.join(BASE_DIR, "data")

# 嵌入模型配置
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# 文档分块配置
CHUNK_SIZE = 500          # 每个文本块的最大字符数
CHUNK_OVERLAP = 50        # 文本块之间的重叠字符数

# 检索配置
TOP_K_RETRIEVAL = 5       # 检索时返回的最相关文档块数量

# ============================================================
# LLM 后端配置（优先级：云端 API > Ollama > 纯检索）
# ============================================================

# --- 云端 API（推荐，无需下载模型）---
# 支持任意 OpenAI 兼容接口（DeepSeek / 阿里百炼 / 智谱 / OpenAI 等）
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")         # 从环境变量读取 API Key
LLM_API_BASE = os.environ.get("LLM_API_BASE", "https://api.deepseek.com")
LLM_API_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
LLM_API_TEMPERATURE = 0.7
LLM_API_MAX_TOKENS = 1024

# --- 本地 Ollama（需先下载模型，速度慢）---
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_TEMPERATURE = 0.7
OLLAMA_MAX_TOKENS = 1024

# 系统 prompt 模板
RAG_SYSTEM_PROMPT = """你是一个基于知识库的智能问答助手。请根据提供的参考资料回答用户的问题。

要求：
1. 如果参考资料中包含答案，请基于资料内容进行回答，保持准确和简洁。
2. 如果参考资料不足以回答问题，请如实说明，不要编造信息。
3. 回答时请使用中文，保持专业和友好的语气。
4. 如果适用，可以在回答末尾注明信息来源。

参考资料：
{context}"""

NO_CONTEXT_PROMPT = """你是一个智能问答助手。请根据你的知识回答用户的问题。
注意：当前没有找到相关的参考资料，请基于你的通用知识回答，并提醒用户此回答并非基于知识库内容。"""
