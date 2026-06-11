# RAG 智能问答系统

基于 **检索增强生成（RAG）** 技术的智能问答系统，支持多格式知识库文档上传、语义检索和 LLM 智能回答生成。

## 🏗️ 系统架构

```
用户提问 → 向量检索(Top-K) → 上下文构建 → LLM 生成 → 返回回答
                ↑
          ChromaDB 向量数据库
                ↑
    文档上传 → 文本分块 → 向量嵌入
```

## ✨ 功能特点

- **多格式文档支持**：TXT、PDF、Word(.docx)、Markdown
- **语义检索**：基于 sentence-transformers 多语言模型的 384 维向量检索
- **智能生成**：支持云端 API（DeepSeek/OpenAI 兼容）和本地模型（Ollama）
- **三级降级**：云端 API → Ollama 本地模型 → 纯检索模式
- **Web 界面**：Flask 构建的现代化聊天界面
- **CLI 模式**：支持命令行交互
- **流式输出**：LLM 回答实时流式显示

## 🚀 快速开始

### 环境要求

- Python 3.9+
- （可选）Ollama 本地模型
- （可选）DeepSeek / OpenAI API Key

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境变量

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 填入你的 API Key
# LLM_API_KEY=你的API密钥
```

或直接设置环境变量：

```bash
# Windows PowerShell
$env:LLM_API_KEY = "你的API密钥"

# Linux / macOS
export LLM_API_KEY="你的API密钥"
```

### 启动系统

```bash
# Web 界面模式（默认）
python run.py

# 命令行模式
python run.py --cli

# 批量导入文档
python run.py --ingest <文档目录路径>
```

启动后访问：**http://127.0.0.1:7860**

### 配置 LLM

**方式一：云端 API（推荐）**

默认使用 DeepSeek API，可通过环境变量切换：
- `LLM_API_KEY`：API 密钥（必需）
- `LLM_API_BASE`：API 地址（默认: https://api.deepseek.com）
- `LLM_MODEL`：模型名称（默认: deepseek-chat）

**方式二：Ollama 本地模型**

1. 安装 [Ollama](https://ollama.com)
2. 下载模型：`ollama pull qwen2.5:7b`
3. 确保 Ollama 服务运行在 `http://localhost:11434`

## 📁 项目结构

```
├── run.py                       # 启动脚本（Web/CLI/文档导入）
├── requirements.txt             # Python 依赖
├── .env.example                 # 环境变量配置模板
├── LICENSE                      # MIT 许可证
├── src/                         # 核心源码包
│   ├── __init__.py
│   ├── app.py                   # Flask Web 服务与 REST API
│   ├── config.py                # 全局配置（模型、分块、检索参数）
│   ├── qa_engine.py             # RAG 问答引擎（检索 + LLM 生成）
│   ├── vector_store.py          # 向量存储（ChromaDB + 嵌入模型）
│   └── document_processor.py    # 文档加载、解析与分块
├── data/                        # 示例知识库
│   ├── 自然语言处理基础.txt
│   ├── 深度学习与Transformer.txt
│   └── Python编程知识.txt
└── chroma_db/                   # 向量数据库（运行时自动创建）
```

## 🔧 API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web 问答界面 |
| `/api/chat` | POST | 发送问题，获取回答 |
| `/api/upload` | POST | 上传文档到知识库 |
| `/api/load-sample` | POST | 加载示例知识库 |
| `/api/status` | GET | 获取系统状态 |
| `/api/clear` | POST | 清空对话历史 |

## 📦 依赖项

- **Flask** — Web 框架
- **ChromaDB** — 向量数据库
- **sentence-transformers** — 文本嵌入模型
- **langchain-text-splitters** — 文档分块
- **pypdf** — PDF 解析
- **python-docx** — Word 文档解析
- **requests** — HTTP 请求

## 👤 作者

- **邮箱**：3108565030@qq.com
- **GitHub**：[https://github.com/LMQS-Commits](https://github.com/LMQS-Commits)

## 📄 许可证

本项目基于 MIT 许可证开源，仅用于学习交流目的。
