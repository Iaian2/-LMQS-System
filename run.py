"""
启动脚本 - 一键启动 RAG 智能问答系统

使用方法:
    python run.py              # 启动 Web 界面
    python run.py --cli        # 命令行交互模式
    python run.py --ingest DIR # 将指定目录的文档导入知识库
"""
import os
import sys

# 必须在导入其他模块之前设置 HuggingFace 镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_web(port=7860):
    """启动 Web 界面"""
    from src.app import app
    print("\n" + "=" * 50)
    print("  RAG 智能问答系统已启动！")
    print(f"  访问地址: http://127.0.0.1:{port}")
    print("=" * 50 + "\n")
    import webbrowser
    webbrowser.open(f"http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)


def run_cli():
    """命令行交互模式"""
    from src.document_processor import DocumentProcessor
    from src.vector_store import VectorStore
    from src.qa_engine import QAEngine
    from src import config

    print("=" * 60)
    print("  RAG 智能问答系统 - 命令行模式")
    print("=" * 60)

    vector_store = VectorStore()
    vector_store.initialize()
    qa_engine = QAEngine(vector_store)
    doc_processor = DocumentProcessor()

    # 检查 Ollama 状态
    ollama_status = qa_engine.get_ollama_status()
    if ollama_status["available"]:
        print(f"\n[OK] Ollama 服务可用，模型: {config.OLLAMA_MODEL}")
        use_llm = True
    else:
        print("\n[提示] Ollama 服务不可用，将使用纯检索模式")
        print("  安装 Ollama: https://ollama.com")
        print("  然后运行: ollama pull qwen2.5:7b")
        use_llm = False

    # 加载已有数据
    doc_count = vector_store.get_document_count()
    print(f"\n[信息] 知识库中有 {doc_count} 个文档块")

    if doc_count == 0:
        print("[信息] 正在加载示例知识库...")
        sample_dir = config.DATA_DIR
        if os.path.exists(sample_dir):
            chunks = doc_processor.process_documents(dir_path=sample_dir)
            vector_store.add_documents(chunks)
            print(f"[信息] 已加载 {len(chunks)} 个文档块")
        else:
            print("[警告] 示例数据目录不存在")

    print("\n输入问题开始对话，输入 /clear 清空历史，输入 /quit 退出")
    print("-" * 60)

    while True:
        try:
            query = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not query:
            continue

        if query == "/quit":
            print("再见！")
            break
        elif query == "/clear":
            qa_engine.clear_history()
            print("[信息] 对话历史已清空")
            continue
        elif query == "/status":
            stats = qa_engine.get_stats()
            print(f"  文档块数量: {stats['vector_store']['document_count']}")
            print(f"  Ollama 状态: {'可用' if stats['ollama']['available'] else '不可用'}")
            continue

        print("\n助手: ", end="", flush=True)

        if use_llm:
            try:
                for chunk in qa_engine.rag_query(query, stream=True):
                    print(chunk, end="", flush=True)
                print()
            except Exception as e:
                print(f"\n[错误] {e}")
                print("切换到纯检索模式...")
                result = qa_engine.search_only(query)
                print(result["answer"])
        else:
            result = qa_engine.search_only(query)
            print(result["answer"])


def ingest_documents(dir_path: str):
    """将目录中的文档导入知识库"""
    from src.document_processor import DocumentProcessor
    from src.vector_store import VectorStore

    if not os.path.exists(dir_path):
        print(f"错误: 目录 '{dir_path}' 不存在")
        sys.exit(1)

    print(f"正在处理目录: {dir_path}")
    doc_processor = DocumentProcessor()
    vector_store = VectorStore()
    vector_store.initialize()

    chunks = doc_processor.process_documents(dir_path=dir_path)
    if not chunks:
        print("未找到支持的文档文件 (.txt, .pdf, .docx, .md)")
        sys.exit(1)

    count = vector_store.add_documents(chunks)
    print(f"完成！共导入 {count} 个文档块到知识库。")


def main():
    parser = argparse.ArgumentParser(description="RAG 智能问答系统")
    parser.add_argument("--cli", action="store_true", help="命令行交互模式")
    parser.add_argument("--ingest", type=str, metavar="DIR", help="将指定目录的文档导入知识库")
    parser.add_argument("--port", type=int, default=7860, help="Web 服务端口 (默认: 7860)")

    args = parser.parse_args()

    if args.ingest:
        ingest_documents(args.ingest)
    elif args.cli:
        run_cli()
    else:
        run_web(port=args.port)


if __name__ == "__main__":
    main()
