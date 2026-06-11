"""
Flask Web 界面 - RAG 问答系统交互界面
"""
import os
import json
from flask import Flask, request, jsonify, render_template_string

from .document_processor import DocumentProcessor
from .vector_store import VectorStore
from .qa_engine import QAEngine
from . import config

app = Flask(__name__)

# 全局实例
vector_store = VectorStore()
doc_processor = DocumentProcessor()
qa_engine = QAEngine(vector_store)

print("正在初始化系统...")
vector_store.initialize()
print("系统初始化完成！")

# 自动加载示例数据
if vector_store.get_document_count() == 0 and os.path.exists(config.DATA_DIR):
    chunks = doc_processor.process_documents(dir_path=config.DATA_DIR)
    if chunks:
        vector_store.add_documents(chunks)
        print(f"已自动加载 {len(chunks)} 个示例文档块")

# HTML 模板
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG 智能问答系统</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Microsoft YaHei', 'PingFang SC', sans-serif; background: #f0f2f5; height: 100vh; display: flex; }

        /* 左侧面板 */
        .sidebar { width: 340px; background: #fff; border-right: 1px solid #e0e0e0; display: flex; flex-direction: column; padding: 20px; overflow-y: auto; }
        .sidebar h2 { font-size: 18px; color: #1a1a1a; margin-bottom: 16px; }
        .sidebar h3 { font-size: 14px; color: #666; margin: 16px 0 8px; }

        .upload-area { border: 2px dashed #d0d0d0; border-radius: 8px; padding: 20px; text-align: center; margin-bottom: 12px; transition: border-color 0.3s; }
        .upload-area:hover { border-color: #4a90d9; }
        .upload-area input { display: none; }
        .upload-area label { cursor: pointer; color: #4a90d9; font-size: 14px; }

        .btn { display: block; width: 100%; padding: 10px 16px; border: none; border-radius: 6px; font-size: 14px; cursor: pointer; margin-bottom: 8px; transition: all 0.2s; }
        .btn-primary { background: #4a90d9; color: #fff; }
        .btn-primary:hover { background: #3a7bc8; }
        .btn-secondary { background: #e8f0fe; color: #4a90d9; }
        .btn-secondary:hover { background: #d0e0f8; }
        .btn-danger { background: #fff; color: #e74c3c; border: 1px solid #e74c3c; }
        .btn-danger:hover { background: #fdf0ef; }

        .status-msg { font-size: 12px; color: #666; margin: 4px 0 12px; padding: 8px; background: #f8f9fa; border-radius: 6px; min-height: 20px; }

        .info-card { background: #f8f9fa; border-radius: 8px; padding: 12px; margin-bottom: 12px; }
        .info-card .label { font-size: 12px; color: #888; }
        .info-card .value { font-size: 16px; font-weight: 600; color: #333; }
        .info-row { display: flex; justify-content: space-between; margin-bottom: 8px; }
        .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; }
        .status-dot.online { background: #27ae60; }
        .status-dot.offline { background: #e74c3c; }

        /* 右侧聊天 */
        .main { flex: 1; display: flex; flex-direction: column; background: #fff; margin: 12px 12px 12px 0; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 12px rgba(0,0,0,0.06); }
        .chat-header { padding: 16px 24px; border-bottom: 1px solid #f0f0f0; font-size: 16px; font-weight: 600; }
        .chat-messages { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 16px; }

        .message { max-width: 80%; padding: 12px 16px; border-radius: 12px; line-height: 1.6; font-size: 14px; white-space: pre-wrap; word-wrap: break-word; }
        .message.user { align-self: flex-end; background: #4a90d9; color: #fff; }
        .message.assistant { align-self: flex-start; background: #f0f2f5; color: #333; }
        .message .sources { margin-top: 12px; padding-top: 8px; border-top: 1px solid rgba(0,0,0,0.1); font-size: 12px; color: #888; }

        .loading { display: flex; align-items: center; gap: 8px; padding: 8px 16px; align-self: flex-start; color: #888; font-size: 14px; }
        .loading::after { content: ''; width: 20px; height: 20px; border: 2px solid #ddd; border-top-color: #4a90d9; border-radius: 50%; animation: spin 0.8s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }

        .chat-input-area { padding: 16px 24px; border-top: 1px solid #f0f0f0; display: flex; gap: 12px; }
        .chat-input-area input { flex: 1; padding: 12px 16px; border: 1px solid #e0e0e0; border-radius: 8px; font-size: 14px; outline: none; transition: border-color 0.3s; }
        .chat-input-area input:focus { border-color: #4a90d9; }
        .chat-input-area button { padding: 10px 24px; background: #4a90d9; color: #fff; border: none; border-radius: 8px; font-size: 14px; cursor: pointer; }
        .chat-input-area button:hover { background: #3a7bc8; }

        .toggle-row { display: flex; align-items: center; gap: 8px; margin-top: 8px; font-size: 13px; color: #666; }
        .toggle-row input { accent-color: #4a90d9; }
    </style>
</head>
<body>
    <div class="sidebar">
        <h2>知识库管理</h2>

        <div class="upload-area">
            <label for="file-input">点击上传文档</label>
            <input type="file" id="file-input" multiple accept=".txt,.pdf,.docx,.md">
            <div style="font-size:12px; color:#999; margin-top:4px;">支持 TXT, PDF, Word, Markdown</div>
        </div>
        <button class="btn btn-primary" onclick="uploadFiles()">上传到知识库</button>
        <div class="status-msg" id="upload-status"></div>

        <button class="btn btn-secondary" onclick="loadSampleData()">加载示例知识库</button>
        <div class="status-msg" id="sample-status"></div>

        <h3>系统状态</h3>
        <div class="info-card" id="sys-info">
            <div class="info-row"><span class="label">文档块数量</span><span class="value" id="doc-count">--</span></div>
            <div class="info-row"><span class="label">LLM 后端</span><span class="value"><span class="status-dot offline"></span><span id="llm-status">检测中...</span></span></div>
        </div>
        <button class="btn btn-secondary" onclick="refreshStatus()">刷新状态</button>

        <div style="margin-top: auto; padding-top: 16px;">
            <label class="toggle-row">
                <input type="checkbox" id="use-llm" checked onchange="toggleLLM()">
                <span>使用 LLM 生成回答 (需 Ollama)</span>
            </label>
        </div>
    </div>

    <div class="main">
        <div class="chat-header">RAG 智能问答系统</div>
        <div class="chat-messages" id="chat-messages">
            <div class="message assistant">
                你好！我是基于知识库的智能问答助手。

                快速开始：
                1. 点击左侧「加载示例知识库」
                2. 直接在下方输入问题即可提问
            </div>
        </div>
        <div class="chat-input-area">
            <input type="text" id="msg-input" placeholder="请输入您的问题... (按 Enter 发送)" onkeydown="if(event.key==='Enter')sendMessage()">
            <button onclick="sendMessage()">发送</button>
        </div>
    </div>

    <script>
        let useLLM = true;

        function toggleLLM() { useLLM = document.getElementById('use-llm').checked; }

        async function uploadFiles() {
            const files = document.getElementById('file-input').files;
            if (!files.length) { showStatus('upload-status', '请先选择文件', true); return; }

            const formData = new FormData();
            for (let f of files) formData.append('files', f);

            showStatus('upload-status', '正在上传处理...', false);
            try {
                const resp = await fetch('/api/upload', { method: 'POST', body: formData });
                const data = await resp.json();
                showStatus('upload-status', data.message);
                refreshStatus();
            } catch(e) {
                showStatus('upload-status', '上传失败: ' + e.message, true);
            }
        }

        async function loadSampleData() {
            showStatus('sample-status', '正在加载...', false);
            try {
                const resp = await fetch('/api/load-sample', { method: 'POST' });
                const data = await resp.json();
                showStatus('sample-status', data.message);
                refreshStatus();
            } catch(e) {
                showStatus('sample-status', '加载失败: ' + e.message, true);
            }
        }

        function showStatus(id, msg, isError) {
            const el = document.getElementById(id);
            el.textContent = (isError ? '错误: ' : '') + msg;
            el.style.color = isError ? '#e74c3c' : '#666';
        }

        async function sendMessage() {
            const input = document.getElementById('msg-input');
            const msg = input.value.trim();
            if (!msg) return;
            input.value = '';

            const container = document.getElementById('chat-messages');
            addMessage('user', msg);
            const loadingEl = addLoading();

            container.scrollTop = container.scrollHeight;

            try {
                const resp = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: msg, use_llm: useLLM })
                });
                const data = await resp.json();
                loadingEl.remove();
                addMessage('assistant', data.answer, data.sources);
            } catch(e) {
                loadingEl.remove();
                addMessage('assistant', '请求失败: ' + e.message);
            }

            container.scrollTop = container.scrollHeight;
        }

        function addMessage(role, content, sources) {
            const container = document.getElementById('chat-messages');
            const div = document.createElement('div');
            div.className = 'message ' + role;
            div.textContent = content;

            if (sources && sources.length > 0) {
                const srcDiv = document.createElement('div');
                srcDiv.className = 'sources';
                srcDiv.textContent = '参考资料: ' + sources.map(s => s.filename + '(相关度:' + s.score + ')').join(', ');
                div.appendChild(srcDiv);
            }

            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
        }

        function addLoading() {
            const container = document.getElementById('chat-messages');
            const div = document.createElement('div');
            div.className = 'loading';
            div.textContent = '正在思考...';
            container.appendChild(div);
            return div;
        }

        async function refreshStatus() {
            try {
                const resp = await fetch('/api/status');
                const data = await resp.json();
                document.getElementById('doc-count').textContent = data.vector_store.document_count;
                const llmEl = document.getElementById('llm-status');
                const dotEl = document.querySelector('.status-dot');
                const llm = data.llm;
                if (llm.backend === 'cloud') {
                    llmEl.textContent = '云端 API (' + llm.cloud_api.model + ')';
                    dotEl.className = 'status-dot online';
                } else if (llm.backend === 'ollama') {
                    llmEl.textContent = 'Ollama (' + llm.ollama.model + ')';
                    dotEl.className = 'status-dot online';
                } else {
                    llmEl.textContent = '未配置 (纯检索模式)';
                    dotEl.className = 'status-dot offline';
                }
            } catch(e) { console.error(e); }
        }

        async function clearChat() {
            try { await fetch('/api/clear', { method: 'POST' }); } catch(e) {}
            document.getElementById('chat-messages').innerHTML = '';
            addMessage('assistant', '对话已清空，请继续提问。');
        }

        refreshStatus();
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/status")
def api_status():
    stats = qa_engine.get_stats()
    return jsonify(stats)


@app.route("/api/upload", methods=["POST"])
def api_upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"message": "请选择要上传的文件"}), 400

    saved_paths = []
    for f in files:
        if f.filename:
            save_path = os.path.join(config.DATA_DIR, f.filename)
            os.makedirs(config.DATA_DIR, exist_ok=True)
            f.save(save_path)
            saved_paths.append(save_path)

    chunks = doc_processor.process_documents(file_paths=saved_paths)
    if not chunks:
        return jsonify({"message": "未能从文件中提取到有效文本内容"})

    count = vector_store.add_documents(chunks)
    filenames = [os.path.basename(p) for p in saved_paths]
    return jsonify({
        "message": f"成功处理 {len(filenames)} 个文件，添加了 {count} 个文本块。\n文件: {', '.join(filenames)}"
    })


@app.route("/api/load-sample", methods=["POST"])
def api_load_sample():
    if not os.path.exists(config.DATA_DIR):
        return jsonify({"message": "示例数据目录不存在"})

    chunks = doc_processor.process_documents(dir_path=config.DATA_DIR)
    if not chunks:
        return jsonify({"message": "未找到示例数据文件"})

    count = vector_store.add_documents(chunks)
    return jsonify({"message": f"已加载示例知识库，共 {count} 个文本块。你可以开始提问了！"})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json()
    message = data.get("message", "").strip()
    use_llm = data.get("use_llm", True)

    if not message:
        return jsonify({"answer": "请输入问题", "sources": [], "mode": "error"})

    result = qa_engine.query(message, use_llm=use_llm)
    return jsonify(result)


@app.route("/api/clear", methods=["POST"])
def api_clear():
    qa_engine.clear_history()
    return jsonify({"message": "对话历史已清空"})


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  RAG 智能问答系统已启动！")
    print("  访问地址: http://127.0.0.1:7860")
    print("=" * 50 + "\n")
    app.run(host="127.0.0.1", port=7860, debug=False)
