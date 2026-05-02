import os
import sys
import json
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from flask import Flask, request, jsonify

# --- 全局配置 ---
app = Flask(__name__)
SELECTED_FILE = None
root = None

# 用于线程间传递数据的全局容器
GUI_DATA = {
    "old_code": "",
    "new_code": "",
    "result": None,  # "accept", "reject"
    "event": None
}


# ----------------- Tkinter GUI 逻辑 -----------------

def show_diff_window():
    """弹出对比窗口（由主线程的事件处理器调用）"""
    global GUI_DATA, root

    top = tk.Toplevel(root)
    top.title("代码审核 - 请确认修改")
    top.geometry("1200x700")
    top.configure(bg="#2b2b2b")

    # 使窗口始终在最上层，确保能看到
    top.attributes("-topmost", True)
    top.lift()
    top.focus_force()

    # 界面布局
    frame = tk.Frame(top, bg="#2b2b2b")
    frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # 左侧：旧代码
    old_frame = tk.LabelFrame(frame, text="当前文件 (旧)", fg="#ffaa66", bg="#2b2b2b", font=("微软雅黑", 12))
    old_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
    old_text = scrolledtext.ScrolledText(old_frame, wrap=tk.WORD, bg="#1e1e1e", fg="#cccccc")
    old_text.pack(fill=tk.BOTH, expand=True)
    old_text.insert(tk.END, GUI_DATA["old_code"])
    old_text.config(state=tk.DISABLED)

    # 右侧：新代码
    new_frame = tk.LabelFrame(frame, text="建议修改 (新)", fg="#9bcd6c", bg="#2b2b2b", font=("微软雅黑", 12))
    new_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
    new_text = scrolledtext.ScrolledText(new_frame, wrap=tk.WORD, bg="#1e1e1e", fg="#cccccc")
    new_text.pack(fill=tk.BOTH, expand=True)
    new_text.insert(tk.END, GUI_DATA["new_code"])

    # 底部按钮逻辑
    def on_accept():
        final_code = new_text.get("1.0", tk.END).rstrip('\n')
        try:
            with open(SELECTED_FILE, "w", encoding="utf-8") as f:
                f.write(final_code)
            GUI_DATA["result"] = "accept"
        except Exception as e:
            messagebox.showerror("错误", f"写入文件失败: {str(e)}")
            GUI_DATA["result"] = "error"
        top.destroy()

    def on_reject():
        GUI_DATA["result"] = "reject"
        top.destroy()

    btn_frame = tk.Frame(top, bg="#2b2b2b")
    btn_frame.pack(pady=15)

    tk.Button(btn_frame, text="✅ 接受并覆盖", command=on_accept, bg="#4caf50", fg="white", padx=30, pady=8,
              font=("微软雅黑", 11, "bold")).pack(side=tk.LEFT, padx=20)
    tk.Button(btn_frame, text="❌ 拒绝修改", command=on_reject, bg="#f44336", fg="white", padx=30, pady=8,
              font=("微软雅黑", 11, "bold")).pack(side=tk.LEFT, padx=20)

    # 等待窗口关闭
    top.wait_window()

    # 通知 Flask 线程结果已出
    if GUI_DATA["event"]:
        GUI_DATA["event"].set()


def handle_gui_event(event):
    """接收虚拟事件并触发窗口"""
    show_diff_window()


# ----------------- Flask Web 逻辑 -----------------

@app.route("/code", methods=["GET"])
def get_code():
    if not SELECTED_FILE or not os.path.exists(SELECTED_FILE):
        return jsonify({"error": "no_file_selected"}), 400
    with open(SELECTED_FILE, "r", encoding="utf-8") as f:
        return jsonify({"file": SELECTED_FILE, "code": f.read()})


@app.route("/update", methods=["POST"])
def update_code():
    global GUI_DATA, root

    if not SELECTED_FILE or not os.path.exists(SELECTED_FILE):
        return jsonify({"error": "no_file"}), 400

    # 1. 解析输入数据
    data = request.get_json(silent=True)
    new_code = ""
    if data and "code" in data:
        new_code = data["code"]
    else:
        # 兼容纯文本提交
        new_code = request.get_data(as_text=True)

    if not new_code.strip():
        return jsonify({"error": "empty_code"}), 400

    # 2. 读取当前文件
    with open(SELECTED_FILE, "r", encoding="utf-8") as f:
        old_code = f.read()

    if old_code.strip() == new_code.strip():
        return jsonify({"status": "no_change"})

    # 3. 准备同步对象
    event = threading.Event()
    GUI_DATA["old_code"] = old_code
    GUI_DATA["new_code"] = new_code
    GUI_DATA["result"] = None
    GUI_DATA["event"] = event

    # 4. 关键：向主线程发送虚拟事件，要求弹窗
    # event_generate 是线程安全的
    root.event_generate("<<ShowDiffEvent>>", when="tail")

    # 5. 等待用户操作（超时设为 5 分钟）
    if not event.wait(timeout=300):
        return jsonify({"error": "timeout"}), 408

    # 6. 返回结果
    res = GUI_DATA["result"]
    if res == "accept":
        return jsonify({"status": "success", "action": "accepted"})
    elif res == "reject":
        return jsonify({"status": "rejected", "action": "rejected"})
    else:
        return jsonify({"error": "unknown"}), 500


# ----------------- 启动程序 -----------------

def main():
    global root, SELECTED_FILE

    # 1. 先创建 Tkinter 根窗口
    root = tk.Tk()
    root.title("CodeReviewService")
    # 隐藏主窗口（只显示弹出的对比窗口）
    root.withdraw()

    # 绑定自定义虚拟事件
    root.bind("<<ShowDiffEvent>>", handle_gui_event)

    # 2. 选择文件
    SELECTED_FILE = filedialog.askopenfilename(
        title="请选择要监控的 Python 文件",
        filetypes=[("Python files", "*.py"), ("All files", "*.*")]
    )

    if not SELECTED_FILE:
        print("未选择文件，程序退出。")
        return

    print(f"[系统] 已监控文件: {SELECTED_FILE}")
    print(f"[系统] 服务已启动: http://127.0.0.1:5500/update")
    print(f"[系统] 等待 HTTP 请求以弹出对比窗口...")

    # 3. 在子线程启动 Flask
    threading.Thread(
        target=app.run,
        kwargs={"host": "0.0.0.0", "port": 5500, "threaded": True, "use_reloader": False, "debug": False},
        daemon=True
    ).start()

    # 4. 启动 Tkinter 主循环（主线程阻塞在这里）
    root.mainloop()


if __name__ == "__main__":
    main()