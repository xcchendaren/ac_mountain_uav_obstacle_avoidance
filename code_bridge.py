import os
import sys
import json
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from flask import Flask, request, jsonify

# ---------- 全局变量 ----------
app = Flask(__name__)
PROJECT_ROOT = None          # 项目根目录（或所选文件所在目录）
DEFAULT_FILE = None          # 若用户选择了单个文件，则保存文件名；否则为 None
root = None                  # tkinter 根窗口

GUI_DATA = {
    "files": {},             # {relative_path: {"old": str, "new": str, "accepted": False}}
    "result_event": None,
    "final_result": None,
}


# ---------- 工具函数 ----------
def safe_join(base, relative_path):
    """安全拼接路径，防止路径遍历攻击"""
    relative_path = relative_path.replace('\\', '/').strip('/')
    if not relative_path or '..' in relative_path.split('/'):
        raise ValueError(f"非法文件路径: {relative_path}")
    target = os.path.abspath(os.path.join(base, relative_path))
    if not target.startswith(os.path.abspath(base)):
        raise ValueError(f"越权访问: {relative_path}")
    return target


def collect_project_files(base_dir):
    """
    递归收集项目根目录下所有 .py 文件（忽略常见非源码目录与隐藏文件）
    返回字典: {相对路径: 文件内容}
    """
    ignore_dirs = {".git", "__pycache__", ".venv", "venv", ".idea", ".pytest_cache"}
    py_files = {}
    base_abs = os.path.abspath(base_dir)
    for dirpath, dirnames, filenames in os.walk(base_dir):
        # 忽略特定目录
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs and not d.startswith(".")]
        for fname in filenames:
            if fname.endswith(".py") and not fname.startswith("."):
                abs_path = os.path.join(dirpath, fname)
                try:
                    with open(abs_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception:
                    continue  # 读取失败则跳过
                rel_path = os.path.relpath(abs_path, base_abs).replace('\\', '/')
                py_files[rel_path] = content
    return py_files


def apply_edits(edits: list) -> dict:
    """
    将编辑列表应用到对应文件，返回 {相对路径: {"old":旧内容, "new":新内容}}
    若文件不存在，旧内容为空；若 DEFAULT_FILE 有值且 file 字段为空或缺失，则使用 DEFAULT_FILE
    """
    global DEFAULT_FILE
    file_edits = {}
    for edit in edits:
        fname = edit.get("file", "").strip()
        if not fname and DEFAULT_FILE:
            fname = DEFAULT_FILE
        if not fname:
            continue   # 忽略没有目标文件的编辑
        old_snippet = edit.get("old_code", "")
        new_snippet = edit.get("new_code", "")
        file_edits.setdefault(fname, []).append((old_snippet, new_snippet))

    result = {}
    for rel_path, replacements in file_edits.items():
        abs_path = safe_join(PROJECT_ROOT, rel_path)
        if os.path.exists(abs_path):
            with open(abs_path, "r", encoding="utf-8") as f:
                old_content = f.read()
        else:
            old_content = ""

        new_content = old_content
        for old_str, new_str in replacements:
            new_content = new_content.replace(old_str, new_str, 1)

        result[rel_path] = {
            "old": old_content,
            "new": new_content,
            "accepted": False
        }
    return result


# ---------- GUI 多文件对比窗口 ----------
class MultiDiffWindow:
    def __init__(self, master, files_info):
        self.master = master
        self.files_info = files_info  # {path: {"old":..., "new":..., "accepted": False}}
        self.file_list = list(files_info.keys())
        self.current_file = None

        self.top = tk.Toplevel(master)
        self.top.title("代码审核 - 多文件修改确认")
        self.top.geometry("1200x750")
        self.top.configure(bg="#2b2b2b")
        self.top.attributes("-topmost", True)
        self.top.lift()
        self.top.focus_force()

        self.create_widgets()
        self.load_file_list()
        self.top.protocol("WM_DELETE_WINDOW", self.on_close)

        self.wait_var = tk.IntVar()
        self.top.wait_variable(self.wait_var)

    def create_widgets(self):
        main_pw = tk.PanedWindow(self.top, orient=tk.HORIZONTAL, bg="#2b2b2b", sashrelief=tk.RAISED)
        main_pw.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = tk.Frame(main_pw, bg="#2b2b2b")
        tk.Label(left_frame, text="修改文件列表", fg="#ffaa66", bg="#2b2b2b", font=("微软雅黑", 12, "bold")).pack(pady=5)
        self.listbox = tk.Listbox(left_frame, bg="#1e1e1e", fg="#cccccc", selectbackground="#4a6a8a")
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.listbox.bind("<<ListboxSelect>>", self.on_select_file)

        right_frame = tk.Frame(main_pw, bg="#2b2b2b")
        paned = tk.PanedWindow(right_frame, orient=tk.VERTICAL, bg="#2b2b2b", sashrelief=tk.RAISED)
        paned.pack(fill=tk.BOTH, expand=True)

        old_fr = tk.LabelFrame(paned, text="当前文件内容 (旧)", fg="#ffaa66", bg="#2b2b2b", font=("微软雅黑", 11))
        self.old_text = scrolledtext.ScrolledText(old_fr, wrap=tk.WORD, bg="#1e1e1e", fg="#cccccc")
        self.old_text.pack(fill=tk.BOTH, expand=True)
        paned.add(old_fr, stretch="always")

        new_fr = tk.LabelFrame(paned, text="修改后内容 (新)", fg="#9bcd6c", bg="#2b2b2b", font=("微软雅黑", 11))
        self.new_text = scrolledtext.ScrolledText(new_fr, wrap=tk.WORD, bg="#1e1e1e", fg="#cccccc")
        self.new_text.pack(fill=tk.BOTH, expand=True)
        paned.add(new_fr, stretch="always")

        main_pw.add(left_frame, stretch="never", width=200)
        main_pw.add(right_frame, stretch="always")

        btn_frame = tk.Frame(self.top, bg="#2b2b2b")
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="✅ 接受当前文件", command=self.accept_current, bg="#4caf50", fg="white",
                  padx=20, pady=5, font=("微软雅黑", 10, "bold")).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="❌ 拒绝当前文件", command=self.reject_current, bg="#f44336", fg="white",
                  padx=20, pady=5, font=("微软雅黑", 10, "bold")).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="✔ 接受所有", command=self.accept_all, bg="#2196f3", fg="white",
                  padx=20, pady=5, font=("微软雅黑", 10, "bold")).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="✘ 拒绝所有", command=self.reject_all, bg="#ff9800", fg="white",
                  padx=20, pady=5, font=("微软雅黑", 10, "bold")).pack(side=tk.LEFT, padx=10)

    def load_file_list(self):
        for f in self.file_list:
            self.listbox.insert(tk.END, f)
        if self.file_list:
            self.listbox.selection_set(0)
            self.on_select_file()

    def on_select_file(self, event=None):
        selection = self.listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        self.current_file = self.file_list[idx]
        info = self.files_info[self.current_file]
        self.old_text.config(state=tk.NORMAL)
        self.old_text.delete("1.0", tk.END)
        self.old_text.insert(tk.END, info["old"])
        self.old_text.config(state=tk.DISABLED)

        self.new_text.config(state=tk.NORMAL)
        self.new_text.delete("1.0", tk.END)
        self.new_text.insert(tk.END, info["new"])
        self.new_text.config(state=tk.NORMAL)

    def accept_current(self):
        if not self.current_file:
            return
        final_new = self.new_text.get("1.0", tk.END).rstrip('\n')
        self.files_info[self.current_file]["new"] = final_new
        self.files_info[self.current_file]["accepted"] = True
        self._mark_item(self.current_file, "✔")
        self._next_file()

    def reject_current(self):
        if not self.current_file:
            return
        self.files_info[self.current_file]["accepted"] = False
        self._mark_item(self.current_file, "✘")
        self._next_file()

    def _mark_item(self, filename, symbol):
        for i, f in enumerate(self.file_list):
            if f == filename:
                self.listbox.delete(i)
                self.listbox.insert(i, f"{symbol} {f}")
                self.listbox.selection_clear(0, tk.END)
                if i + 1 < len(self.file_list):
                    self.listbox.selection_set(i + 1)
                break

    def _next_file(self):
        for i, f in enumerate(self.file_list):
            if not self.files_info[f].get("marked", False):
                self.listbox.selection_clear(0, tk.END)
                self.listbox.selection_set(i)
                self.listbox.see(i)
                self.files_info[f]["marked"] = True
                self.on_select_file()
                return
        messagebox.showinfo("提示", "所有文件已处理完毕，可关闭窗口或使用“接受所有/拒绝所有”")

    def accept_all(self):
        for f in self.file_list:
            if f == self.current_file:
                final_new = self.new_text.get("1.0", tk.END).rstrip('\n')
                self.files_info[f]["new"] = final_new
            self.files_info[f]["accepted"] = True
        self._finish(True)

    def reject_all(self):
        for f in self.file_list:
            self.files_info[f]["accepted"] = False
        self._finish(False)

    def on_close(self):
        if not self.files_info.get("_finished", False):
            self._finish(False)

    def _finish(self, finish_flag):
        self.files_info["_finished"] = True
        self.top.destroy()
        self.wait_var.set(1)

    def get_results(self):
        return {f: info["accepted"] for f, info in self.files_info.items()
                if not f.startswith("_")}


# ---------- Flask 路由 ----------
@app.route("/code", methods=["GET"])
def get_code():
    """返回所选文件或项目内指定文件的代码，或整个项目所有 .py 文件"""
    global PROJECT_ROOT, DEFAULT_FILE

    if not PROJECT_ROOT:
        return jsonify({"error": "no_project_root_selected"}), 400

    # 1. 如果请求中指定了具体文件，返回该文件内容
    file_param = request.args.get("file", "").strip()
    if file_param:
        try:
            abs_path = safe_join(PROJECT_ROOT, file_param)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        if not os.path.exists(abs_path):
            return jsonify({"error": f"file not found: {file_param}"}), 404
        with open(abs_path, "r", encoding="utf-8") as f:
            return jsonify({"file": file_param, "code": f.read()})

    # 2. 如果有默认文件（启动时选择了单个文件），返回该文件内容
    if DEFAULT_FILE:
        abs_path = os.path.join(PROJECT_ROOT, DEFAULT_FILE)
        if os.path.exists(abs_path):
            with open(abs_path, "r", encoding="utf-8") as f:
                return jsonify({"file": DEFAULT_FILE, "code": f.read()})
        else:
            return jsonify({"error": "selected file missing"}), 404

    # 3. 否则返回整个项目的所有 .py 文件内容
    all_files = collect_project_files(PROJECT_ROOT)
    if not all_files:
        return jsonify({"error": "no .py files found in project root"}), 404

    return jsonify({"files": all_files})


@app.route("/update", methods=["POST"])
def handle_update():
    global PROJECT_ROOT, GUI_DATA, root

    if not PROJECT_ROOT:
        return jsonify({"error": "no_project_root_selected"}), 400

    data = request.get_json(silent=True)
    if not data or "code" not in data:
        return jsonify({"error": "missing code field"}), 400

    code_value = data["code"]
    if isinstance(code_value, dict):
        edits = code_value.get("edits", [])
    elif isinstance(code_value, str):
        try:
            edits_obj = json.loads(code_value)
            edits = edits_obj.get("edits", [])
        except json.JSONDecodeError:
            return jsonify({"error": "invalid JSON string in code field"}), 400
    else:
        return jsonify({"error": "unsupported code type"}), 400

    if not edits:
        return jsonify({"error": "no edits found"}), 400

    try:
        files_info = apply_edits(edits)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if not files_info:
        return jsonify({"error": "no valid edits after processing"}), 400

    event = threading.Event()
    GUI_DATA["files"] = files_info
    GUI_DATA["result_event"] = event
    GUI_DATA["final_result"] = None

    root.event_generate("<<ShowDiffEvent>>", when="tail")

    if not event.wait(timeout=300):
        return jsonify({"error": "timeout waiting for user decision"}), 408

    final = GUI_DATA.get("final_result")
    if final is None:
        return jsonify({"error": "unknown result from GUI"}), 500

    accepted_files = []
    rejected_files = []

    for rel_path, accepted in final.items():
        if accepted:
            try:
                abs_path = safe_join(PROJECT_ROOT, rel_path)
                new_content = files_info[rel_path]["new"]
                with open(abs_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                accepted_files.append(rel_path)
            except Exception as e:
                return jsonify({"error": f"写入 {rel_path} 失败: {str(e)}"}), 500
        else:
            rejected_files.append(rel_path)

    return jsonify({
        "status": "success",
        "accepted": accepted_files,
        "rejected": rejected_files
    })


# ---------- Tkinter 事件处理 ----------
def handle_gui_event(event):
    global GUI_DATA
    win = MultiDiffWindow(root, GUI_DATA["files"])
    results = win.get_results()
    GUI_DATA["final_result"] = results
    if GUI_DATA["result_event"]:
        GUI_DATA["result_event"].set()


# ---------- 启动选择（支持单个文件或项目目录） ----------
def choose_target():
    choice = messagebox.askyesno(
        "选择目标",
        "是否选择单个文件？\n（选择“是”将指定单个 .py 文件；选择“否”将选择项目根目录）"
    )
    if choice:
        path = filedialog.askopenfilename(
            title="选择要监控的 Python 文件",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")]
        )
        if not path:
            return None, None
        folder = os.path.dirname(os.path.abspath(path))
        filename = os.path.basename(path)
        return folder, filename
    else:
        folder = filedialog.askdirectory(title="请选择项目根目录")
        if not folder:
            return None, None
        return folder, None


# ---------- 主程序 ----------
def main():
    global root, PROJECT_ROOT, DEFAULT_FILE

    root = tk.Tk()
    root.withdraw()

    PROJECT_ROOT, DEFAULT_FILE = choose_target()
    if not PROJECT_ROOT:
        print("未选择目标，程序退出。")
        return

    if DEFAULT_FILE:
        print(f"[系统] 已选择单个文件: {os.path.join(PROJECT_ROOT, DEFAULT_FILE)}")
    else:
        print(f"[系统] 已选择项目根目录: {PROJECT_ROOT}")

    root.bind("<<ShowDiffEvent>>", handle_gui_event)

    print("[系统] 服务启动在 http://127.0.0.1:5500")
    print("[系统] 读取接口：http://127.0.0.1:5500/code")
    print("        - 单文件模式：直接返回所选文件内容")
    print("        - 项目模式：不加参数返回所有 .py 文件，或 ?file=main.py 指定文件")
    print("[系统] 写入接口：http://127.0.0.1:5500/update")
    print("[系统] 等待 Dify 发送请求...")

    threading.Thread(
        target=app.run,
        kwargs={"host": "0.0.0.0", "port": 5500, "threaded": True, "use_reloader": False, "debug": False},
        daemon=True
    ).start()

    root.mainloop()


if __name__ == "__main__":
    main()