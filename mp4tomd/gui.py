"""图形界面（Tkinter）。"""
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox

from . import core

MODEL_CHOICES = ["tiny", "base", "small", "medium", "large-v3"]
LANG_CHOICES = ["自动检测", "zh", "en", "ja", "ko", "fr", "de", "es", "ru"]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("音视频转文字 → Markdown")
        self.geometry("660x680")
        self.resizable(True, True)
        self._build()
        if not core.check_ffmpeg():
            self._log("⚠ 未检测到 ffmpeg，视频文件将无法处理。请安装 ffmpeg 并加入 PATH。\n")

    def _build(self):
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)

        def row(label, widget, r, c=1, span=1):
            if label:
                ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w", pady=3)
            widget.grid(row=r, column=c, columnspan=span, sticky="ew", padx=6, pady=3)

        ttk.Label(frm, text="音视频文件").grid(row=0, column=0, sticky="w", pady=3)
        self.input_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.input_var).grid(row=0, column=1, sticky="ew", padx=6, pady=3)
        ttk.Button(frm, text="浏览…", command=self._pick_input).grid(row=0, column=2)

        ttk.Label(frm, text="输出 MD").grid(row=1, column=0, sticky="w", pady=3)
        self.output_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.output_var).grid(row=1, column=1, sticky="ew", padx=6, pady=3)
        ttk.Button(frm, text="浏览…", command=self._pick_output).grid(row=1, column=2)

        ttk.Label(frm, text="模型").grid(row=2, column=0, sticky="w", pady=3)
        self.model_var = tk.StringVar(value="small")
        ttk.OptionMenu(frm, self.model_var, "small", *MODEL_CHOICES).grid(row=2, column=1, sticky="ew", padx=6, pady=3)

        ttk.Label(frm, text="语言").grid(row=3, column=0, sticky="w", pady=3)
        self.lang_var = tk.StringVar(value="自动检测")
        ttk.OptionMenu(frm, self.lang_var, "自动检测", *LANG_CHOICES).grid(row=3, column=1, sticky="ew", padx=6, pady=3)

        ttk.Label(frm, text="设备").grid(row=4, column=0, sticky="w", pady=3)
        self.device_var = tk.StringVar(value="自动")
        ttk.OptionMenu(frm, self.device_var, "自动", "自动", "cpu", "cuda").grid(row=4, column=1, sticky="ew", padx=6, pady=3)

        # 选项区
        self.timeline_var = tk.BooleanVar(value=True)
        self.fulltext_var = tk.BooleanVar(value=True)
        self.diarize_var = tk.BooleanVar(value=False)
        self.chapters_var = tk.BooleanVar(value=False)
        self.keep_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text="生成时间轴", variable=self.timeline_var).grid(row=5, column=1, sticky="w", padx=6)
        ttk.Checkbutton(frm, text="生成全文", variable=self.fulltext_var).grid(row=6, column=1, sticky="w", padx=6)
        ttk.Checkbutton(frm, text="说话人分离（需 HF token）", variable=self.diarize_var).grid(row=7, column=1, sticky="w", padx=6)
        ttk.Checkbutton(frm, text="自动分章节标题", variable=self.chapters_var).grid(row=8, column=1, sticky="w", padx=6)
        ttk.Checkbutton(frm, text="保留临时音频", variable=self.keep_var).grid(row=9, column=1, sticky="w", padx=6)

        ttk.Label(frm, text="HF Token").grid(row=10, column=0, sticky="w", pady=3)
        self.hf_var = tk.StringVar(value=os.environ.get("HF_TOKEN", ""))
        ttk.Entry(frm, textvariable=self.hf_var, show="*").grid(row=10, column=1, sticky="ew", padx=6, pady=3)

        ttk.Label(frm, text="说话人数").grid(row=11, column=0, sticky="w", pady=3)
        self.numspk_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.numspk_var, width=8).grid(row=11, column=1, sticky="w", padx=6, pady=3)

        ttk.Label(frm, text="章节间隔(秒)").grid(row=12, column=0, sticky="w", pady=3)
        self.gap_var = tk.StringVar(value="3.0")
        ttk.Entry(frm, textvariable=self.gap_var, width=8).grid(row=12, column=1, sticky="w", padx=6, pady=3)

        self.progress = ttk.Progressbar(frm, mode="determinate", maximum=100)
        self.progress.grid(row=13, column=0, columnspan=3, sticky="ew", pady=10)

        self.run_btn = ttk.Button(frm, text="开始转换", command=self._run)
        self.run_btn.grid(row=14, column=1, sticky="ew", padx=6, pady=4)

        self.log = scrolledtext.ScrolledText(frm, height=14, state="disabled")
        self.log.grid(row=15, column=0, columnspan=3, sticky="nsew", pady=6)
        frm.rowconfigure(15, weight=1)
        frm.columnconfigure(1, weight=1)

    def _log(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _set_progress(self, fraction, message=""):
        def update():
            self.progress["value"] = int(fraction * 100)
            if message:
                self._log(message + "\n")
        self.after(0, update)

    def _pick_input(self):
        path = filedialog.askopenfilename(
            title="选择音视频文件",
            filetypes=[("音视频", "*.mp4 *.mkv *.mov *.avi *.flv *.wmv *.webm *.m4v *.mp3 *.wav *.m4a *.aac *.flac *.ogg *.opus"), ("全部", "*.*")],
        )
        if path:
            self.input_var.set(path)
            if not self.output_var.get():
                self.output_var.set(os.path.splitext(path)[0] + ".md")

    def _pick_output(self):
        path = filedialog.asksaveasfilename(
            title="保存 Markdown", defaultextension=".md",
            filetypes=[("Markdown", "*.md")],
        )
        if path:
            self.output_var.set(path)

    def _run(self):
        in_path = self.input_var.get().strip()
        if not in_path:
            messagebox.showerror("提示", "请先选择音视频文件")
            return
        out_path = self.output_var.get().strip() or (os.path.splitext(in_path)[0] + ".md")
        language = None if self.lang_var.get() == "自动检测" else self.lang_var.get()
        device = None if self.device_var.get() == "自动" else self.device_var.get()

        hf_token = self.hf_var.get().strip() or None
        num_speakers = None
        if self.numspk_var.get().strip():
            try:
                num_speakers = int(self.numspk_var.get().strip())
            except ValueError:
                messagebox.showerror("提示", "说话人数必须是整数")
                return
        try:
            chapter_gap = float(self.gap_var.get().strip() or "3.0")
        except ValueError:
            messagebox.showerror("提示", "章节间隔必须是数字")
            return

        self.run_btn["state"] = "disabled"
        self._log(f"开始处理：{in_path}\n")

        def worker():
            try:
                result = core.process_file(
                    in_path, output_path=out_path,
                    model_size=self.model_var.get(),
                    language=language, device=device,
                    include_timeline=self.timeline_var.get(),
                    include_fulltext=self.fulltext_var.get(),
                    keep_audio=self.keep_var.get(),
                    diarize=self.diarize_var.get(),
                    hf_token=hf_token,
                    num_speakers=num_speakers,
                    chapters=self.chapters_var.get(),
                    chapter_gap=chapter_gap,
                    progress_callback=self._set_progress,
                )
                self._log(f"✅ 已生成：{result}\n")
            except Exception as e:
                self._log(f"❌ 错误：{e}\n")
            finally:
                self.run_btn["state"] = "normal"
                self._set_progress(0)

        threading.Thread(target=worker, daemon=True).start()


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
