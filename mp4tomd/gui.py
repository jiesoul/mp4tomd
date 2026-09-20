"""图形界面（Tkinter）。"""
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox

from . import core

MODEL_CHOICES = ["tiny", "base", "small", "medium", "large-v3"]
LANG_CHOICES = ["自动检测", "zh", "en", "ja", "ko", "fr", "de", "es", "ru"]
MEDIA_FILTER = (
    "音视频",
    "*.mp4 *.mkv *.mov *.avi *.flv *.wmv *.webm *.m4v *.mp3 *.wav *.m4a *.aac *.flac *.ogg *.opus",
)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("音视频转文字 → Markdown")
        self.geometry("680x720")
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

        ttk.Label(frm, text="音视频/目录").grid(row=0, column=0, sticky="w", pady=3)
        self.input_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.input_var).grid(row=0, column=1, sticky="ew", padx=6, pady=3)
        ttk.Button(frm, text="浏览文件", command=self._pick_input_file).grid(row=0, column=2, padx=2)
        ttk.Button(frm, text="浏览目录", command=self._pick_input_dir).grid(row=0, column=3, padx=2)

        ttk.Label(frm, text="输出 MD/目录").grid(row=1, column=0, sticky="w", pady=3)
        self.output_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.output_var).grid(row=1, column=1, sticky="ew", padx=6, pady=3)
        ttk.Button(frm, text="浏览输出", command=self._pick_output).grid(row=1, column=2, columnspan=2)

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
        self.batch_var = tk.BooleanVar(value=False)
        self.combined_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text="生成时间轴", variable=self.timeline_var).grid(row=5, column=1, sticky="w", padx=6)
        ttk.Checkbutton(frm, text="生成全文", variable=self.fulltext_var).grid(row=6, column=1, sticky="w", padx=6)
        ttk.Checkbutton(frm, text="说话人分离（需 HF token）", variable=self.diarize_var).grid(row=7, column=1, sticky="w", padx=6)
        ttk.Checkbutton(frm, text="自动分章节标题", variable=self.chapters_var).grid(row=8, column=1, sticky="w", padx=6)
        ttk.Checkbutton(frm, text="保留临时音频", variable=self.keep_var).grid(row=9, column=1, sticky="w", padx=6)
        ttk.Checkbutton(frm, text="批量处理（输入为目录/通配符，输出为文件夹）", variable=self.batch_var).grid(row=10, column=1, sticky="w", padx=6)
        ttk.Checkbutton(frm, text="合并为单个 MD（批量时可用）", variable=self.combined_var).grid(row=11, column=1, sticky="w", padx=6)

        ttk.Label(frm, text="HF Token").grid(row=12, column=0, sticky="w", pady=3)
        self.hf_var = tk.StringVar(value=os.environ.get("HF_TOKEN", ""))
        ttk.Entry(frm, textvariable=self.hf_var, show="*").grid(row=12, column=1, columnspan=2, sticky="ew", padx=6, pady=3)

        ttk.Label(frm, text="说话人数").grid(row=13, column=0, sticky="w", pady=3)
        self.numspk_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.numspk_var, width=8).grid(row=13, column=1, sticky="w", padx=6, pady=3)

        ttk.Label(frm, text="章节间隔(秒)").grid(row=14, column=0, sticky="w", pady=3)
        self.gap_var = tk.StringVar(value="3.0")
        ttk.Entry(frm, textvariable=self.gap_var, width=8).grid(row=14, column=1, sticky="w", padx=6, pady=3)

        self.progress = ttk.Progressbar(frm, mode="determinate", maximum=100)
        self.progress.grid(row=15, column=0, columnspan=4, sticky="ew", pady=10)

        self.run_btn = ttk.Button(frm, text="开始转换", command=self._run)
        self.run_btn.grid(row=16, column=1, columnspan=2, sticky="ew", padx=6, pady=4)

        self.log = scrolledtext.ScrolledText(frm, height=12, state="disabled")
        self.log.grid(row=17, column=0, columnspan=4, sticky="nsew", pady=6)
        frm.rowconfigure(17, weight=1)
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

    def _pick_input_file(self):
        path = filedialog.askopenfilename(title="选择音视频文件", filetypes=[MEDIA_FILTER, ("全部", "*.*")])
        if path:
            self.batch_var.set(False)
            self.input_var.set(path)
            if not self.output_var.get():
                self.output_var.set(os.path.splitext(path)[0] + ".md")

    def _pick_input_dir(self):
        path = filedialog.askdirectory(title="选择包含音视频的文件夹")
        if path:
            self.batch_var.set(True)
            self.input_var.set(path)
            if not self.output_var.get():
                self.output_var.set(path)

    def _pick_output(self):
        if self.batch_var.get():
            path = filedialog.askdirectory(title="选择输出文件夹")
        else:
            path = filedialog.asksaveasfilename(title="保存 Markdown", defaultextension=".md",
                                                filetypes=[("Markdown", "*.md")])
        if path:
            self.output_var.set(path)

    def _common_params(self, language, device, hf_token, num_speakers, chapter_gap):
        return dict(
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
        )

    def _run(self):
        in_path = self.input_var.get().strip()
        if not in_path:
            messagebox.showerror("提示", "请先选择音视频文件或目录")
            return
        out_path = self.output_var.get().strip()
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
        self._log(f"开始处理：{in_path}（批量={self.batch_var.get()}）\n")
        common = self._common_params(language, device, hf_token, num_speakers, chapter_gap)

        if self.batch_var.get():
            def batch_cb(current, total, filepath, frac, message):
                overall = (current - 1 + frac) / max(total, 1)
                self._set_progress(overall, f"[{current}/{total}] {os.path.basename(filepath)} {message}")
            worker = lambda: self._batch_worker(in_path, out_path, common, batch_cb)
        else:
            single_out = out_path or (os.path.splitext(in_path)[0] + ".md")
            worker = lambda: self._single_worker(in_path, single_out, common)

        threading.Thread(target=worker, daemon=True).start()

    def _single_worker(self, in_path, out_path, common):
        try:
            result = core.process_file(in_path, output_path=out_path,
                                       progress_callback=self._set_progress, **common)
            self._log("✅ 已生成：\n")
            for k in ("md", "txt", "srt"):
                if result.get(k):
                    self._log(f"   - {result[k]}\n")
        except Exception as e:
            self._log(f"❌ 错误：{e}\n")
        finally:
            self.run_btn["state"] = "normal"
            self._set_progress(0)

    def _batch_worker(self, in_path, out_path, common, batch_cb):
        try:
            result = core.process_batch(
                [in_path],
                output_dir=out_path if os.path.isdir(out_path) or not out_path else None,
                output_path=None,
                combined=self.combined_var.get(),
                recursive=False,
                progress_callback=batch_cb,
                **common,
            )
            self._log(f"✅ 成功 {len(result['outputs'])} 个，失败 {len(result['errors'])} 个。\n")
            for o in result["outputs"]:
                for k in ("md", "txt", "srt"):
                    if o.get(k):
                        self._log(f"   - {o[k]}\n")
            if result["combined"]:
                self._log(f"   [合并] {result['combined']}\n")
            for fp, msg in result["errors"]:
                self._log(f"   [失败] {fp}: {msg}\n")
        except Exception as e:
            self._log(f"❌ 错误：{e}\n")
        finally:
            self.run_btn["state"] = "normal"
            self._set_progress(0)


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
