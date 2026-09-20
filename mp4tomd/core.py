"""音视频转文字并导出 Markdown 的本地程序。

核心依赖：
- ffmpeg：从视频中提取音频（处理 .mp4 等视频文件时需要）
- faster-whisper：本地语音识别模型（支持 CPU / GPU，离线运行）
- pyannote.audio（可选）：说话人分离，需要 HuggingFace token
"""
from __future__ import annotations

import datetime
import glob
import os
import shutil
import subprocess

# 支持的文件类型
SUPPORTED_VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".flv", ".wmv", ".webm", ".m4v"}
SUPPORTED_AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"}


def is_supported(filepath: str) -> bool:
    """判断文件是否为受支持的类型。"""
    ext = os.path.splitext(filepath)[1].lower()
    return ext in SUPPORTED_VIDEO_EXT or ext in SUPPORTED_AUDIO_EXT


def resolve_inputs(inputs, recursive=False):
    """把「文件 / 目录 / 通配符」混合列表展开为去重、排序后的受支持文件清单。

    不支持的文件会被跳过（不抛异常）。
    """
    resolved = []
    for item in inputs:
        for path in glob.glob(item) or [item]:
            if os.path.isdir(path):
                if recursive:
                    for root, _dirs, files in os.walk(path):
                        for name in files:
                            fp = os.path.join(root, name)
                            if is_supported(fp):
                                resolved.append(fp)
                else:
                    for name in sorted(os.listdir(path)):
                        fp = os.path.join(path, name)
                        if os.path.isfile(fp) and is_supported(fp):
                            resolved.append(fp)
            elif os.path.isfile(path):
                if is_supported(path):
                    resolved.append(path)
    # 去重并保持相对顺序，最后排序便于稳定输出
    seen, ordered = set(), []
    for fp in resolved:
        af = os.path.abspath(fp)
        if af not in seen:
            seen.add(af)
            ordered.append(fp)
    return sorted(ordered)


def _shift_headings(markdown: str, levels: int) -> str:
    """把 Markdown 的 ATX 标题整体下移 levels 级（用于合并文档时避免标题层级冲突）。"""
    out = []
    for line in markdown.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#") and not stripped.startswith("######"):
            prefix = stripped[:len(stripped) - len(stripped.lstrip("#"))]
            if len(prefix) < 6:
                line = "#" * levels + line
        out.append(line)
    return "\n".join(out)

# 说话人标签字母表（A, B, C ...）
_SPEAKER_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def check_ffmpeg() -> bool:
    """检测系统是否安装了 ffmpeg。"""
    return shutil.which("ffmpeg") is not None


def extract_audio(video_path: str, output_wav: str | None = None, sample_rate: int = 16000) -> str:
    """使用 ffmpeg 从视频中提取单声道 wav 音频。"""
    if output_wav is None:
        output_wav = os.path.splitext(video_path)[0] + "_audio.wav"
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "wav", output_wav,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg 提取音频失败：{e.stderr}") from e
    return output_wav


def extract_key_frames(video_path: str, timestamps, out_dir: str) -> list:
    """在指定时间戳处用 ffmpeg 截取关键画面，返回 [(timestamp, 图片路径), ...]。

    仅对视频文件有效；任一帧截取失败会被跳过。
    """
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for i, ts in enumerate(timestamps, start=1):
        out_path = os.path.join(out_dir, f"frame_{i:03d}.png")
        cmd = [
            "ffmpeg", "-y", "-ss", f"{float(ts):.3f}", "-i", video_path,
            "-frames:v", "1", "-q:v", "2", out_path,
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            results.append((float(ts), out_path))
        except subprocess.CalledProcessError:
            continue
    return results


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 语音识别
# ---------------------------------------------------------------------------
def transcribe(audio_path, model_size="small", language=None,
               device=None, compute_type="auto", progress_callback=None):
    """调用 faster-whisper 将音频转录为文字。

    返回 (segments, info)，segments 为包含 start/end/text 的字典列表。
    """
    from faster_whisper import WhisperModel

    if device is None:
        device = "cuda" if _cuda_available() else "cpu"

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments_gen, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    segments = []
    duration = getattr(info, "duration", None) or 0.0
    for seg in segments_gen:
        segments.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip(),
            "speaker": None,
        })
        if progress_callback is not None and duration:
            progress_callback(min(seg.end / duration, 1.0))

    return segments, info


# ---------------------------------------------------------------------------
# 说话人分离（可选，依赖 pyannote.audio）
# ---------------------------------------------------------------------------
def diarize(audio_path, hf_token=None, num_speakers=None, device=None):
    """对音频做说话人分离，返回 [{start, end, speaker}] 列表。"""
    try:
        from pyannote.audio import Pipeline
    except ImportError as e:
        raise RuntimeError(
            "说话人分离需要安装 pyannote.audio，请运行：pip install -r requirements-extra.txt"
        ) from e

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "说话人分离需要 HuggingFace token。请在 https://huggingface.co 注册并在 "
            "https://hf.co/pyannote/speaker-diarization-3.1 与 "
            "https://hf.co/pyannote/segmentation-3.0 同意许可后，设置 HF_TOKEN 环境变量或传入 --hf-token。"
        )

    if device is None:
        device = "cuda" if _cuda_available() else "cpu"

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token,
    )
    if device == "cuda":
        try:
            import torch
            pipeline.to(torch.device("cuda"))
        except Exception:
            pass

    annotation = pipeline(audio_path, num_speakers=num_speakers)
    results = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        results.append({
            "start": float(segment.start),
            "end": float(segment.end),
            "speaker": speaker,
        })
    return results


def assign_speakers(segments, diarization):
    """将说话人标签按时间重叠指派到每个识别片段。"""
    out = []
    for seg in segments:
        best, best_overlap = None, 0.0
        for d in diarization:
            overlap = min(seg["end"], d["end"]) - max(seg["start"], d["start"])
            if overlap > best_overlap:
                best_overlap, best = overlap, d["speaker"]
        s = dict(seg)
        s["speaker"] = best if best is not None else "未知"
        out.append(s)
    return out


def _build_speaker_map(segments):
    """把 SPEAKER_00 等映射为「发言人 A / B / C …」。"""
    speakers = sorted({
        s["speaker"] for s in segments
        if s.get("speaker") and s["speaker"] != "未知"
    })
    return {
        sp: f"发言人 {_SPEAKER_ALPHABET[i]}"
        for i, sp in enumerate(speakers)
    }


def _speaker_prefix(seg, sp_map):
    sp = seg.get("speaker")
    if not sp or sp == "未知":
        return ""
    label = sp_map.get(sp, sp)
    return f"[{label}] "


# ---------------------------------------------------------------------------
# 自动分章节
# ---------------------------------------------------------------------------
def split_chapters(segments, gap_threshold=3.0):
    """根据片段间的静音间隔把内容切成多个章节。"""
    chapters, current = [], None
    for seg in segments:
        if current is None:
            current = {"index": len(chapters) + 1, "start": seg["start"], "segments": [seg]}
            continue
        prev = current["segments"][-1]
        if seg["start"] - prev["end"] >= gap_threshold:
            chapters.append(current)
            current = {"index": len(chapters) + 1, "start": seg["start"], "segments": [seg]}
        else:
            current["segments"].append(seg)
    if current is not None:
        chapters.append(current)
    return chapters


def _chapter_heading(chapter, max_len=20):
    """用章节首句生成标题（截断到首个标点或指定长度）。"""
    text = chapter["segments"][0]["text"].strip()
    for p in ["。", "，", ".", ",", "?", "？", "!", "！", "；", ";"]:
        idx = text.find(p)
        if 0 < idx <= max_len:
            text = text[:idx]
            break
    if len(text) > max_len:
        text = text[:max_len] + "…"
    return text or "未命名"


# ---------------------------------------------------------------------------
# Markdown 生成
# ---------------------------------------------------------------------------
def format_timestamp(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def format_srt_timestamp(seconds: float) -> str:
    """SRT 格式时间戳：HH:MM:SS,mmm（毫秒用逗号）。"""
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_txt(segments) -> str:
    """生成最初的纯文本稿（不含时间戳/说话人，逐段一行）。"""
    return "\n".join(seg["text"] for seg in segments if seg["text"].strip()) + "\n"


def build_srt(segments, speakers=False) -> str:
    """生成 SRT 字幕（带起止时间戳，可选说话人前缀）。"""
    sp_map = _build_speaker_map(segments) if speakers else {}
    blocks = []
    for i, seg in enumerate(segments, start=1):
        start = format_srt_timestamp(seg["start"])
        end = format_srt_timestamp(seg["end"])
        prefix = _speaker_prefix(seg, sp_map) if speakers else ""
        text = (prefix + seg["text"]).strip()
        blocks.append(f"{i}\n{start} --> {end}\n{text}")
    return "\n\n".join(blocks) + "\n"


def build_markdown(segments, source_name, info,
                   include_timeline=True, include_fulltext=True,
                   speakers=False, chapters=False, chapter_gap=3.0,
                   frames=None) -> str:
    sp_map = _build_speaker_map(segments) if speakers else {}

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    language = getattr(info, "language", "自动检测")
    duration = getattr(info, "duration", None)
    duration_str = format_timestamp(duration) if duration else "未知"
    speaker_note = f"\n> 说话人：已分离（{len(sp_map)} 人）" if (speakers and sp_map) else ""

    lines = [f"# {source_name} · 转录文稿", ""]
    lines.append(f"> 生成时间：{now}  ")
    lines.append(f"> 识别语言：{language}  ")
    lines.append(f"> 音频时长：{duration_str}  {speaker_note}")
    lines.append("")

    if include_timeline:
        lines.append("## 时间轴")
        lines.append("")
        if not segments:
            lines.append("_（未识别到语音内容）_")
            lines.append("")
        for seg in segments:
            ts = format_timestamp(seg["start"])
            prefix = _speaker_prefix(seg, sp_map) if speakers else ""
            lines.append(f"- **[{ts}]** {prefix}{seg['text']}")
        lines.append("")

    if chapters:
        # 章节 -> 关键画面 映射（按章节起点时间戳）
        frame_by_ts = {ts: path for ts, path in (frames or [])}
        lines.append("## 章节")
        lines.append("")
        for ch in split_chapters(segments, chapter_gap):
            heading = _chapter_heading(ch)
            ts = format_timestamp(ch["start"])
            lines.append(f"### 第 {ch['index']} 节 · {heading}  `[{ts}]`")
            lines.append("")
            fpath = frame_by_ts.get(ch["start"])
            if fpath:
                lines.append(f"![关键画面 @{ts}]({fpath})")
                lines.append("")
            for seg in ch["segments"]:
                prefix = _speaker_prefix(seg, sp_map) if speakers else ""
                lines.append(f"{prefix}{seg['text']}")
            lines.append("")

    if frames and not chapters:
        lines.append("## 关键画面")
        lines.append("")
        for ts, path in frames:
            lines.append(f"![关键画面 @{format_timestamp(ts)}]({path})")
            lines.append("")

    if include_fulltext:
        lines.append("## 全文")
        lines.append("")
        full = " ".join(seg["text"] for seg in segments).strip()
        lines.append(full if full else "_（未识别到语音内容）_")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 端到端处理
# ---------------------------------------------------------------------------
def process_file(input_path, output_path=None, model_size="small",
                 language=None, device=None, include_timeline=True,
                 include_fulltext=True, keep_audio=False,
                 diarize=False, hf_token=None, num_speakers=None,
                 chapters=False, chapter_gap=3.0,
                 include_txt=True, include_srt=True,
                 extract_frames=False, frame_interval=60.0,
                 progress_callback=None) -> dict:
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"找不到输入文件：{input_path}")

    ext = os.path.splitext(input_path)[1].lower()
    if ext not in SUPPORTED_VIDEO_EXT and ext not in SUPPORTED_AUDIO_EXT:
        raise ValueError(f"不支持的文件类型：{ext}")

    def _transcribe_progress(frac):
        if progress_callback:
            progress_callback(0.1 + frac * 0.6, "正在识别文字…")

    tmp_audio = None
    try:
        if ext in SUPPORTED_VIDEO_EXT:
            if not check_ffmpeg():
                raise RuntimeError(
                    "未检测到 ffmpeg。请先安装 ffmpeg 并加入 PATH，"
                    "Windows 可访问 https://www.gyan.dev/ffmpeg/builds/ 下载。"
                )
            if progress_callback:
                progress_callback(0.02, "正在提取音频…")
            audio_path = extract_audio(input_path)
            tmp_audio = audio_path
        else:
            audio_path = input_path

        segments, info = transcribe(
            audio_path, model_size=model_size, language=language,
            device=device, progress_callback=_transcribe_progress,
        )

        if diarize:
            if progress_callback:
                progress_callback(0.72, "正在分离说话人…")
            diarization = diarize(audio_path, hf_token=hf_token,
                                  num_speakers=num_speakers, device=device)
            segments = assign_speakers(segments, diarization)
            if progress_callback:
                progress_callback(0.92, "说话人分离完成")

        source_name = os.path.splitext(os.path.basename(input_path))[0]

        if output_path is None:
            output_path = os.path.splitext(input_path)[0] + ".md"
        md_dir = os.path.dirname(output_path)
        base = os.path.splitext(output_path)[0]

        # 关键画面提取（仅视频文件有效）
        frames = []
        if extract_frames and ext in SUPPORTED_VIDEO_EXT:
            if not check_ffmpeg():
                raise RuntimeError(
                    "未检测到 ffmpeg，关键画面提取需要 ffmpeg。"
                    "请安装 ffmpeg 并加入 PATH（Windows：https://www.gyan.dev/ffmpeg/builds/）。"
                )
            if chapters:
                chs = split_chapters(segments, chapter_gap)
                timestamps = [ch["start"] for ch in chs]
            else:
                dur = getattr(info, "duration", None) or 0.0
                step = max(int(frame_interval), 1)
                timestamps = list(range(0, int(dur), step))
                if not timestamps:
                    timestamps = [0.0]
            frame_dir = os.path.join(md_dir, source_name + "_frames")
            extracted = extract_key_frames(input_path, timestamps, frame_dir)
            frames = [
                (ts, os.path.relpath(p, md_dir).replace(os.sep, "/"))
                for ts, p in extracted
            ]

        markdown = build_markdown(
            segments, source_name, info,
            include_timeline=include_timeline,
            include_fulltext=include_fulltext,
            speakers=diarize,
            chapters=chapters,
            chapter_gap=chapter_gap,
            frames=frames,
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        out_files = {"md": output_path}

        # 最初的纯文本稿
        if include_txt:
            txt_path = base + ".txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(build_txt(segments))
            out_files["txt"] = txt_path

        # 最初的 SRT 字幕（带时间戳）
        if include_srt:
            srt_path = base + ".srt"
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(build_srt(segments, speakers=diarize))
            out_files["srt"] = srt_path

        if progress_callback:
            progress_callback(1.0, "完成")
        return out_files
    finally:
        if tmp_audio and not keep_audio and os.path.exists(tmp_audio):
            try:
                os.remove(tmp_audio)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# 批量处理
# ---------------------------------------------------------------------------
def process_batch(inputs, output_dir=None, output_path=None, combined=False,
                  model_size="small", language=None, device=None,
                  include_timeline=True, include_fulltext=True,
                  keep_audio=False, diarize=False, hf_token=None,
                  num_speakers=None, chapters=False, chapter_gap=3.0,
                  include_txt=True, include_srt=True,
                  extract_frames=False, frame_interval=60.0,
                  recursive=False, progress_callback=None) -> dict:
    """批量处理多个文件 / 目录 / 通配符。

    progress_callback 签名为 (current, total, filepath, frac, message)。

    返回字典：
        {
            "files": [...输入清单...],
            "outputs": [...每个文件生成的 md 路径...],
            "combined": 合并文档路径或 None,
            "errors": [(filepath, error_str), ...],
        }
    """
    files = resolve_inputs(inputs, recursive=recursive)
    if not files:
        raise FileNotFoundError("没有匹配到任何受支持的文件。")

    total = len(files)
    outputs, errors, bodies = [], [], []

    for idx, fp in enumerate(files, start=1):
        if progress_callback:
            progress_callback(idx, total, fp, 0.0, "排队中…")
        try:
            single_output = output_path if (total == 1 and output_path) else None
            if single_output is None:
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    base = os.path.splitext(os.path.basename(fp))[0] + ".md"
                    single_output = os.path.join(output_dir, base)
                else:
                    single_output = None  # process_file 默认与输入同名

            def _cb(frac, msg):
                if progress_callback:
                    progress_callback(idx, total, fp, frac, msg)

            out = process_file(
                fp, output_path=single_output,
                model_size=model_size, language=language, device=device,
                include_timeline=include_timeline,
                include_fulltext=include_fulltext,
                keep_audio=keep_audio,
                diarize=diarize, hf_token=hf_token,
                num_speakers=num_speakers, chapters=chapters,
                chapter_gap=chapter_gap,
                include_txt=include_txt, include_srt=include_srt,
                extract_frames=extract_frames, frame_interval=frame_interval,
                progress_callback=_cb,
            )
            outputs.append(out)
            if combined:
                with open(out["md"], "r", encoding="utf-8") as f:
                    bodies.append((os.path.splitext(os.path.basename(fp))[0], f.read()))
        except Exception as e:  # 单个文件失败不影响其余
            errors.append((fp, str(e)))
            if progress_callback:
                progress_callback(idx, total, fp, 1.0, f"失败：{e}")

    combined_path = None
    if combined and bodies:
        combined_path = output_path if (output_path and total > 1) else None
        if combined_path is None:
            if output_dir:
                combined_path = os.path.join(output_dir, "批量转录汇总.md")
            else:
                combined_path = os.path.join(
                    os.path.dirname(files[0]) or ".", "批量转录汇总.md"
                )
        header = [
            f"# 批量转录汇总（共 {len(bodies)} 个文件）",
            "",
            f"> 生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"> 模型：{model_size}  ",
            f"> 识别语言：{language or '自动检测'}",
            "",
        ]
        parts = [("\n".join(header))]
        for name, body in bodies:
            # 把每篇文档的标题整体下移一级，避免与汇总标题冲突
            shifted = _shift_headings(body, 1)
            # 把首行 "# xxx · 转录文稿" 替换为 "## xxx"
            parts.append(shifted)
        with open(combined_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(parts) + "\n")

    return {
        "files": files,
        "outputs": outputs,
        "combined": combined_path,
        "errors": errors,
    }
