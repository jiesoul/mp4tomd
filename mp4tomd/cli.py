"""命令行入口。"""
import argparse
import os
import sys

from . import core


def _print_progress(fraction, message=""):
    pct = int(fraction * 100)
    bar = "#" * (pct // 2)
    sys.stdout.write(f"\r[{bar:<50}] {pct:3d}% {message}")
    sys.stdout.flush()
    if fraction >= 1.0:
        sys.stdout.write("\n")


def _print_batch_progress(current, total, filepath, frac, message):
    name = os.path.basename(filepath)
    pct = int(frac * 100)
    bar = "#" * (pct // 2)
    sys.stdout.write(
        f"\r[文件 {current}/{total}] {name:<30.30} [{bar:<50}] {pct:3d}% {message}"
    )
    sys.stdout.flush()
    if frac >= 1.0:
        sys.stdout.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="mp4tomd",
        description="将音视频文件转录为文字并导出 Markdown 文档（本地运行，基于 faster-whisper）。"
                    "支持多个文件 / 目录 / 通配符批量处理。",
    )
    parser.add_argument("input", nargs="+",
                        help="音视频文件、目录或通配符（可多个，如 *.mp3 文件夹/）")
    parser.add_argument("-o", "--output", help="单文件输出路径；多文件/合并时会被忽略")
    parser.add_argument("-d", "--output-dir", default=None,
                        help="批量输出目录（默认与每个输入文件同目录）")
    parser.add_argument("-m", "--model", default="small",
                        choices=["tiny", "base", "small", "medium", "large-v3"],
                        help="模型大小，越大越准但越慢（默认 small）")
    parser.add_argument("-l", "--language", default=None,
                        help="语言代码，如 zh、en；留空自动检测")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"],
                        help="推理设备，默认自动（有 GPU 用 cuda）")
    parser.add_argument("--no-timeline", action="store_true", help="不生成时间轴")
    parser.add_argument("--no-fulltext", action="store_true", help="不生成全文")
    parser.add_argument("--no-txt", action="store_true", help="不生成最初的纯文本稿 .txt")
    parser.add_argument("--no-srt", action="store_true", help="不生成最初的 SRT 字幕 .srt")
    # 关键画面
    parser.add_argument("--frames", action="store_true",
                        help="提取关键画面（仅视频；默认按章节起点，无章节则按固定间隔）")
    parser.add_argument("--frame-interval", type=float, default=60.0,
                        help="无章节时关键画面的截取间隔（秒，默认 60）")
    parser.add_argument("--keep-audio", action="store_true", help="保留提取的临时音频")
    # 说话人分离
    parser.add_argument("--diarize", action="store_true", help="启用说话人分离（需 pyannote + HF token）")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token（也可设环境变量 HF_TOKEN）")
    parser.add_argument("--num-speakers", type=int, default=None, help="已知说话人数量（可选，帮助提升分离效果）")
    # 自动分章节
    parser.add_argument("--chapters", action="store_true", help="按静音间隔自动分章节并生成标题")
    parser.add_argument("--chapter-gap", type=float, default=3.0, help="章节切分间隔阈值（秒，默认 3.0）")
    # 批量
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="递归处理目录（含子目录）")
    parser.add_argument("-c", "--combined", action="store_true",
                        help="额外生成一个合并的汇总 Markdown（所有文件合入一篇）")
    parser.add_argument("--combined-output", default=None,
                        help="合并汇总文件的输出路径（配合 -c 使用）")
    args = parser.parse_args(argv)

    common = dict(
        model_size=args.model,
        language=args.language,
        device=args.device,
        include_timeline=not args.no_timeline,
        include_fulltext=not args.no_fulltext,
        keep_audio=args.keep_audio,
        include_txt=not args.no_txt,
        include_srt=not args.no_srt,
        extract_frames=args.frames,
        frame_interval=args.frame_interval,
        diarize=args.diarize,
        hf_token=args.hf_token,
        num_speakers=args.num_speakers,
        chapters=args.chapters,
        chapter_gap=args.chapter_gap,
    )

    # 预先展开输入，判断是单文件还是批处理
    try:
        resolved = core.resolve_inputs(args.input, recursive=args.recursive)
    except Exception as e:
        print(f"错误：{e}", file=sys.stderr)
        return 1

    if not resolved:
        print("错误：没有匹配到任何受支持的文件。", file=sys.stderr)
        return 1

    if len(resolved) == 1 and not args.combined:
        # 单文件走原路径，支持 -o 指定输出
        try:
            out = core.process_file(
                resolved[0], output_path=args.output,
                progress_callback=_print_progress, **common,
            )
        except Exception as e:
            print(f"\n错误：{e}", file=sys.stderr)
            return 1
        print("\n[OK] 已生成：")
        for k in ("md", "txt", "srt"):
            if out.get(k):
                print(f"  - {out[k]}")
        return 0

    # 批处理（含 --combined 单文件汇总场景）
    try:
        result = core.process_batch(
            args.input,
            output_dir=args.output_dir,
            output_path=args.combined_output or args.output,
            combined=args.combined,
            recursive=args.recursive,
            progress_callback=_print_batch_progress,
            **common,
        )
    except Exception as e:
        print(f"\n错误：{e}", file=sys.stderr)
        return 1

    ok = len(result["outputs"])
    err = len(result["errors"])
    print(f"\n[完成] 成功 {ok} 个，失败 {err} 个，共 {len(result['files'])} 个文件。")
    for o in result["outputs"]:
        for k in ("md", "txt", "srt"):
            if o.get(k):
                print(f"  - {o[k]}")
    if result["combined"]:
        print(f"  [合并] {result['combined']}")
    for fp, msg in result["errors"]:
        print(f"  [失败] {fp}: {msg}", file=sys.stderr)
    return 1 if err else 0


if __name__ == "__main__":
    raise SystemExit(main())
