"""命令行入口。"""
import argparse
import sys

from . import core


def _print_progress(fraction, message=""):
    pct = int(fraction * 100)
    bar = "#" * (pct // 2)
    sys.stdout.write(f"\r[{bar:<50}] {pct:3d}% {message}")
    sys.stdout.flush()
    if fraction >= 1.0:
        sys.stdout.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="mp4tomd",
        description="将音视频文件转录为文字并导出 Markdown 文档（本地运行，基于 faster-whisper）。",
    )
    parser.add_argument("input", help="音视频文件路径（mp4/mkv/mp3/wav 等）")
    parser.add_argument("-o", "--output", help="输出 Markdown 路径，默认与输入同名 .md")
    parser.add_argument("-m", "--model", default="small",
                        choices=["tiny", "base", "small", "medium", "large-v3"],
                        help="模型大小，越大越准但越慢（默认 small）")
    parser.add_argument("-l", "--language", default=None,
                        help="语言代码，如 zh、en；留空自动检测")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"],
                        help="推理设备，默认自动（有 GPU 用 cuda）")
    parser.add_argument("--no-timeline", action="store_true", help="不生成时间轴")
    parser.add_argument("--no-fulltext", action="store_true", help="不生成全文")
    parser.add_argument("--keep-audio", action="store_true", help="保留提取的临时音频")
    # 说话人分离
    parser.add_argument("--diarize", action="store_true", help="启用说话人分离（需 pyannote + HF token）")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token（也可设环境变量 HF_TOKEN）")
    parser.add_argument("--num-speakers", type=int, default=None, help="已知说话人数量（可选，帮助提升分离效果）")
    # 自动分章节
    parser.add_argument("--chapters", action="store_true", help="按静音间隔自动分章节并生成标题")
    parser.add_argument("--chapter-gap", type=float, default=3.0, help="章节切分间隔阈值（秒，默认 3.0）")
    args = parser.parse_args(argv)

    try:
        out = core.process_file(
            args.input,
            output_path=args.output,
            model_size=args.model,
            language=args.language,
            device=args.device,
            include_timeline=not args.no_timeline,
            include_fulltext=not args.no_fulltext,
            keep_audio=args.keep_audio,
            diarize=args.diarize,
            hf_token=args.hf_token,
            num_speakers=args.num_speakers,
            chapters=args.chapters,
            chapter_gap=args.chapter_gap,
            progress_callback=_print_progress,
        )
    except Exception as e:
        print(f"\n错误：{e}", file=sys.stderr)
        return 1

    print(f"\n✅ 已生成 Markdown：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
