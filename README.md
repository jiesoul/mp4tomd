# mp4tomd · 音视频转文字 → Markdown（本地运行）

一个完全在本地运行的程序：把 **视频 / 音频** 文件转录成文字，并导出为结构化的 **Markdown** 文档。识别使用 [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper)，**离线、无需联网、无需 API Key**。

## 功能

- 支持视频（mp4/mkv/mov/avi/flv/wmv/webm/m4v）与音频（mp3/wav/m4a/aac/flac/ogg/opus）
- 本地语音识别，自动检测语言（也可指定 `zh` / `en` 等）
- 输出 Markdown：带时间戳的 **时间轴** + 合并的 **全文**
- 🆕 **说话人分离**：区分不同发言人（基于 `pyannote.audio`，可选）
- 🆕 **自动分章节**：按静音间隔自动切分并生成章节标题
- 🆕 **可打包为 .exe**：免 Python 环境运行
- 提供 **图形界面（GUI）** 与 **命令行（CLI）** 两种用法
- 支持 CPU 与 NVIDIA GPU（CUDA）自动切换

## 安装

### 1. 基础依赖（语音识别 + 界面）

需要 Python 3.8+：

```powershell
cd d:\projects\mp4tomd
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> 首次运行会自动从 HuggingFace 下载识别模型（约几十 MB ~ 几 GB，取决于模型大小）。

### 2. 安装 ffmpeg（仅处理视频文件需要）

- 访问 https://www.gyan.dev/ffmpeg/builds/ 下载 `ffmpeg-release-essentials.zip`
- 解压后将 `bin` 目录（含 `ffmpeg.exe`）加入系统 **PATH**
- 验证：`ffmpeg -version`
- （仅处理 `.wav` / `.mp3` 等纯音频文件时可以不装 ffmpeg。）

### 3. 说话人分离（可选）

```powershell
pip install -r requirements-extra.txt
```

然后到以下页面 **同意模型许可**（需用 HuggingFace 账号）：

- https://hf.co/pyannote/speaker-diarization-3.1
- https://hf.co/pyannote/segmentation-3.0

并准备 token：https://hf.co/settings/tokens ，运行时通过 `--hf-token` / 界面填写，或设置环境变量：

```powershell
$env:HF_TOKEN = "hf_xxxxxxxxxxxx"
```

## 使用方法

### 图形界面（推荐）

```powershell
python -m mp4tomd
```

打开窗口后：选择音视频文件 → 设置输出、模型、语言 → 勾选「说话人分离」「自动分章节」→ 点击「开始转换」。

### 命令行

```powershell
# 基本用法（自动检测语言，默认 small 模型）
python -m mp4tomd.cli 会议录音.mp4 -o 会议录音.md

# 指定中文、较大模型
python -m mp4tomd.cli 视频.mp4 -l zh -m medium

# 开启说话人分离 + 自动分章节
python -m mp4tomd.cli 会议.mp4 --diarize --hf-token hf_xxx --num-speakers 3 --chapters --chapter-gap 4

# 仅生成全文、不生成时间轴
python -m mp4tomd.cli 视频.mp4 --no-timeline
```

参数说明：

| 参数 | 说明 |
| --- | --- |
| `input` | 音视频文件路径（必填） |
| `-o, --output` | 输出 Markdown 路径，默认与输入同名 `.md` |
| `-m, --model` | 模型大小：`tiny`/`base`/`small`/`medium`/`large-v3`（默认 `small`） |
| `-l, --language` | 语言代码，如 `zh` `en`，留空自动检测 |
| `--device` | `cpu` 或 `cuda`，默认自动 |
| `--diarize` | 启用说话人分离（需安装 `requirements-extra.txt` 并提供 HF token） |
| `--hf-token` | HuggingFace token（也可设环境变量 `HF_TOKEN`） |
| `--num-speakers` | 已知说话人数量（可选，提升分离效果） |
| `--chapters` | 按静音间隔自动分章节并生成标题 |
| `--chapter-gap` | 章节切分间隔阈值（秒，默认 3.0） |
| `--no-timeline` | 不生成时间轴 |
| `--no-fulltext` | 不生成全文 |
| `--keep-audio` | 保留提取的临时音频 |

## 输出的 Markdown 示例

```markdown
# 会议录音 · 转录文稿

> 生成时间：2026-09-20 14:30:00
> 识别语言：zh
> 音频时长：12:30
> 说话人：已分离（3 人）

## 时间轴

- **[00:00]** [发言人 A] 各位好，今天我们讨论一下项目进度。
- **[00:08]** [发言人 B] 首先由开发组汇报本周完成情况。

## 章节

### 第 1 节 · 各位好今天我们讨论  `[00:00]`

[发言人 A] 各位好，今天我们讨论一下项目进度。
[发言人 B] 首先由开发组汇报本周完成情况。

## 全文

各位好，今天我们讨论一下项目进度。首先由开发组汇报本周完成情况。
```

## 模型选择建议

| 模型 | 体积 | 速度 | 精度 | 适用 |
| --- | --- | --- | --- | --- |
| tiny / base | 小 | 最快 | 一般 | 快速预览 |
| small | 中 | 快 | 较好 | 日常（默认） |
| medium | 大 | 慢 | 好 | 高质量 |
| large-v3 | 很大 | 最慢 | 最好 | 追求最高精度 |

## 打包为 .exe（免 Python 运行）

1. 安装打包工具：`pip install pyinstaller`（或使用 `pip install -e ".[build]"`）
2. 执行打包：

```powershell
pyinstaller build.spec
```

打包完成后，可执行文件在 `dist\mp4tomd\mp4tomd.exe`（图形界面版，无黑窗口）。

- 如需 **命令行版** `.exe`：编辑 `build.spec`，将首行 `['mp4tomd/gui.py']` 改为 `['mp4tomd/cli.py']`，并把 `console=False` 改为 `console=True`。
- 注意：打包体积较大（含 PyTorch / faster-whisper 运行时，数 GB），首次打包耗时较长。
- 打包后的程序仍需本机安装 **ffmpeg**（视频文件）并联网下载一次模型权重（或预先缓存到用户目录）。

## 常见问题

- **没有 ffmpeg**：处理视频时报错，按上文安装并加入 PATH 即可。
- **首次很慢**：需要下载模型权重，之后会缓存复用。
- **识别不准**：换用更大的模型（如 `medium` / `large-v3`），或显式指定 `-l zh`。
- **说话人分离报错**：确认已安装 `requirements-extra.txt`、已同意 pyannote 模型许可、并已提供有效的 `HF_TOKEN`。
- **GPU 未使用**：确保已安装 CUDA 版 PyTorch；程序会自动检测并启用 `cuda`。
