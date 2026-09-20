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

## 受限网络环境与离线测试

### 模型 / 依赖下载走国内镜像

如果在中国大陆等网络受限环境，`huggingface.co` 与 `github.com` 的部分资源可能被屏蔽，导致模型权重或 pip 包下载缓慢甚至失败。可用镜像解决：

```powershell
# 1) 模型权重走 HF 镜像
$env:HF_ENDPOINT = "https://hf-mirror.com"

# 2) pip 包走清华镜像（安装更快）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 设置 `HF_ENDPOINT` 后，faster-whisper 会从镜像拉取模型；运行本程序前保持该环境变量即可（也可写入系统环境变量永久生效）。

### 没有样例音频？用 Windows 自带 TTS 离线生成

仓库内置 `make_sample.py`：利用 Windows 系统语音合成（`pyttsx3`，无需联网、无需外部音频文件）生成一段测试语音 `sample.wav`，可直接喂给本程序验证「音频 → 文字 → MD」全流程：

```powershell
# 安装 TTS 依赖（仅测试时需要）
pip install pyttsx3

# 生成 sample.wav（约 4 秒英文语音，内容为预设测试句）
python make_sample.py

# 转录并生成带章节的 Markdown
$env:HF_ENDPOINT = "https://hf-mirror.com"   # 如网络受限请保留
python -m mp4tomd.cli sample.wav -m tiny --chapters -o sample.md
```

生成的 `sample.md` 即为一个真实转录示例（结构见上方「输出的 Markdown 示例」）。验证完成后可删除 `sample.wav` / `sample.md`（`sample.wav` 已在 `.gitignore` 中忽略，不会误提交）。

## 使用方法

### 图形界面（推荐）

```powershell
python -m mp4tomd
```

打开窗口后：点击「浏览文件」选择单个音视频，或点击「浏览目录」选择文件夹（自动勾选「批量处理」）→ 设置输出、模型、语言 → 勾选「说话人分离」「自动分章节」「合并为单个 MD」等 → 点击「开始转换」。批量模式下「输出」为文件夹，可为每个文件生成同名 `.md`，或合并为一篇汇总。

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

#### 批量处理（多个文件 / 目录 / 通配符）

`input` 可接收 **多个** 参数，并支持目录与通配符（`*`），自动跳过不支持的文件：

```powershell
# 处理一个目录下的全部音视频（同目录输出 .md）
python -m mp4tomd.cli 会议录音/ -o 忽略此值

# 处理匹配通配符的多个文件，统一输出到 out/ 目录
python -m mp4tomd.cli "*.mp3" -d out/

# 递归处理子目录
python -m mp4tomd.cli 素材/ -r -d out/

# 同时传入多种来源（文件 + 目录 + 通配符）
python -m mp4tomd.cli 会议.mp4 讲座/ "tmp/*.wav" -d out/

# 额外生成一个合并汇总文档（所有文件合入一篇 Markdown）
python -m mp4tomd.cli 会议录音/ -c -d out/
python -m mp4tomd.cli 会议录音/ -c --combined-output 汇总.md
```

> 当 `input` 多于一个，或显式传入目录 / 通配符时，会进入批量模式：每个文件生成一个同名 `.md`，失败的文件不影响其余；加上 `-c` 还会额外生成 `批量转录汇总.md`（每篇文档作为 `##` 小节归入一篇）。
> 单文件用法保持与以前一致，`-o` 指定输出路径仍然有效。

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
| `--no-txt` | 不生成最初的纯文本稿 `.txt` |
| `--no-srt` | 不生成最初的 SRT 字幕 `.srt` |
| `--frames` | 提取关键画面并嵌入 Markdown（仅视频） |
| `--frame-interval` | 无章节时关键画面截取间隔（秒，默认 60） |
| `-d, --output-dir` | 批量输出目录（默认与每个输入文件同目录） |
| `-r, --recursive` | 递归处理目录（含子目录） |
| `-c, --combined` | 额外生成合并汇总 Markdown（所有文件合入一篇） |
| `--combined-output` | 合并汇总文件的输出路径（配合 `-c`） |

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

## 输出文件说明（txt / srt / md）

每次转换默认会生成 **三个** 文件（同名、同目录，或统一输出到 `-d` 指定目录）：

| 文件 | 内容 | 用途 |
| --- | --- | --- |
| `文件名.txt` | 最初的**纯文本稿**（逐段一行，无时间戳、无说话人） | 原始识别结果，便于二次校对 |
| `文件名.srt` | 最初的 **SRT 字幕**（带起止时间戳；开启说话人分离时含 `[发言人 X]` 前缀） | 直接导入播放器 / 剪辑软件做字幕 |
| `文件名.md` | **最终校正式 Markdown**：带时间戳时间轴 + 自动分章节 + 全文（即上方示例） | 阅读、归档、二次编辑 |

> 工作流建议：`txt` / `srt` 是第一手原始稿，人工校对或润色后，整理进 `md` 作为最终成品。
> 若不需要原始稿，可用 `--no-txt` / `--no-srt` 关闭（GUI 暂默认全部生成）。

## 关键画面提取（仅视频）

对视频文件，可在关键时间点用 ffmpeg 截取画面，并以 `![关键画面 @时间](图片)` 的形式嵌入 Markdown：

- 开启 `--chapters` 时：在每个**章节起点**截图，直接嵌在对应章节标题下方；
- 未开启 `--chapters` 时：按固定间隔 `--frame-interval`（默认 60 秒）截图，单独生成「## 关键画面」小节；
- 截取的图片保存在 `<源文件名>_frames/` 子目录（PNG），Markdown 中以相对路径引用，**请连同该文件夹一起分发**才能保证图片显示。

> 纯音频文件无画面，开启此选项会被自动忽略。此功能依赖 ffmpeg。

CLI：

```powershell
# 按章节提取关键画面
python -m mp4tomd.cli 会议.mp4 --chapters --frames

# 无章节时每 30 秒截一帧
python -m mp4tomd.cli 讲座.mp4 --frames --frame-interval 30
```

GUI：勾选「提取关键画面（仅视频）」，并可在「画面间隔(秒)」设置间隔。

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
