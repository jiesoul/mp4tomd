# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 打包配置（生成免 Python 运行的 .exe）
# 用法：pyinstaller build.spec
# 默认打包图形界面（windowed）。如需命令行版，把 ProcessGui 改为 ProcessCli 即可。

block_cipher = None

a = Analysis(
    ['mp4tomd/gui.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'faster_whisper',
        'ctranslate2',
        'onnxruntime',
        'av',
        'soundfile',
        'huggingface_hub',
        'pyannote.audio',
        'pyannote.core',
        'pyannote.pipeline',
        'pyannote.metrics',
        'torch',
        'torchaudio',
        'numpy',
        'scipy',
        'sklearn',
        'yaml',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tk.test', 'unittest', 'test'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='mp4tomd',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,           # 图形界面：False（无黑窗口）；若要命令行版设为 True 并改用 cli.py
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
