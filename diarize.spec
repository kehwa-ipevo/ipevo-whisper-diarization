# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('=./lib/3.10/site-packages/pytorch_lightning/version.info', './pytorch_lightning'), ('=./lib/3.10/site-packages/lightning_fabric/version.info', './lightning_fabric'), ('=./lib/3.10/site-packages/ctc_forced_aligner/punctuations.lst', 'ctc_forced_aligner'), ('=./lib/3.10/site-packages/ctc_forced_aligner/uroman', 'uroman'), ('=./lib/3.10/site-packages/ctc_forced_aligner/uroman/bin/uroman.pl', 'uroman'), ('=./lib/3.10/site-packages/ctc_forced_aligner/uroman/lib/NLP/Chinese.pm', 'uroman/lib/NLP'), ('=./lib/3.10/site-packages/faster_whisper/assets/silero_encoder_v5.onnx', 'faster_whisper/assets'), ('=./lib/3.10/site-packages/faster_whisper/assets/silero_decoder_v5.onnx', 'faster_whisper/assets')]
binaries = [('=./lib/3.10/site-packages/torchaudio/lib/libtorchaudio.so', 'torchaudio/lib'), ('=./lib/3.10/lib-dynload/_sqlite3.cpython-310-x86_64-linux-gnu.so', '.')]
hiddenimports = ['_multibytecodec', 'unicodedata', '_scproxy', '_decimal', '_contextvars', 'sqlite3', '_sqlite3']
tmp_ret = collect_all('torch')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('torchvision')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('rich')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('inflect')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('nemo')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pyannote.audio')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('chardet')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('requests')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('huggingface_hub')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('transformers')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('charset_normalizer')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('whisperx')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('nltk')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['diarize.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['.'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['onnxscript'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='diarize',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
