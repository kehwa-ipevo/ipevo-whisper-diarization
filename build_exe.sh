#!/bin/bash

# 設置環境變數
PYTHON_PATH=$(which python)
ENV_PATH==$(dirname $(dirname "$python_path"))

PYTHON_VERSION=$(python --version)
PYTHON_VERSION=$(echo $PYTHON_VERSION | cut -d " " -f 2 | cut -d "." -f 1,2)

PYTHON_LIB_PATH="$ENV_PATH/lib/$PYTHON_VERSION/site-packages"
PYTHON_LIB_DYNLOAD="$ENV_PATH/lib/$PYTHON_VERSION/lib-dynload"

# PyInstaller 命令
pyinstaller \
    --add-binary "$PYTHON_LIB_PATH/torchaudio/lib/libtorchaudio.so:torchaudio/lib" \
    --add-data "$PYTHON_LIB_PATH/pytorch_lightning/version.info:./pytorch_lightning" \
    --add-data "$PYTHON_LIB_PATH/lightning_fabric/version.info:./lightning_fabric" \
    --add-data "$PYTHON_LIB_PATH/ctc_forced_aligner/punctuations.lst:ctc_forced_aligner" \
    --add-data "$PYTHON_LIB_PATH/ctc_forced_aligner/uroman:uroman" \
    --add-data "$PYTHON_LIB_PATH/ctc_forced_aligner/uroman/bin/uroman.pl:uroman" \
    --add-data "$PYTHON_LIB_PATH/ctc_forced_aligner/uroman/lib/NLP/Chinese.pm:uroman/lib/NLP" \
    --add-binary "$PYTHON_LIB_DYNLOAD/_sqlite3.cpython-310-x86_64-linux-gnu.so:." \
    --add-data "$PYTHON_LIB_PATH/faster_whisper/assets/silero_encoder_v5.onnx:faster_whisper/assets" \
    --add-data "$PYTHON_LIB_PATH/faster_whisper/assets/silero_decoder_v5.onnx:faster_whisper/assets" \
    --collect-all torch \
    --collect-all torchvision \
    --collect-all rich \
    --collect-all inflect \
    --collect-all nemo \
    --collect-all pyannote.audio \
    --collect-all chardet \
    --collect-all requests \
    --collect-all huggingface_hub \
    --collect-all transformers \
    --collect-all charset_normalizer \
    --collect-all whisperx \
    --collect-all nltk \
    --hidden-import _multibytecodec \
    --hidden-import unicodedata \
    --hidden-import _scproxy \
    --hidden-import _decimal \
    --hidden-import _contextvars \
    --hidden-import sqlite3 \
    --hidden-import _sqlite3 \
    --additional-hooks-dir . \
    --onefile -y \
    --exclude-module onnxscript \
    diarize.py