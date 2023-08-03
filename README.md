<h1 align="center">Speaker Diarization Using OpenAI Whisper</h1>
<p align="center">
  <a href="https://github.com/MahmoudAshraf97/whisper-diarization/stargazers">
    <img src="https://img.shields.io/github/stars/MahmoudAshraf97/whisper-diarization.svg?colorA=orange&colorB=orange&logo=github"
         alt="GitHub stars">
  </a>
  <a href="https://github.com/MahmoudAshraf97/whisper-diarization/issues">
        <img src="https://img.shields.io/github/issues/MahmoudAshraf97/whisper-diarization.svg"
             alt="GitHub issues">
  </a>
  <a href="https://github.com/MahmoudAshraf97/whisper-diarization/blob/master/LICENSE">
        <img src="https://img.shields.io/github/license/MahmoudAshraf97/whisper-diarization.svg"
             alt="GitHub license">
  </a>
  <a href="https://twitter.com/intent/tweet?text=&url=https%3A%2F%2Fgithub.com%2FMahmoudAshraf97%2Fwhisper-diarization">
  <img src="https://img.shields.io/twitter/url/https/github.com/MahmoudAshraf97/whisper-diarization.svg?style=social" alt="Twitter">
  </a> 
  </a>
  <a href="https://colab.research.google.com/github/MahmoudAshraf97/whisper-diarization/blob/main/Whisper_Transcription_%2B_NeMo_Diarization.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
  </a>
 
</p>


# 
Speaker Diarization pipeline based on OpenAI Whisper
I'd like to thank [@m-bain](https://github.com/m-bain) for Wav2Vec2 forced alignment, [@mu4farooqi](https://github.com/mu4farooqi) for punctuation realignment algorithm

<img src="https://github.blog/wp-content/uploads/2020/09/github-stars-logo_Color.png" alt="drawing" width="25"/> **Please, star the project on github (see top-right corner) if you appreciate my contribution to the community!**

## What is it
This repository combines Whisper ASR capabilities with Voice Activity Detection (VAD) and Speaker Embedding to identify the speaker for each sentence in the transcription generated by Whisper. First, the vocals are extracted from the audio to increase the speaker embedding accuracy, then the transcription is generated using Whisper, then the timestamps are corrected and aligned using WhisperX to help minimize diarization error due to time shift. The audio is then passed into MarbleNet for VAD and segmentation to exclude silences, TitaNet is then used to extract speaker embeddings to identify the speaker for each segment, the result is then associated with the timestamps generated by WhisperX to detect the speaker for each word based on timestamps and then realigned using punctuation models to compensate for minor time shifts.


Whisper, WhisperX and NeMo parameters are coded into diarize.py and helpers.py, I will add the CLI arguments to change them later
## Installation
`FFMPEG` and `Cython` are needed as prerquisites to install the requirements
```
pip install cython
```
```
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```
```
pip install -r requirements.txt
```
## Usage 

```
python diarize.py -a AUDIO_FILE_NAME
```

If your system has enough VRAM (>=10GB), you can use `diarize_parallel.py` instead, the difference is that it runs NeMo in parallel with Whisper, this can be benifecial in some cases and the result is the same since the two models are not dependant on each other. This is still experimental, so expect errors and sharp edges. Your feedback is welcome.

## Command Line Options

- `-a AUDIO_FILE_NAME`: The name of the audio file to be processed
- `--no-stem`: Disables source separation
- `--whisper-model`: The model to be used for ASR, default is `medium.en`

## Known Limitations
- Only tested on english but several other languages are supported
- Overlapping speakers are yet to be addressed, a possible approach would be to separate the audio file and isolate only one speaker, then feed it into the pipeline but this will need much more computation
- There might be some errors, please raise an issue if you encounter any.

## Future Improvements
- Implement a maximum length per sentence for SRT
- Improve Batch Processing

## Aknowledgements
Special Thanks for [@adamjonas](https://github.com/adamjonas) for supporting this project
This work is based on [OpenAI's Whisper](https://github.com/openai/whisper) , [Faster Whisper](https://github.com/guillaumekln/faster-whisper) , [Nvidia NeMo](https://github.com/NVIDIA/NeMo) , and [Facebook's Demucs](https://github.com/facebookresearch/demucs)
