import argparse
import logging
import os
import re
import uuid
import faster_whisper
import whisper
import torch
import torchaudio
import signal
from typing import List, Dict, Any
from utils import timeit
from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

from helpers import (
    cleanup,
    create_config,
    find_numeral_symbol_tokens,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    langs_to_iso,
    process_language_arg,
    punct_model_langs,
    whisper_langs,
    write_srt,
)

mtypes = {"cpu": "int8", "cuda": "float16"}
TEMP_PATH = ""


def split_english_character(text):
    merge_text = []
    temp = []
    previous = ""
    for i in list(text):
        try:
            if 65<=ord(i)<=122:
                if i.isupper() and previous.islower():
                    temp.append(" ")
                temp.append(i)
            else:
                if temp:
                    merge_text.append("".join(temp))
                    temp=[]
                merge_text.append(i)
        except TypeError:
            raise ValueError(f"failed to convert text {i} to ord")
        previous = i
    return merge_text

@timeit
def main(args: argparse.Namespace):
    """whisper diarization main process

    Args:
        args (argparse.Namespace): parsed arguments
    """    
    language = process_language_arg(args.language, args.model_name)

    if args.stemming:
        # Isolate vocals from the rest of the audio

        return_code = os.system(
            f'python3 -m demucs.separate -n htdemucs --two-stems=vocals "{args.audio}" -o temp_outputs'
        )

        if return_code != 0:
            logging.warning(
                "Source splitting failed, using original audio file. "
                "Use --no-stem argument to disable it."
            )
            vocal_target = args.audio
        else:
            vocal_target = os.path.join(
                "temp_outputs",
                "htdemucs",
                os.path.splitext(os.path.basename(args.audio))[0],
                "vocals.wav",
            )
    else:
        vocal_target = args.audio

    # Transcribe the audio file
    import librosa
    audio_waveform, _sr = librosa.load(vocal_target)
    new_sample_rate = 16_000
    audio_waveform = librosa.resample(audio_waveform, orig_sr=_sr, target_sr=new_sample_rate)
    print(f"type: {type(audio_waveform)} shape: {audio_waveform.shape}")
    model = whisper.load_model("turbo")
    result = model.transcribe(vocal_target)
    full_transcript = result["text"]

    del model
    torch.cuda.empty_cache()

    # Forced Alignment
    alignment_model, alignment_tokenizer = load_alignment_model(
        args.device,
        dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )

    emissions, stride = generate_emissions(
        alignment_model,
        torch.from_numpy(audio_waveform)
        .to(alignment_model.dtype)
        .to(alignment_model.device),
        batch_size=args.batch_size,
    )

    del alignment_model
    torch.cuda.empty_cache()

    tokens_starred, text_starred = preprocess_text(
        full_transcript,
        romanize=True,
        language = langs_to_iso[result["language"]],
    )

    segments, scores, blank_token = get_alignments(
        emissions,
        tokens_starred,
        alignment_tokenizer,
    )

    spans = get_spans(tokens_starred, segments, blank_token)

    word_timestamps = postprocess_results(text_starred, spans, stride, scores)


    # convert audio to mono for NeMo combatibility
    ROOT = os.getcwd()
    u = uuid.uuid1()
    temp_path = os.path.join(ROOT, f"temp_outputs_{u.__str__()}")
    global TEMP_PATH
    TEMP_PATH = temp_path
    os.makedirs(temp_path, exist_ok=True)
    torchaudio.save(
        os.path.join(temp_path, "mono_file.wav"),
        torch.from_numpy(audio_waveform).unsqueeze(0).float(),
        16000,
        channels_first=True,
    )


    # Initialize NeMo MSDD diarization model
    msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(args.device)
    msdd_model.diarize()

    del msdd_model
    torch.cuda.empty_cache()

    # Reading timestamps <> Speaker Labels mapping


    speaker_ts = []
    with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split(" ")
            s = int(float(line_list[5]) * 1000)
            e = s + int(float(line_list[8]) * 1000)
            speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

    if result["language"] in punct_model_langs:
        # restoring punctuation in the transcript to help realign the sentences
        punct_model = PunctuationModel(model="kredor/punctuate-all")

        words_list = list(map(lambda x: x["word"], wsm))

        labled_words = punct_model.predict(words_list, chunk_size=args.chunk_size)

        ending_puncts = ".?!"
        model_puncts = ".,;:!?"

        # We don't want to punctuate U.S.A. with a period. Right?
        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

        for word_dict, labeled_tuple in zip(wsm, labled_words):
            word = word_dict["word"]
            if (
                word
                and labeled_tuple[1] in ending_puncts
                and (word[-1] not in model_puncts or is_acronym(word))
            ):
                word += labeled_tuple[1]
                if word.endswith(".."):
                    word = word.rstrip(".")
                word_dict["word"] = word

    else:
        lang = result["language"]
        logging.warning(
            f"Punctuation restoration is not available for {lang} language."
            " Using the original punctuation."
        )

    wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)
    print(f"ssm : {ssm}")
    # if info.language in ["ja", "zh"]:
    # if max(probs, key=probs.get) in ["ja", "zh"]:
    if result["language"] in ["ja", "zh"]:
        ssm = merge_text(ssm)

    with open(f"{os.path.splitext(args.audio)[0]}.txt", "w", encoding="utf-8-sig") as f:
        get_speaker_aware_transcript(ssm, f)

    with open(f"{os.path.splitext(args.audio)[0]}.srt", "w", encoding="utf-8-sig") as srt:
        write_srt(ssm, srt)

    cleanup(temp_path)


def merge_text(ssm: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """merge jpn, chi text

    Args:
        ssm (List[Dict[str, Any]]): list of sentence speaker mapping

    Returns:
        List[Dict[str, Any]]: merged list
    """
    for ss in ssm:
        split_text = [x for x in ss["text"].split(" ") if len(x)]
        tmp = ""
        for s in split_text:
            if len(s) == 1:
                tmp += s
            else:
                tmp += " " + s + " "
        ss["text"] = tmp.strip().replace("  "," ")
    return ssm

def get_args() -> argparse.Namespace:
    """get arguments

    Returns:
        argparse.Namespace : parsed arguments
    """

    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", 
        "--audio", 
        help="name of the target audio file", 
        required=True
    )
    parser.add_argument(
        "--no-stem",
        action="store_false",
        dest="stemming",
        default=True,
        help="Disables source separation."
        "This helps with long files that don't contain a lot of music.",
    )

    parser.add_argument(
        "--suppress_numerals",
        action="store_true",
        dest="suppress_numerals",
        default=False,
        help="Suppresses Numerical Digits."
        "This helps the diarization accuracy but converts all digits into written text.",
    )

    parser.add_argument(
        "--whisper-model",
        dest="model_name",
        default="base",
        help="name of the Whisper model to use",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        dest="batch_size",
        default=8,
        help="Batch size for batched inference, reduce if you run out of memory, "
        "set to 0 for original whisper longform inference",
    )

    parser.add_argument(
        "--language",
        type=str,
        default=None,
        choices=whisper_langs,
        help="Language spoken in the audio, specify None to perform language detection",
    )

    parser.add_argument(
        "--device",
        dest="device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="if you have a GPU use 'cuda', otherwise 'cpu'",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="chunk size for punct_model",
    )

    parser.add_argument(
        "--check-update",
        action="store_true",
        help="Check for updates app update",
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="show current app version"
    )

    args = parser.parse_args()

    return args

# 定義處理函数
def signal_handler(sig, frame):

    print("catch Ctrl+C (SIGINT)")
    
    ## remove temp files
    temp_path = "./temp_outputs"
    if os.path.exists(temp_path):
        cleanup(temp_path)    
    
    if TEMP_PATH and os.path.exists(TEMP_PATH):
        cleanup(TEMP_PATH)
    exit(0)


if __name__ == "__main__":

    # 設置信號處理器
    signal.signal(signal.SIGINT, signal_handler)

    ## get args
    args = get_args()

    ## if version is on, show version and exit
    if args.version:
        from updater import get_current_version
        print(f"current version: {get_current_version()}")
        exit(0)

    ## if update is on, update app and exit
    if args.check_update:
        from updater import update
        update()
        exit(0)

    ## main process
    main(args)