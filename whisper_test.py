import os
import whisper
import argparse



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
        default="turbo",
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

    parser.add_argument(
        "--use-quick",
        action="store_true",
        help="use quick mode"
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    ## get args
    args = get_args()
    model = whisper.load_model("turbo")
    if args.use_quick:
        result = model.transcribe(args.audio)
        print(result["text"])
        # print(result["segments"])
        # print(result["language"])
    else:
        audio = whisper.load_audio(args.audio)
        # audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio,  n_mels=128).to(model.device)

        # detect the spoken language
        _, probs = model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")

        # decode the audio
        options = whisper.DecodingOptions()
        result = whisper.decode(model, mel, options)

        # print the recognized text
        print(result.text)
