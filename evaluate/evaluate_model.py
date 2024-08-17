import pandas as pd
import torch
from tqdm.auto import tqdm

pd.set_option("display.max_colwidth", None)
tqdm.pandas()

import string

import huggingface_hub
from datasets import Audio, load_dataset
from jiwer import process_words, wer_default
from pythainlp.tokenize import word_tokenize as tokenize
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import whisper
from faster_whisper import WhisperModel, available_models

import argparse

import evaluate
import time
import os

import gc


sampling_rate = 16_000
device = "cuda"
metric = evaluate.load("wer")

#region Text Compute
def clean_text(sentence: str, remove_punctuation: bool = True):
    sentence = sentence.strip()
    # Remove zero-width and non-breaking space.
    sentence = sentence.replace("\u200b", " ")
    sentence = sentence.replace("\xa0", " ")
    # remove redundant punctuations
    sentence = sentence.replace("เเ", "แ")

    # วรรณยุกต์/สระ
    sentence = sentence.replace("ํา", "ำ")
    sentence = sentence.replace("่ำ", "่ำ")
    sentence = sentence.replace("ำ้", "้ำ")
    sentence = sentence.replace("ํ่า", "่ำ")

    # replace special underscore and dash.
    sentence = sentence.replace("▁", "_")
    sentence = sentence.replace("—", "-")
    sentence = sentence.replace("–", "-")
    sentence = sentence.replace("−", "-")

    # replace special characters.
    sentence = sentence.replace("’", "'")
    sentence = sentence.replace("‘", "'")
    sentence = sentence.replace("”", '"')
    sentence = sentence.replace("“", '"')

    if remove_punctuation:
        sentence = "".join(
            [character for character in sentence if character not in string.punctuation]
        )
    return " ".join(sentence.split()).strip()
def compute_metrics_thai_text(
    pred_texts: list[str], ref_texts: list[str]
) -> dict[str, float | int]:
    """
    Compute the WER, IER, SER, and DER between the predicted and reference texts.
    Parameters
    ==========
    pred_texts: list[str]
        The list of predicted texts.
    ref_texts: list[str]
        The list of reference or ground truth texts.

    Returns
    =======
    dict
        A dictionary containing the WER, IER, SER, and DER.
    """
    # normalize everything and re-compute the WER
    norm_pred_texts = [clean_text(pred).lower() for pred in pred_texts]
    norm_ref_texts = [clean_text(label).lower() for label in ref_texts]

    # Since the `process_words` tokenizes the words based on space, we need to tokenize the Thai text and join them with space
    norm_pred_texts = [
        " ".join(tokenize(pred, engine="deepcut")) for pred in norm_pred_texts
    ]
    norm_ref_texts = [
        " ".join(tokenize(label, engine="deepcut")) for label in norm_ref_texts
    ]

    # filtering step to only evaluate the samples that correspond to non-zero normalized references:
    norm_pred_texts = [
        norm_pred_texts[i]
        for i in range(len(norm_pred_texts))
        if len(norm_ref_texts[i]) > 0
    ]
    norm_ref_texts = [
        norm_ref_texts[i]
        for i in range(len(norm_ref_texts))
        if len(norm_ref_texts[i]) > 0
    ]

    wer_output = process_words(
        norm_ref_texts, norm_pred_texts, wer_default, wer_default
    )
    wer_norm = 100 * wer_output.wer
    ier_norm = (
        100 * wer_output.insertions / sum([len(ref) for ref in wer_output.references])
    )
    ser_norm = (
        100
        * wer_output.substitutions
        / sum([len(ref) for ref in wer_output.references])
    )
    der_norm = (
        100 * wer_output.deletions / sum([len(ref) for ref in wer_output.references])
    )

    return {"wer": wer_norm, "ier": ier_norm, "ser": ser_norm, "der": der_norm}
#endregion

#region Transcribe
@torch.no_grad()
def transcribe(
    audio_dataset,
    batch_size: int = 8,
    device: str = "cuda",
    transcribe_fn=None
):
    if model_type == 'huggingface':
        model.eval()
    all_predictions = []
    for i in tqdm(range(0, len(audio_dataset), batch_size)):
        torch.cuda.empty_cache()
        gc.collect()
        audio_batch = audio_dataset[i : i + batch_size]
        predictions = transcribe_fn(audio_batch)
        all_predictions += predictions
    return all_predictions
def huggingface_transcribe(audio_batch):
    input_speech_array_list = [
            audio_dict["array"] for audio_dict in audio_batch["audio"]
    ]
    inputs = processor(
        input_speech_array_list,
        sampling_rate=16_000,
        return_tensors="pt",
        # padding=True,
    )
    predicted_ids = model.generate(
        inputs["input_features"].to(device).half(),
        language="th",
        return_timestamps=False,
    )
    predictions = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return predictions
def whisper_transcribe(audio_batch):
    input_speech_array_list = [
            whisper.pad_or_trim(audio_dict["array"]) for audio_dict in audio_batch["audio"]
    ]
    input_speech_tensor = torch.tensor(input_speech_array_list)
    inputs = whisper.log_mel_spectrogram(input_speech_tensor, n_mels=128)
    options = whisper.DecodingOptions(language='th', without_timestamps=True)
    results = whisper.decode(model, inputs, options)
    predictions = [result.text for result in results]
    return predictions
def faster_whisper_transcribe(audio_batch):
    predictions = []
    for audio_dict in audio_batch["audio"]:
        audio = whisper.pad_or_trim(audio_dict["array"])
        segments, info = model.transcribe(audio, beam_size=5, language='th')
        predictions.append(' '.join(segment.text for segment in segments))

    return predictions
#endregion

#region Argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Whisper model to use")
    parser.add_argument("--faster", default=False, help="Use faster whisper model")
    parser.add_argument("--compute_type", default='float16', choices=["float16", "float32", "int8"])

    args = parser.parse_args()
    return args
#endregion

#region Main
if __name__ == "__main__":

    # Model init
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transcribe_fn = None

    if "biodatlab" in args.model:
        huggingface_hub.login(token='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')
        model_type = 'huggingface'
        compute_type = {"float16":torch.float16, "float32":torch.float32, "int8":torch.int8}
        processor = WhisperProcessor.from_pretrained(
            args.model, language="thai", task="transcribe", fast_tokenizer=True
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=compute_type[args.compute_type], use_flash_attention_2=True, device_map=device
        )
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="th", task="transcribe"
        )
        transcribe_fn = huggingface_transcribe
    elif not args.faster:
        model_type = 'whisper'
        model = whisper.load_model(args.model)
        transcribe_fn = whisper_transcribe
    elif args.faster:
        model_type = 'faster_whisper'
        model = WhisperModel(args.model, device=device, compute_type=args.compute_type)
        transcribe_fn = faster_whisper_transcribe

    # Datset init
    test_dataset = load_dataset("mozilla-foundation/common_voice_13_0", "th", split="test[:10%]", trust_remote_code=True, token=True)
    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))
    test_dataset = test_dataset.rename_column("sentence", "text")

    # Transcribe
    start_time = time.time()

    transcriptions = transcribe(
        audio_dataset=test_dataset,
        batch_size=16,
        device=device,
        transcribe_fn=transcribe_fn,
    )
    
    duration = time.time() - start_time

    # Evaluate
    audio_transcript_df = pd.DataFrame(
        {"text": test_dataset["text"], "prediction": transcriptions}
    )
    filename = f"whisper-{args.model.split('/')[-1]}-{'faster' if args.faster else 'non_faster'}-{args.compute_type}-cmv13-test"
    audio_transcript_df.to_csv(filename + ".csv")

    results = compute_metrics_thai_text(
        [p.lower() for p in audio_transcript_df["prediction"]],
        [t.lower() for t in test_dataset["text"]],
    )

    # Save result
    print('filename:', filename)
    print('result:', results)
    if not os.path.exists("test_result.csv"):
        df = pd.DataFrame.from_dict({'filename':[filename], 'wer':[results['wer']], 'ier':[results['ier']], 'ser':[results['ser']], 'der':[results['der']], 'duration':[duration]})
    else:
        data = {'filename':[filename], 'wer':[results['wer']], 'ier':[results['ier']], 'ser':[results['ser']], 'der':[results['der']], 'duration':[duration]}
        data = [filename, results['wer'], results['ier'], results['ser'], results['der'], duration]
        df = pd.read_csv("test_result.csv")
        df.loc[len(df.index)] = data
    df.to_csv("test_result.csv", index=False)

#endregion