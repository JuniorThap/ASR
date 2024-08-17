#! python3.7

import argparse
import os
import numpy as np
import speech_recognition as sr
# import whisper
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import huggingface_hub
import torch

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform

import string
import gc


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
#endregion


def main():
    #region parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model biodatlab to use")
    parser.add_argument("--energy_threshold", default=750,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=0.5,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=0.5,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()
    #endregion

    #region setup
    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    # Important for linux users.
    # Prevents permanent application hang and crash by using the wrong Microphone
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    huggingface_hub.login(token='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')
    processor = WhisperProcessor.from_pretrained(
        args.model, language="thai", task="transcribe", fast_tokenizer=True
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float16, use_flash_attention_2=True, device_map=device
    )
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="th", task="transcribe"
    )

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    transcription = ['']

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
    print("Model loaded.\n")
    #endregion

    #region loop
    while True:
        try:
            torch.cuda.empty_cache()
            gc.collect()
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now
                
                # Combine audio data from queue
                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()
                
                # Convert in-ram buffer to something the model can use directly without needing a temp file.
                # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
                # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0


                # Read the transcription.
                input = processor(
                    [audio_np],
                    sampling_rate=16_000,
                    return_tensors="pt",
                )
                predicted_ids = model.generate(
                    input["input_features"].to(device).half(),
                    language="th",
                    return_timestamps=False,
                )
                predictions = processor.batch_decode(predicted_ids, skip_special_tokens=True)
                text = ' '.join(segment.strip() for segment in predictions).strip()
                text = clean_text(text)

                # segments, info = model.transcribe(audio_np, beam_size=5, language='th')

                # If we detected a pause between recordings, add a new item to our transcription.
                # Otherwise edit the existing one.
                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text

                # Clear the console to reprint the updated transcription.
                os.system('cls' if os.name=='nt' else 'clear')
                for line in transcription:
                    print(line)
                # Flush stdout.
                print('', end='', flush=True)
            else:
                # Infinite loops are bad for processors, must sleep.
                sleep(0.05)
        except KeyboardInterrupt:
            break
    #endregion

    print("\n\nTranscription:")
    for line in transcription:
        print(line)

if __name__ == "__main__":
    main()