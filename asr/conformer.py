import re

import nemo.collections.asr as nemo_asr
import torch
from inverse_text_normalization.run_predict import inverse_normalize_text
from pyctcdecode import build_ctcdecoder


def standardize_output(text, lang):
    text = text.lower()
    text = re.sub(r"[^A-Za-z0-9\u0900-\u097F\s]+", "", text)
    text = " ".join(text.split())
    try:
        itn_text = inverse_normalize_text([text], lang=lang)[0]
    except Exception:
        itn_text = text
    text = text.lower()
    itn_text = itn_text.lower()
    return text, itn_text


class ConformerRecognizer:
    def __init__(self, model_path, lang, use_hotwords=False):
        self.NEMO_PATH = model_path
        self.lang = lang
        self.asr_model = nemo_asr.models.ASRModel.restore_from(
            self.NEMO_PATH, map_location=torch.device("cuda")
        )
        self.use_hotwords = use_hotwords
        if self.use_hotwords:
            self.decoder = build_ctcdecoder(self.asr_model.decoder.vocabulary)

    def transcribe(self, files, inference_hotwords=[], hotword_weight=10.0):
        if self.use_hotwords:
            return self.transcribe_hotword(files, inference_hotwords, hotword_weight)
        else:
            return self.transcribe_greedy(files)

    def transcribe_greedy(self, files):
        transcript = self.asr_model.transcribe(paths2audio_files=files, batch_size=1)
        transcript = transcript[0]
        transcript, itn_transcript = standardize_output(transcript, self.lang)
        return transcript, itn_transcript

    def transcribe_hotword(self, files, inference_hotwords, hotword_weight=10.0):
        logits = self.asr_model.transcribe(
            paths2audio_files=files, batch_size=1, logprobs=True
        )[0]
        transcript = self.decoder.decode(
            logits, hotwords=inference_hotwords, hotword_weight=hotword_weight
        )
        transcript, itn_transcript = standardize_output(transcript, self.lang)
        return transcript, itn_transcript
