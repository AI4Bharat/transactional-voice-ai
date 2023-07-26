import re

import nemo.collections.asr as nemo_asr
import torch
from inverse_text_normalization.run_predict import inverse_normalize_text
from pyctcdecode import build_ctcdecoder


def standardize_output(text, lang):
    text = text.lower()
    text = " ".join(text.split())
    itn_text = inverse_normalize_text([text], lang=lang)[0]
    text = text.lower()
    itn_text = itn_text.lower()
    return text, itn_text


class ConformerRecognizer:
    def __init__(
        self, model_path, lang, lm_path=None, alpha=1.0, beta=1.5, use_hotwords=False
    ):
        self.NEMO_PATH = model_path
        self.lang = lang
        self.asr_model = nemo_asr.models.ASRModel.restore_from(
            self.NEMO_PATH, map_location=torch.device("cuda")
        )
        self.use_hotwords = use_hotwords

        self.lm_path = lm_path
        if not self.lm_path:
            self.use_lm = False
        else:
            self.use_lm = True

        if self.use_lm:
            self.decoder = build_ctcdecoder(
                self.asr_model.decoder.vocabulary, self.lm_path, alpha=alpha, beta=beta
            )
        elif self.use_hotwords:
            self.decoder = build_ctcdecoder(self.asr_model.decoder.vocabulary)
        print(
            f"Initialized ASR for lang {lang} | Use LM {self.use_lm} Alpha {alpha} Beta {beta} | Use HW {self.use_hotwords}"
        )

    def transcribe(self, files, inference_hotwords=[], hotword_weight=10.0):
        if self.use_hotwords or self.use_lm:
            return self.transcribe_ctcdecoder(files, inference_hotwords, hotword_weight)
        else:
            return self.transcribe_greedy(files)

    def transcribe_greedy(self, files):
        transcript = self.asr_model.transcribe(paths2audio_files=files, batch_size=1)
        transcript = transcript[0]
        transcript, itn_transcript = standardize_output(transcript, self.lang)
        return transcript, itn_transcript

    def transcribe_ctcdecoder(self, files, inference_hotwords, hotword_weight=10.0):
        logits = self.asr_model.transcribe(
            paths2audio_files=files, batch_size=1, logprobs=True
        )[0]
        transcript = self.decoder.decode(
            logits, hotwords=inference_hotwords, hotword_weight=hotword_weight
        )
        transcript, itn_transcript = standardize_output(transcript, self.lang)
        return transcript, itn_transcript
