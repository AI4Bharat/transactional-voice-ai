import yaml

import checks
from asr import hotword_utils
from asr.conformer import ConformerRecognizer
from entity.entity_recognizer import EntityRecognizer
from intent.intent_recognizer import IntentRecognizer


class PredictionPipeline:
    def __init__(self):
        with open("config.yaml") as f:
            self.config = yaml.load(f, yaml.BaseLoader)
        if not checks.check_intent():
            raise ValueError(
                "Unsupported intents found. Compare config and supported labels"
            )
        if not checks.check_entity():
            raise ValueError(
                "Unsupported entities found. Compare config and supported entities"
            )

        self.asr_mode = self.config["asr"]["hotword_mode"]
        if self.asr_mode == "none":
            use_hotwords = False
            self.hotwords_en = list()
            self.hotwords_hi = list()
        else:
            use_hotwords = True
            self.hotwords_en = hotword_utils.hotword_to_fn[self.asr_mode](lang="en")
            self.hotwords_hi = hotword_utils.hotword_to_fn[self.asr_mode](lang="hi")
        self.hotword_weight = float(self.config["asr"]["hotword_weight"])

        self.asr_en = ConformerRecognizer(
            self.config["asr"]["model_path"]["en"], "en", use_hotwords
        )
        self.asr_hi = ConformerRecognizer(
            self.config["asr"]["model_path"]["hi"], "hi", use_hotwords
        )

        self.intent_recognizer = IntentRecognizer(
            self.config["intent"]["model_path"],
            self.config["intent"]["label_dict_path"],
            float(self.config["intent"]["confidence_threshold"]),
        )

        self.entity_recognizer_en = EntityRecognizer("en")
        self.entity_recognizer_hi = EntityRecognizer("hi")

        self.lang_to_asr = {"en": self.asr_en, "hi": self.asr_hi}
        self.lang_to_hotword = {"en": self.hotwords_en, "hi": self.hotwords_hi}

        self.lang_to_entity = {
            "en": self.entity_recognizer_en,
            "hi": self.entity_recognizer_hi,
        }

    def predict(self, audio_file, lang, additional_hotwords=[], hotword_weight=None):
        list_hotwords = self.lang_to_hotword[lang] + additional_hotwords
        if not hotword_weight:
            hotword_weight = self.hotword_weight
        transcript, transcript_itn = self.lang_to_asr[lang].transcribe(
            [audio_file], list_hotwords, hotword_weight
        )
        intent, intent_orig, intent_prob = self.intent_recognizer.predict(
            transcript_itn
        )
        entities = self.lang_to_entity[lang].predict(transcript, transcript_itn)
        return {
            "transcript": transcript,
            "transcript_itn": transcript_itn,
            "intent": intent,
            "intent_orig": intent_orig,
            "intent_prob": intent_prob,
            "entities": entities,
        }
