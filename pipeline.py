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
        if not checks.check_intent(self.config):
            raise ValueError(
                "Unsupported intents found. Compare config and supported labels"
            )
        if not checks.check_entity(self.config):
            raise ValueError(
                "Unsupported entities found. Compare config and supported entities"
            )
        # TODO CHECK ASR DEFINITIONS FOR SUPPORTED LANGUAGES
        # TODO CHECK ASR LM PARAMS IF LM IS TO BE USED
        # TODO CHECK ASR HW PARAMS IF HW IS TO BE USED

        self.supported_languages = self.config["supported_languages"]

        # Initialize the ASR
        self.lang_to_asr = dict()
        self.lang_to_hw_list = dict()
        self.lang_to_hw_weight = dict()
        for lang in self.supported_languages:
            lang_config = self.config["asr"][lang]
            lang_am = lang_config["am_path"]
            lang_lm = lang_config["lm_path"]
            lang_lm_alpha = lang_config["lm_alpha"]
            lang_lm_beta = lang_config["lm_beta"]
            lang_hw_mode = lang_config["hotword_mode"]
            lang_hw_weight = lang_config["hotword_weight"]

            if lang_lm:
                lang_lm_alpha = float(lang_lm_alpha)
                lang_lm_beta = float(lang_lm_beta)

            if lang_hw_mode == "none":
                lang_use_hw = False
                self.lang_to_hw_weight[lang] = 0.0
                self.lang_to_hw_list[lang] = list()
            else:
                lang_use_hw = True
                self.lang_to_hw_weight[lang] = float(lang_hw_weight)
                self.lang_to_hw_list[lang] = hotword_utils.hotword_to_fn[lang_hw_mode](
                    lang=lang
                )

            self.lang_to_asr[lang] = ConformerRecognizer(
                lang=lang,
                model_path=lang_am,
                lm_path=lang_lm,
                alpha=lang_lm_alpha,
                beta=lang_lm_beta,
                use_hotwords=lang_use_hw,
            )

        self.intent_recognizer = IntentRecognizer(
            self.config["intent"]["model_path"],
            self.config["intent"]["label_dict_path"],
            float(self.config["intent"]["confidence_threshold"]),
        )

        self.lang_to_entity_recognizer = dict()
        for lang in self.supported_languages:
            self.lang_to_entity_recognizer[lang] = EntityRecognizer(lang)

    def predict(self, audio_file, lang, additional_hotwords=[], hotword_weight=None):
        if lang not in self.supported_languages:
            return {
                "transcript": "",
                "transcript_itn": "",
                "intent": "unsupported_lang",
                "intent_orig": "unsupported_lang",
                "intent_prob": 1.0,
                "entities": [],
            }

        list_hotwords = self.lang_to_hw_list[lang] + additional_hotwords
        if not hotword_weight:
            hotword_weight = self.lang_to_hw_weight[lang]
        list_hotwords = self.lang_to_hw_list[lang] + additional_hotwords
        transcript, transcript_itn = self.lang_to_asr[lang].transcribe(
            [audio_file], list_hotwords, hotword_weight
        )

        intent, intent_orig, intent_prob = self.intent_recognizer.predict(
            transcript_itn
        )
        entities = self.lang_to_entity_recognizer[lang].predict(
            transcript, transcript_itn
        )
        return {
            "transcript": transcript,
            "transcript_itn": transcript_itn,
            "intent": intent,
            "intent_orig": intent_orig,
            "intent_prob": intent_prob,
            "entities": entities,
        }
