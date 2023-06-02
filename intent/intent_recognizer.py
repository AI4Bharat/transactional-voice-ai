import pickle

import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

from intent import utils

MODEL = "ai4bharat/indic-bert"


def softmax(x):
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=1, keepdims=True)


class IntentRecognizer:
    def __init__(self, model_path, label_dict_pkl, conf_threshold):
        with open(label_dict_pkl, "rb") as f:
            self.labels_to_ids = pickle.load(f)
        self.ids_to_labels = {
            intent_id: intent_label
            for intent_label, intent_id in self.labels_to_ids.items()
        }
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=len(self.labels_to_ids)
        )
        self.test_trainer = Trainer(self.model)
        self.conf_threshold = conf_threshold

    def predict(self, sentence):
        sentence = [sentence]
        sentence_tokenized = self.tokenizer(
            sentence, padding=True, truncation=True, max_length=512
        )
        model_in = utils.Dataset(sentence_tokenized)
        raw_pred, _, _ = self.test_trainer.predict(model_in)
        probs = softmax(raw_pred)
        y_pred = np.argmax(probs, axis=1)[0]
        pred_prob = np.max(probs)

        orig_pred = self.ids_to_labels[y_pred]
        pred = self.ids_to_labels[y_pred]
        if pred_prob < self.conf_threshold:
            pred = "unknown"
        return pred, orig_pred, pred_prob
