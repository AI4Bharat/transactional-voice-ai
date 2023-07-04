import argparse

import pandas as pd
from intent_recognizer import IntentRecognizer
from sklearn.metrics import classification_report
from tqdm import tqdm


def test(args):
    df = pd.read_csv(args.gt_file)
    df = df.dropna()

    intent_recognizer = IntentRecognizer(
        args.model_path, args.label_dict_pkl, conf_threshold=0.85
    )

    true_intent = list()
    pred_intent = list()

    for i, row in tqdm(df.iterrows()):
        true_intent.append(row["True Intent"])
        intent, _, _ = intent_recognizer.predict(row["Transcript"])
        pred_intent.append(intent)

    df["Predicted Intent"] = pred_intent
    print(classification_report(true_intent, pred_intent))
    df.to_csv(args.sample_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-file", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--label-dict-pkl", required=True)
    parser.add_argument("--sample-file", required=True)
    args = parser.parse_args()
    test(args)
