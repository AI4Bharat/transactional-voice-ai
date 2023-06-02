import argparse
import json
from collections import defaultdict

import pandas as pd
import tqdm
from entity_recognizer import EntityRecognizer


def calc_metrics(true_entities, pred_entities):
    n_TP = defaultdict(lambda: 0)
    n_FP = defaultdict(lambda: 0)
    n_FN = defaultdict(lambda: 0)
    n_count = defaultdict(lambda: 0)
    count = len(true_entities)
    is_correct = list()
    for i in range(count):
        pred = pred_entities[i]
        true = true_entities[i]
        is_correct.append(true == pred)
        for p in pred:
            if p in true:
                n_TP[p.split("-")[0]] += 1
            else:
                n_FP[p.split("-")[0].strip()] += 1
        for t in true:
            n_count[t.split("-")[0].strip()] += 1
            if t not in pred:
                n_FN[t.split("-")[0]] += 1

    entity_types = sorted(
        list(set(list(n_TP.keys()) + list(n_FP.keys()) + list(n_FN.keys())))
    )
    entity_report = list()
    for ent in entity_types:
        try:
            precision = (n_TP[ent]) / (n_TP[ent] + n_FN[ent])
        except ZeroDivisionError:
            precision = 0
        try:
            recall = (n_TP[ent]) / (n_TP[ent] + n_FP[ent])
        except ZeroDivisionError:
            recall = 0
        try:
            f1 = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            f1 = 0
        precision, recall, f1 = (
            round(precision * 100),
            round(recall * 100),
            round(f1 * 100),
        )
        entity_report.append([ent, n_count[ent], precision, recall, f1])
    entity_report.append(
        ["Total", count, "", "", round(100 * sum(is_correct) / len(is_correct))]
    )
    entity_report_df = pd.DataFrame(
        entity_report,
        columns=["Entity Type", "Count", "Precision", "Recall", "F1 Score"],
    )
    return entity_report_df


def normalize_entities(ent_list):
    normalized_ent = list()
    for ent in ent_list:
        ent_type, ent_val = ent.split("-", 1)
        ent_type = ent_type.strip().lower()
        ent_val = ent_val.strip().lower().replace(" ", "")
        normalized_ent.append("{}-{}".format(ent_type, ent_val))
    return normalized_ent


def main(args):
    entity_recognizer = EntityRecognizer(lang=args.lang)
    data_df = pd.read_csv(args.gt_file)
    entity_data = list()
    for i, row in tqdm.tqdm(data_df.iterrows()):
        entities = entity_recognizer.predict(row["Transcript"], row["Transcript ITN"])
        entity_data.append(entities)
    data_df["Predicted Entities"] = entity_data
    data_df["Predicted Entities"] = data_df["Predicted Entities"].apply(
        lambda x: json.dumps(
            sorted(["{}-{}".format(ent["entity"], ent["value"]) for ent in x])
        )
    )

    true_entities = data_df["True Entities"].apply(json.loads).apply(normalize_entities)
    pred_entities = (
        data_df["Predicted Entities"].apply(json.loads).apply(normalize_entities)
    )

    data_df["Entity Correct"] = true_entities == pred_entities
    entity_report_df = calc_metrics(true_entities, pred_entities)
    entity_report_df.to_csv(args.report_file, index=False)
    data_df.to_csv(args.sample_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-file", required=True)
    parser.add_argument("--lang", choices=["en", "hi"], required=True)
    parser.add_argument("--report-file", required=True)
    parser.add_argument("--sample-file", required=True)
    args = parser.parse_args()
    main(args)
