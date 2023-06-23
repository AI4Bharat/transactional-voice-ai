import argparse
import json
import os

import pandas as pd

WRITE_PATH = "data/tagged-data-formatted/"

parser = argparse.ArgumentParser()
parser.add_argument("--data-file", required=True)
args = parser.parse_args()

df = pd.read_csv(args.data_file)
df = df[df["Is verified"] == "Yes"]

true_entities = list()
for i, row in df.iterrows():
    column_suffix = [""]
    column_suffix.extend([".{}".format(n) for n in range(1, 5)])
    entities = sorted(
        [
            "{}-{}".format(row["Entity Type" + suff], row["Entity Value" + suff])
            for suff in column_suffix
            if isinstance(row["Entity Type" + suff], str)
        ]
    )
    entities = [
        "{}-{}".format(t.strip(), v.strip().lower())
        for t, v in map(lambda x: x.split("-", 1), entities)
    ]
    true_entities.append(json.dumps(entities))

final_df = pd.DataFrame()
final_df["URL"] = df["URL"]
final_df["True Intent"] = df["Intent"]
final_df["True Entities"] = true_entities

final_df.to_csv(
    os.path.join(WRITE_PATH, args.data_file.rsplit("/", 1)[-1]), index=False
)
