import argparse
from datetime import datetime

import pandas as pd

DB_READ_PATH = "db/logger.tsv"
URL_BASE = "https://classlm.blob.core.windows.net/backend-logs/{}.wav"

parser = argparse.ArgumentParser()
parser.add_argument("--start-date", required=True)
parser.add_argument("--end-date", required=True)
args = parser.parse_args()


def convert_to_data_format(df):
    data = list()
    for i, row in df.iterrows():
        url = URL_BASE.format(row["audio"])
        pred_intent = row["intent"]
        pred_entities = eval(row["entities"])
        pred_entities_type = [ent["entity"] for ent in pred_entities]
        pred_entities_val = [ent["value"] for ent in pred_entities]
        sample_data = [url, row["date"], pred_intent]
        for i in range(len(pred_entities)):
            sample_data.extend([pred_entities_type[i], pred_entities_val[i]])
        data.append(sample_data)
    max_len = max(map(len, data))
    for d in data:
        d.extend([""] * (max_len - len(d)))
    column_names = ["URL", "Date", "Intent"]
    column_names.extend(["Entity Type", "Entity Value"] * ((max_len - 2) // 2))
    data_df = pd.DataFrame(data, columns=column_names)
    return data_df


db_df = pd.read_csv(DB_READ_PATH, sep="\t")
db_df["date"] = db_df["time"].apply(lambda x: datetime.strptime(x[:10], "%d/%m/%Y"))

start_date = datetime.strptime(args.start_date, "%d-%m-%Y")
end_date = datetime.strptime(args.end_date, "%d-%m-%Y")

db_df = db_df[db_df["date"].apply(lambda x: start_date <= x <= end_date)]
db_df = db_df.dropna()
db_df = db_df.drop_duplicates(subset=["lang", "transcript"])

db_df_en = db_df[db_df["lang"] == "English"]
db_df_hi = db_df[db_df["lang"] == "Hindi"]

data_en = convert_to_data_format(db_df_en)
data_hi = convert_to_data_format(db_df_hi)

data_en.to_csv(
    f"data-tagging/data/data-to-upload/{args.start_date}-to-{args.end_date}-en.csv", index=False
)
data_hi.to_csv(
    f"data-tagging/data/data-to-upload/{args.start_date}-to-{args.end_date}-hi.csv", index=False
)
