import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None, labels_to_ids=None):
        self.encodings = encodings
        self.labels = labels
        self.labels_to_ids = labels_to_ids

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels_to_ids[self.labels[idx]])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])
