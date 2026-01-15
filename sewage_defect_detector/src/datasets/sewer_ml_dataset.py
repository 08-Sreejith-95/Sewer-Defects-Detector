import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image



class SewerMLDataset(Dataset):
    def __init__(self, cfg=None, split="train", transform=None, df=None):
        self.cfg = cfg
        self.split = split
        self.transform = transform

        if df is not None:
            self.data = df.reset_index(drop=True)
        elif cfg is not None:
            from src.path import get_csv_path
            self.data = pd.read_csv(get_csv_path(cfg, split))
        else:
            raise ValueError("Provide either cfg or df")

        # dataset root
        from src.path import get_image_dir
        self.image_dir = get_image_dir(cfg, split) if cfg else None
        if split != "test":
            self.label_cols = self.data.columns[1:]

            # FORCE numeric, handle strings & NaNs
            self.data[self.label_cols] = (
                self.data[self.label_cols]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
                .astype("float32")
            )
        else:
            self.label_cols = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.split != "test":
            label = self.data.loc[idx][self.label_cols].to_numpy(dtype="float32")
            label = torch.from_numpy(label)
            return image, label
        else:
            return image, img_name
