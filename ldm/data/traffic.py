import os, warnings, yaml, pickle, shutil, tarfile, glob
import cv2
import albumentations
import PIL
import numpy as np
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from functools import partial
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
import random


class TrafficBase(Dataset):
    def __init__(self,
                 data_root,
                 txt_file,
                 size = None,
                 ):
        self.data_root = os.path.abspath(data_root)
        self.data_paths = txt_file
        self.split = os.path.splitext(os.path.basename(txt_file))[0]
        with open(self.data_paths, "r") as f:
            raw_image_paths = [line.strip() for line in f.read().splitlines() if line.strip()]

        self.image_paths = []
        invalid_paths = []
        for entry in raw_image_paths:
            try:
                resolved_path = self._resolve_data_path(entry)
            except FileNotFoundError:
                invalid_paths.append(entry)
                continue

            if os.path.getsize(resolved_path) == 0:
                invalid_paths.append(entry)
                continue

            self.image_paths.append(resolved_path)

        if invalid_paths:
            warnings.warn(
                f"Skipped {len(invalid_paths)} missing or empty traffic samples from {self.data_paths}."
            )

        if not self.image_paths:
            raise ValueError(
                f"No usable traffic samples were found under {self.data_root}. "
                "Download and extract the BjTT dataset so the split data/ directories contain real .npy files."
            )

        self._length = len(self.image_paths)
        self.size = size

    def _resolve_data_path(self, entry):
        entry = entry.strip()
        candidates = [entry]

        if not os.path.splitext(entry)[1]:
            candidates.append(f"{entry}.npy")

        candidates.extend([
            os.path.join(self.data_root, entry),
            os.path.join(self.data_root, f"{entry}.npy"),
            os.path.join(self.data_root, self.split, "data", entry),
            os.path.join(self.data_root, self.split, "data", f"{entry}.npy"),
        ])

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        raise FileNotFoundError(f"Could not resolve traffic sample path for '{entry}' under '{self.data_root}'")

    def _resolve_text_path(self, entry):
        stem = os.path.splitext(os.path.basename(entry))[0]
        candidates = [
            os.path.join(self.data_root, self.split, "text", f"{stem}.txt"),
            os.path.join(self.data_root, "text", f"{stem}.txt"),
        ]

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        raise FileNotFoundError(f"Could not resolve traffic caption path for '{entry}' under '{self.data_root}'")

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict()
        path = self.image_paths[i]
        traffic_npy = np.load(path, allow_pickle=True)

        # traffic_npy = np.array(traffic_npy).astype(np.uint8)
        traffic_npy = traffic_npy.astype(np.float32)
        n_channels = traffic_npy.shape[2]

        # Pad to 3 channels if dataset only provides 2
        if n_channels < 3:
            pad = np.zeros((*traffic_npy.shape[:2], 3 - n_channels), dtype=np.float32)
            traffic_npy = np.concatenate([traffic_npy, pad], axis=2)

        traffic_npy[:,:,2][traffic_npy[:,:,2] > 3600] = 3600
        traffic_npy[:,:,1][traffic_npy[:,:,1] > 150] = 150

        traffic_npy[:,:,0] = (traffic_npy[:,:,0] / 5.0).astype(np.float32)
        traffic_npy[:,:,1] = (traffic_npy[:,:,1] / 150.0).astype(np.float32)
        traffic_npy[:,:,2] = (traffic_npy[:,:,2] / 3600.0).astype(np.float32)
        example['image'] = traffic_npy

        textpath = self._resolve_text_path(path)
        with open(textpath, "r") as f:
            text = str(f.read().splitlines()[0])
        example['caption'] = text
        # example['structure'] = np.load('/home/zcy/latent-diffusion-main/datasets/traffic/matrix_roadclass&length.npy')
        return example




class TrafficTrain(TrafficBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class TrafficValidation(TrafficBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
