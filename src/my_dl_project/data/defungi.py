from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

DEFUNGI_CLASSES = ["H1", "H2", "H3", "H5", "H6"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class DeFungiDataset(Dataset):
    """Custom PyTorch Dataset for DeFungi.

    Expects directory structure:
        root_dir/
        ├── H1/
        ├── H2/
        ├── H3/
        ├── H5/
        └── H6/
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_to_idx = {cls: i for i, cls in enumerate(DEFUNGI_CLASSES)}

        self.samples = []
        for cls in DEFUNGI_CLASSES:
            cls_dir = self.root_dir / cls
            if not cls_dir.is_dir():
                continue
            for path in sorted(cls_dir.iterdir()):
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label_idx = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(label_idx, dtype=torch.long)
        return image, label
