import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_TRANSFORMER = transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=0., std=1.),
    transforms.Resize((48, 48), antialias=False),
])


class FERDataset(Dataset):
    """
    A templated PyTorch dataset class that can be custom fit to any dataset.
    """

    def __init__(self, inputs: pd.Series | list, labels: pd.Series | list, device: torch.device = "cpu"):
        self.inputs = inputs
        self.labels = labels
        self.device = device

        # Add any custom dataset manipulation logic here
        self.image_transformer = IMAGE_TRANSFORMER

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def rgb2gray(pixels):
        """
        Convert RGB image to grayscale.

        :param pixels:
        :return:
        """
        return np.dot(pixels[..., :3], [0.2989, 0.5870, 0.1140])

    def __getitem__(self, idx) -> tuple[dict, torch.Tensor]:
        """
        Inputs in this method must return a dictionary of tensors, for example:

        inputs = {
            "input": torch.tensor(self.inputs[idx], dtype=torch.float, device=self.device),
            "token_type_ids": torch.tensor(self.token_type_ids[idx], dtype=torch.long, device=self.device),
        }

        These inputs are fed into the model as keyword arguments.

        :param idx:
        :return:
        """

        if isinstance(self.inputs, pd.Series):
            inputs = {
                "input": self.image_transformer(self.inputs.iloc[idx])
            }
        else:
            inputs = {
                "input": self.image_transformer(self.inputs[idx])
            }

        if isinstance(self.labels, pd.Series):
            label = torch.tensor(self.labels.iloc[idx], dtype=torch.long, device=self.device)
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.long, device=self.device)

        return inputs, label
