import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from model_utils import torch_active_device

IMAGE_TRANSFORMER_TORCH = transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=0., std=1.),
    transforms.Resize((48, 48), antialias=False)
])


def read_fer_data(train_file, test_file=None):
    train_data = pd.read_csv(train_file)

    # 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral.
    label_map = {
        0: "angry",
        1: "disgust",
        2: "fear",
        3: "happy",
        4: "sad",
        5: "surprise",
        6: "neutral"
    }

    train_data["emotion"] = train_data["emotion"].apply(lambda x: label_map[x])
    train_data['pixels'] = train_data['pixels'].apply(lambda x: np.fromstring(x, dtype='uint8', sep=' ').reshape(48, 48))

    if test_file is not None:
        test_data = pd.read_csv(test_file)
        test_data["emotion"] = test_data["emotion"].apply(lambda x: label_map[x])
        test_data['pixels'] = test_data['pixels'].apply(
            lambda x: np.fromstring(x, dtype='uint8', sep=' ').reshape(48, 48))
    else:
        test_data = None

    return train_data, test_data


class FERDataset(Dataset):
    """
    A templated PyTorch dataset class that can be custom fit to any dataset.
    """

    def __init__(self, inputs: pd.Series | list, labels: pd.Series | list, device: torch.device = "cpu", convert_to_3_channel=False):
        self.inputs = inputs
        self.labels = labels
        self.device = device

        # Add any custom dataset manipulation logic here
        self.image_transformer = IMAGE_TRANSFORMER_TORCH
        self.convert_to_3_channel = convert_to_3_channel

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
            x = self.image_transformer(self.inputs.iloc[idx])
        else:
            x = self.image_transformer(self.inputs[idx])

        if self.convert_to_3_channel:
            x = torch.repeat_interleave(x, 3, 0)

        inputs = {
            "x": x.to(self.device)
        }

        if isinstance(self.labels, pd.Series):
            label = torch.tensor(self.labels.iloc[idx], dtype=torch.long, device=self.device)
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.long, device=self.device)

        return inputs, label


def cv2_face_segmentation(img, padding=25, convert_grayscale=False):

    face_clf = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    if convert_grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_op = face_clf.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    if len(face_op) > 0:  # face detected
        x, y, w, h = face_op[0]

        face = img[y - padding: y + h + padding, x - padding: x + w + padding]
        # face = cv2.resize(face, (48, 48))
    else:
        face = img

    return face
