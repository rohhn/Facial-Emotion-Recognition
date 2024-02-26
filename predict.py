import argparse
import os

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data import IMAGE_TRANSFORMER_TORCH, cv2_face_segmentation
from model_utils import torch_active_device


def make_prediction(model_name, img):
    """

    :param model_name:
    :param img:
    :return:
    """
    # model_dir = os.path.dirname(model_name)

    # Load model and LabelEncoder
    model = torch.load(os.path.join(model_name, model_name), map_location=torch_active_device)
    target_encoder = joblib.load(os.path.join(model_name, f"{model_name}.joblib"))

    model.to(torch_active_device)
    model.eval()

    img_tensor = Image.open(img).convert('L')  # read image

    img_tensor = cv2_face_segmentation(np.array(img_tensor, dtype='uint8'))  # segment face if found

    img_tensor = IMAGE_TRANSFORMER_TORCH(img_tensor)

    y_pred = model(**{"x": img_tensor.unsqueeze(0)})
    y_pred = target_encoder.inverse_transform(F.softmax(y_pred, dim=1).argmax(-1).cpu().numpy())[0]

    return y_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facial Emotion Recognition - Prediction')
    parser.add_argument('--image-file', type=str, action='store', help='Image file path.', required=True)
    parser.add_argument('--model-path', type=str, action='store', help='Path to saved model.', required=True)
    args = parser.parse_args()

    print(make_prediction(args.model_path, args.image_file))
