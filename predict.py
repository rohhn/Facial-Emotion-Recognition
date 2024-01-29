import argparse
import os

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data import IMAGE_TRANSFORMER


def make_prediction(model_path, img):
    model_dir = os.path.dirname(model_path)

    model = torch.load(model_path)
    target_encoder = joblib.load(os.path.join(model_dir, "target_encoder.joblib"))

    model.eval()

    img_tensor = Image.open(img).convert('L')
    img_tensor = IMAGE_TRANSFORMER(np.array(img_tensor))

    y_pred = model(**{"input": img_tensor.unsqueeze(0)})
    y_pred = target_encoder.inverse_transform(F.softmax(y_pred, dim=1).argmax(-1).cpu().numpy())[0]

    return y_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facial Emotion Recognition - Prediction')
    parser.add_argument('--image-file', type=str, action='store', help='Image file path.', required=True)
    parser.add_argument('--model-path', type=str, action='store', help='Path to saved model.', required=True)
    args = parser.parse_args()

    print(make_prediction(args.model_path, args.image_file))
