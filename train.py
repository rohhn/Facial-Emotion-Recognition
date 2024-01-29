"""
python train.py --train-file "dataset/fer_2013/train.csv" --model-save-dir model --epochs 5 --lr 0.001
"""

import argparse

import joblib
import numpy as np
import pandas as pd
import torch.optim as optimizers
from sklearn.preprocessing import LabelEncoder

from data import FERDataset
from model_utils import MultiClassClassifier, train_val_test_split, DataLoaders, torch_active_device
from models import SimpleCNNModel

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Facial Emotion Recognition - Training')
    parser.add_argument('--train-file', type=str, action='store', help='Training file path.', required=True)
    parser.add_argument('--epochs', type=int, help='Number of training epochs.', required=True)
    parser.add_argument('--lr', type=float, help='Learning rate.', required=True)
    parser.add_argument('--model-save-dir', type=str, action='store', help='Model save directory.')
    args = parser.parse_args()

    train_data = pd.read_csv(args.train_file)
    # test_data = pd.read_csv("dataset/fer_2013/test.csv")

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

    # convert string bytes into image arrays
    train_data['pixels'] = train_data['pixels'].apply(
        lambda x: np.fromstring(x, dtype='int', sep=' ').reshape(48, 48) / 255)

    # Data Splits
    x_train, y_train, x_val, y_val, _, _ = train_val_test_split(train_data['pixels'], train_data['emotion'],
                                                                val_size=0.1, stratify=True, test_size=0,
                                                                shuffle=True)

    # Label Encoder
    target_encoder = LabelEncoder()
    y_train = target_encoder.fit_transform(y_train)
    y_val = target_encoder.transform(y_val)

    num_classes = target_encoder.classes_.shape[0]

    # automatically applies IMAGE_TRANSFORMER to each image
    train_dataset = FERDataset(x_train, y_train, device=torch_active_device)
    val_dataset = FERDataset(x_val, y_val, device=torch_active_device)

    data = DataLoaders(batch_size=32)(train_dataset, val_dataset, test_dataset=None)

    model = SimpleCNNModel(num_classes=num_classes)
    print(model)

    train_params = {
        "train_loader": data.train_loader,
        "val_loader": data.val_loader,
        "num_epochs": args.epochs,
        "optimizer": optimizers.Adam,
        "optimizer_params": {
            "lr": args.lr
        },
        "return_outputs": True
    }

    model, outputs = MultiClassClassifier.train(model, **train_params)

    if args.model_save_dir:
        MultiClassClassifier.save_model(model, model.name, args.model_save_dir)
        joblib.dump(target_encoder, f"{args.model_save_dir}/target_encoder.joblib")
