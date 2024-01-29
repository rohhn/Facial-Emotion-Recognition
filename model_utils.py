import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizers
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

torch_active_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_val_test_split(x, y, val_size, test_size=0, shuffle=True, stratify=False, random_state=None):
    if test_size > 0:
        stratify_by = y if stratify else None
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=test_size, random_state=random_state,
                                                            shuffle=shuffle, stratify=stratify_by)
    else:
        x_train, y_train = x, y
        x_test, y_test = None, None

    stratify_by = y_train if stratify else None
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      test_size=val_size, random_state=random_state,
                                                      shuffle=shuffle, stratify=stratify_by)

    print(f"X_train: {x_train.shape}")
    print(f"X_val: {x_val.shape}")
    if x_test is not None:
        print(f"X_test: {x_test.shape}")

    return x_train, y_train, x_val, y_val, x_test, y_test


class DataLoaders:

    def __init__(self, batch_size=4):
        self.batch_size = batch_size
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def __call__(self, train_dataset, val_dataset, test_dataset=None):
        if test_dataset:
            self.test_loader = DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=False
            )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        return self


class MultiClassClassifier:

    @staticmethod
    def train(model: nn.Module, train_loader, val_loader=None, num_epochs=10, learning_rate=0.01,
              optimizer=optimizers.Adam, class_weights=None, optimizer_params=None, return_outputs=False):

        if optimizer_params is None:
            optimizer_params = {}

        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float, device=torch_active_device)

        outputs = {
            "train": {
                "loss": [],
                "accuracy": [],
                "outputs": []
            },
            "val": {
                "loss": [],
                "accuracy": [],
                "outputs": []
            },

        }

        cost_fn = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optimizer(model.parameters(), **optimizer_params)

        for epoch in range(num_epochs):

            train_loss = 0
            train_predictions = np.zeros((len(train_loader.dataset),)) + 999
            train_correct = 0
            with tqdm(total=len(train_loader.dataset) // train_loader.batch_size,
                      desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
                for batch_no, batch_data in enumerate(train_loader):
                    x, y = batch_data

                    optimizer.zero_grad()

                    y_pred = model(**x)

                    loss = cost_fn(y_pred, y)
                    loss.backward()

                    optimizer.step()

                    batch_pred = F.softmax(y_pred.detach(), dim=1).cpu().numpy().argmax(-1)
                    train_predictions[
                    batch_no * train_loader.batch_size: (batch_no + 1) * train_loader.batch_size] = batch_pred
                    batch_correct = (batch_pred == y.numpy()).sum()
                    train_correct += batch_correct
                    train_loss += loss.item() / len(train_loader)

                    pbar.set_postfix(  # update TQDM progress bar
                        {"Train Loss": loss.item(),
                         "Train Accuracy": train_correct / ((batch_no + 1) * train_loader.batch_size),
                         "Correct Values": f"{train_correct}/{train_loader.batch_size * (batch_no + 1)}"
                         })

                    # "Train Accuracy": round(  # old training accuracy calculation, this was not accurate
                    #     accuracy_score(train_loader.dataset.labels[: (batch_no + 1) * train_loader.batch_size],
                    #                    train_predictions[: (batch_no + 1) * train_loader.batch_size]), 3),
                    pbar.update(1)

            train_accuracy = train_correct / ((batch_no + 1) * train_loader.batch_size)

            # update outputs
            outputs['train']['loss'] += [train_loss]
            outputs['train']['accuracy'] += [train_accuracy]

            # temporary for seeing predictions
            print(f"Predicted labels counts in training data:")
            print(np.unique(train_predictions, return_counts=True))

            if val_loader is not None:
                val_loss = 0
                val_predictions = np.zeros((len(val_loader.dataset),))
                with torch.no_grad():
                    for batch_no, batch_data in enumerate(val_loader):
                        x, y = batch_data

                        y_pred = model(**x)
                        loss = cost_fn(y_pred, y).item()

                        val_loss += loss / len(val_loader)
                        val_predictions[
                        batch_no * val_loader.batch_size: (batch_no + 1) * val_loader.batch_size] = F.softmax(
                            y_pred.detach(), dim=1).cpu().numpy().argmax(-1)

                val_accuracy = accuracy_score(val_loader.dataset.labels, val_predictions)

                validation_status = f" | Val loss: {round(val_loss, 3)} | Val accuracy: {round(val_accuracy, 3)}"

                outputs['val']['loss'] += [val_loss]
                outputs['val']['accuracy'] += [val_accuracy]
            else:
                validation_status = ""

            print(
                f"Epoch {epoch + 1}/{num_epochs} | Training loss: {round(train_loss, 3)} | Train accuracy 2: {round(train_accuracy, 3)}{validation_status}")

            if not return_outputs:
                outputs = {}

        return model, outputs

    @staticmethod
    def predict(model, data_loader):

        predictions = np.zeros((len(data_loader.dataset),))
        with torch.no_grad():
            for batch_no, batch_data in tqdm(enumerate(data_loader),
                                             total=len(data_loader.dataset) // data_loader.batch_size,
                                             desc=f"Predictions"):
                x, y = batch_data

                y_pred = model(**x)
                predictions[batch_no * data_loader.batch_size: (batch_no + 1) * data_loader.batch_size] = F.softmax(
                    y_pred.detach(), dim=1).argmax(-1).cpu().numpy()

        return predictions

    @staticmethod
    def save_model(model, file_name, save_dir="saved"):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        save_path = os.path.join(save_dir, file_name)
        torch.save(model, save_path)
        print(f"Model saved to {save_path}")
        return save_path
