import torch.nn as nn


class SimpleCNNModel(nn.Module):
    name = "SimpleCNNModel"

    @staticmethod
    def make_cnn_block(in_channels, out_channels, kernel_size=(3, 3), add_batchnorm=False):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        if add_batchnorm:
            block.add_module(name='batch_norm', module=nn.BatchNorm2d(out_channels))

        return block

    def __init__(self, num_classes: int):
        super().__init__()

        self.cnn_block_1 = self.make_cnn_block(1, 16, (3, 3), add_batchnorm=True)
        self.dropout1 = nn.Dropout(0.1)
        self.cnn_block_2 = self.make_cnn_block(16, 32, (3, 3), add_batchnorm=True)
        self.dropout2 = nn.Dropout(0.1)
        self.cnn_block_3 = self.make_cnn_block(32, 64, (3, 3), add_batchnorm=True)
        self.dropout3 = nn.Dropout(0.1)

        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True)
        )

        self.dropout4 = nn.Dropout(0.1)

        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True)
        )

        self.fc4 = nn.Sequential(
            nn.Linear(128, num_classes)
        )

    def forward(self, **kwargs):
        op = self.cnn_block_1(kwargs['input'])
        op = self.dropout1(op)
        op = self.cnn_block_2(op)
        op = self.dropout1(op)
        op = self.cnn_block_3(op)
        op = self.dropout1(op)

        op = op.flatten(1)

        op = self.fc1(op)
        op = self.dropout1(op)
        op = self.fc2(op)
        op = self.fc3(op)
        op = self.fc4(op)

        return op
