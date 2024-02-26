import torch.nn as nn
import torchvision


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
        op = self.cnn_block_1(kwargs['x'])
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


class SimpleCNNModel2(nn.Module):
    name = "SimpleCNNModel2"

    @staticmethod
    def make_cnn_block(in_channels, out_channels, kernel_size=(3, 3), activate=True, pooling=False,
                       add_batchnorm=False):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
        )

        if activate:
            block.add_module(name='activation', module=nn.ReLU(inplace=True))

        if pooling:
            block.add_module(name='pooling', module=nn.MaxPool2d(kernel_size=(2, 2)))

        if add_batchnorm:
            block.add_module(name='batch_norm', module=nn.BatchNorm2d(out_channels))

        return block

    def __init__(self, num_classes: int):
        super().__init__()

        self.cnn_block_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )  # 52

        self.cnn_block_2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )  # 50

        self.maxpool_1 = nn.MaxPool2d((2, 2))  # 25

        self.cnn_block_3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )  # 23

        self.cnn_block_4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )  # 21

        self.maxpool_2 = nn.MaxPool2d((2, 2))  # 10

        self.fc_block_1 = nn.Sequential(
            nn.Linear(6400, 3200),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(3200)
        )

        self.fc_block_2 = nn.Sequential(
            nn.Linear(3200, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024)
        )

        self.fc_block_3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512)
        )

        self.final_layer = nn.Linear(512, num_classes)

    def forward(self, **kwargs):
        op = self.cnn_block_1(kwargs['x'])
        op = self.cnn_block_2(op)
        op = self.maxpool_1(op)
        op = self.cnn_block_3(op)
        op = self.cnn_block_4(op)
        op = self.maxpool_2(op)

        op = op.flatten(1)

        op = self.fc_block_1(op)
        op = self.fc_block_2(op)
        op = self.fc_block_3(op)
        op = self.final_layer(op)

        return op


class AlexNetTransfer(nn.Module):
    name = "ALexNetTransfer"

    @staticmethod
    def make_cnn_block(in_channels, out_channels, activate=True, pooling=False, add_batchnorm=False, **kwargs):

        cnn_args = kwargs.get('cnn_args', {})

        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=cnn_args.get('kernel_size', (3, 3)),
                      stride=cnn_args.get('stride', (1, 1)), padding=cnn_args.get('padding', (104, 104))
                      )
        )

        if activate:
            block.add_module(name='activation', module=nn.ReLU(inplace=True))

        if pooling:
            pooling_args = kwargs.get('pooling_args', {})
            block.add_module(name='pooling', module=nn.MaxPool2d(kernel_size=pooling_args.get('kernel_size', (2, 2)),
                                                                 stride=pooling_args.get('stride', (1, 1))))

        if add_batchnorm:
            block.add_module(name='batch_norm', module=nn.BatchNorm2d(out_channels))

        return block

    def __init__(self, num_classes: int):
        super().__init__()

        self.alex_net = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT)
        # first_cnn = model.get_submodule('features').get_submodule('0')
        # first_cnn.parameters(in_channels=1)
        clf_module = self.alex_net.get_submodule('classifier')
        clf_module.add_module('7', nn.ReLU(inplace=True))
        # clf_module.add_module('final_layer', nn.Linear(in_features=1000, out_features=num_classes))

        self.final_layer = nn.Linear(1000, num_classes)

    def forward(self, **kwargs):

        op = self.alex_net(kwargs['x'])
        op = self.final_layer(op)

        return op


class MobileNetV2(nn.Module):
    name = "MobileNetV2"

    @staticmethod
    def make_cnn_block(in_channels, out_channels, activate=True, pooling=False, add_batchnorm=False, **kwargs):

        cnn_args = kwargs.get('cnn_args', {})

        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=cnn_args.get('kernel_size', (3, 3)),
                      stride=cnn_args.get('stride', (1, 1)), padding=cnn_args.get('padding', (104, 104))
                      )
        )

        if activate:
            block.add_module(name='activation', module=nn.ReLU(inplace=True))

        if pooling:
            pooling_args = kwargs.get('pooling_args', {})
            block.add_module(name='pooling', module=nn.MaxPool2d(kernel_size=pooling_args.get('kernel_size', (2, 2)),
                                                                 stride=pooling_args.get('stride', (1, 1))))

        if add_batchnorm:
            block.add_module(name='batch_norm', module=nn.BatchNorm2d(out_channels))

        return block

    def __init__(self, num_classes: int):
        super().__init__()

        self.mobilenet = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT)
        mobilenet_clf = self.mobilenet.get_submodule('classifier')
        mobilenet_clf.add_module('2', nn.ReLU(inplace=True))

        self.final_layer = nn.Linear(1000, num_classes)

    def forward(self, **kwargs):

        op = self.mobilenet(kwargs['x'])
        op = self.final_layer(op)

        return op
