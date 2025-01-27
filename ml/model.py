import multiprocessing
import math
import datasets
import torch
from pytorch_lightning import LightningModule
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ml.dataset import dataset_collate_function

#   MACROSS
class ECA(LightningModule):
    def __init__(self, channel, gamma=2, b=1):
        super(ECA, self).__init__()
        t = int(abs(math.log2(channel)) + 0.5)
        k_size = max(t, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.gamma = gamma
        self.b = b

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(self.gamma * y + self.b)
        return x * y.expand_as(x)

class ResidualBlock1D(LightningModule):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet1D_FeatureExtractor(LightningModule):
    def __init__(self, in_channels, layers):
        super(ResNet1D_FeatureExtractor, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        layers = []
        layers.append(ResidualBlock1D(self.inplanes, out_channels, stride, downsample))
        self.inplanes = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(self.inplanes, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x  

class MACROSS(LightningModule):
    def __init__(
        self,
        c1_output_dim,
        c1_kernel_size,
        c1_stride,
        c2_output_dim,
        c2_kernel_size,
        c2_stride,
        output_dim,
        data_path,
        signal_length,
    ):
        super().__init__()
        # save parameters to checkpoint
        self.save_hyperparameters()

        # fourier transform
        self.fft = torch.fft.fft
        self.abs = torch.abs

        # two convolution, then one max pool
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=self.hparams.c1_output_dim,
                kernel_size=self.hparams.c1_kernel_size,
                stride=self.hparams.c1_stride,
            ),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.hparams.c1_output_dim,
                out_channels=self.hparams.c2_output_dim,
                kernel_size=self.hparams.c2_kernel_size,
                stride=self.hparams.c2_stride,
            ),
            nn.ReLU(),
        )

        self.max_pool = nn.MaxPool1d(kernel_size=2)

        # use a dummy input to calculate
        dummy_x = torch.rand(1, 1, self.hparams.signal_length, requires_grad=False)
        dummy_x2 = dummy_x.clone()
        # branch1
        dummy_x = self.fft(dummy_x)
        dummy_x = self.abs(dummy_x)
        dummy_x = self.conv1(dummy_x)
        dummy_x = self.conv2(dummy_x)
        dummy_x = self.max_pool(dummy_x)
        # branch2
        dummy_x2 = self.conv1(dummy_x2)
        dummy_x2 = self.conv2(dummy_x2)
        dummy_x2 = self.max_pool(dummy_x2)
        # concat
        dummy_x = torch.cat([dummy_x, dummy_x2], dim=1)
        channels = dummy_x.shape[1]
        # ECA
        self.eca = ECA(channel=channels)
        dummy_x = self.eca(dummy_x)
        # print(dummy_x)
        # resnet
        self.resnet = nn.Sequential(
            ResNet1D_FeatureExtractor(in_channels=channels, layers=[2, 2, 2, 2]),
        )
        dummy_x = self.resnet(dummy_x)
        max_pool_out = dummy_x.view(1, -1).shape[1]

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=max_pool_out, out_features=200),
            nn.Dropout(p=0.05),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=200, out_features=100), nn.Dropout(p=0.05), nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=100, out_features=50), nn.Dropout(p=0.05), nn.ReLU()
        )

        # finally, output layer
        self.out = nn.Linear(in_features=50, out_features=self.hparams.output_dim)

    def forward(self, x):
        # make sure the input is in [batch_size, channel, signal_length]
        # where channel is 1
        # signal_length is 1500 by default
        # print(x.shape)
        batch_size = x.shape[0]
        # copy input
        x2 = x.clone()

        # branch 1
        # get frequency feature
        x = self.fft(x)
        x = self.abs(x)  # take the amplitude
        # 2 conv 1 max
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_pool(x)

        # branch 2
        # get time feature
        # 2 conv 1 max
        x2 = self.conv1(x2)
        x2 = self.conv2(x2)
        x2 = self.max_pool(x2)
        # concat
        x = torch.cat([x, x2], dim=1)
        # print(x.shape)

        # time-frequency feature fusion
        x = self.eca(x)
        # fusion feature extraction
        # print(x.shape)
        x = self.resnet(x)

        # classification
        x = x.reshape(batch_size, -1)
        # print(x.shape)
        # 3 fc
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        # output
        x = self.out(x)

        return x

    def train_dataloader(self):
        # expect to get train folder
        dataset_dict = datasets.load_dataset(self.hparams.data_path)
        dataset = dataset_dict[list(dataset_dict.keys())[0]]
        try:
            num_workers = multiprocessing.cpu_count()
        except:
            num_workers = 1
        dataloader = DataLoader(
            dataset,
            batch_size=16,
            num_workers=num_workers,
            collate_fn=dataset_collate_function,
            shuffle=True,
        )

        return dataloader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        x = batch["feature"].float()
        y = batch["label"].long()
        y_hat = self(x)

        entropy = F.cross_entropy(y_hat, y)
        self.log(
            "training_loss",
            entropy,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        loss = {"loss": entropy}

        return loss
