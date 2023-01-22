import torch
from torch import nn


class ConvModel(nn.Module):
    __name__ = "CNN"

    def __init__(self, mel_dim=80, dropout_rate=0.25, output_size=50, seq_len=16000 * 2 // 256 + 1):
        super().__init__()
        self.seq_len = seq_len

        self.conv1 = nn.Sequential(
                nn.Conv1d(in_channels=mel_dim, out_channels=128, kernel_size=9, stride=1, padding='same'),
                nn.ReLU(),
                nn.BatchNorm1d(128)
        )
        self.conv2 = nn.Sequential(
                nn.Conv1d(in_channels=128, out_channels=64, kernel_size=7, stride=1, padding='same'),
                nn.ReLU(),
                nn.BatchNorm1d(64)
        )
        self.conv3 = nn.Sequential(
                nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding='same'),
                nn.ReLU(),
                nn.BatchNorm1d(32)
        )
        self.connected_layer = nn.Sequential(
                nn.Flatten(),
                nn.Linear(seq_len * 32, 256),
                nn.Dropout(dropout_rate),
                nn.Linear(256, output_size),  # 200 unit
        )

    def forward(self, input_data):
        input_data = input_data
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.connected_layer(x)
        return x

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


class GRUModel(nn.Module):
    __name__ = "GRU"

    def __init__(self, mel_dim=80, dropout_rate=0.25, output_size=50, seq_len=16000 * 2 // 256 + 1):
        super().__init__()
        self.seq_len = seq_len
        self.conv1 = nn.Sequential(
                nn.Conv1d(in_channels=mel_dim, out_channels=mel_dim, kernel_size=9, stride=1, padding='same'),
                nn.ReLU(),
                nn.BatchNorm1d(80)
        )
        self.lstm1 = nn.GRU(input_size=seq_len, hidden_size=512, batch_first=True)

        self.connected_layer = nn.Sequential(
                nn.Linear(512, 256),
                nn.Dropout(dropout_rate),
                nn.Linear(256, output_size),  # 200 unit
        )

    def forward(self, input_data):
        input_data = input_data

        x = self.conv1(input_data)
        output, hn = self.lstm1(x)
        hn = hn.squeeze(0)
        x = self.connected_layer(hn)
        return x

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main_conv():
    print("# Conv")
    model = ConvModel().to('cpu')
    model.parameters()
    print(model.count_parameters())
    x = torch.ones((64, 80, 16000 * 2 // 256 + 1)).to('cpu')
    y = model(x)
    print(y.shape)


def main_gru():
    print("# LSTM")
    model = GRUModel().to('cpu')
    model.parameters()
    print(model.count_parameters())
    x = torch.ones((64, 80, 16000 * 2 // 256 + 1)).to('cpu')
    y = model(x)
    print(y.shape)


if __name__ == "__main__":
    main_conv()
    main_gru()
