import torch
from torch import nn
from transformers import HubertModel, Wav2Vec2FeatureExtractor


class HubertDense(nn.Module):
    __name__ = "Hubert"

    def __init__(self, mel_dim=80, dropout_rate=0.25, output_size=50, seq_len=16000 * 2 // 256 + 1):
        super().__init__()

        # self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
        # self.model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=99, out_channels=1, kernel_size=50, stride=10),
            nn.ReLU(),
            nn.BatchNorm1d(1)
        )
        
        self.connected_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(72, 15),
            nn.Dropout(dropout_rate),
            nn.Linear(15, output_size),  
        )

    def forward(self, h_n):
        # inputs = self.feature_extractor(x, sampling_rate=16000, return_tensors="pt")

        # with torch.no_grad():
        #     h_n = self.model(**inputs).last_hidden_state
        x = self.conv1(h_n)
        x = self.connected_layer(x)
        return x
            


class ConvModel(nn.Module):
    __name__ = "CNN"

    def __init__(self, mel_dim=80, dropout_rate=0.25, output_size=50, seq_len=16000 * 2 // 256 + 1):
        super().__init__()
        self.seq_len = seq_len

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=mel_dim, out_channels=30, kernel_size=11, stride=3),
            nn.ReLU(),
            nn.BatchNorm1d(30)
        )
        # self.conv2 = nn.Sequential(
        #     nn.Conv1d(in_channels=40, out_channels=20, kernel_size=11, stride=2),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(20)
        # )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=30, out_channels=10, kernel_size=11, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(10)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=10, out_channels=50, kernel_size=11, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(50)
        )
        self.connected_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(150, 15),
            nn.Dropout(dropout_rate),

            nn.Linear(15, 50),

            # nn.Dropout(dropout_rate),
            # nn.Linear(256, output_size),  # 200 unit
        )

    def forward(self, input_data):
        input_data = input_data
        x = self.conv1(input_data)
        # x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
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
            nn.Conv1d(in_channels=mel_dim, out_channels=16, kernel_size=9, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(16)
        )
        self.lstm1 = nn.GRU(input_size=16, hidden_size=32, batch_first=True)

        self.connected_layer = nn.Sequential(
            nn.Linear(32, 64),
            nn.Dropout(dropout_rate),
            nn.Linear(64, output_size),  # 200 unit
        )

    def forward(self, input_data):
        input_data = input_data

        x = self.conv1(input_data)
        output, hn = self.lstm1(x.transpose(2, 1))
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

def main_hubert():
    model = HubertDense().to('cpu')
    x = torch.ones((16000 * 2 )).to('cpu')
    y = model(x)

if __name__ == "__main__":
    main_hubert()
    main_conv()
    main_gru()
