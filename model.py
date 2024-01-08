import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    def __init__(self, lstm_hidden_dim, attention_dim):
        super(TemporalAttention, self).__init__()
        self.attention_dim = attention_dim
        self.W = nn.Linear(lstm_hidden_dim, attention_dim)
        self.V = nn.Linear(attention_dim, 1)

    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, sequence_length, lstm_hidden_dim)

        batch_size, sequence_length, lstm_hidden_dim = lstm_output.shape
        attention_scores = torch.tanh(self.W(lstm_output))

        # Calculating the attention weights for each time step
        attention_weights = torch.softmax(self.V(attention_scores), dim=1)

        # Applying the attention weights to the LSTM output (element-wise multiplication)
        weighted_output = lstm_output * attention_weights.expand_as(lstm_output)

        # The shape of weighted_output is the same as lstm_output, preserving the temporal dimension
        return weighted_output


class TemporalFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(TemporalFeatureExtractor, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, batch_first=True)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, _)
        output, _ = self.lstm(x)
        return output
    

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out
