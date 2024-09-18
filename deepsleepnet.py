import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepFeatureNet(nn.Module):
    def __init__(self, input_dims, n_classes, use_dropout):
        super(DeepFeatureNet, self).__init__()
        
        # First CNN with small filter size
        self.conv1a = nn.Conv1d(3, 64, kernel_size=50, stride=6, padding=24)  # Padding calculated
        self.pool1a = nn.MaxPool1d(kernel_size=8, stride=8)
        
        # Second CNN with large filter size
        self.conv1b = nn.Conv1d(3, 64, kernel_size=400, stride=50, padding=200)  # Padding calculated
        self.pool1b = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=4)  # Padding calculated
        self.conv22 = nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=4)  # Padding calculated
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.5)
        
        self.fc = nn.Linear(2560, 1024)  # Adjust the dimensions according to your input size

    def forward(self, x):
        # First CNN path
        x1 = F.relu(self.conv1a(x))
        x1 = self.pool1a(x1)
        if self.use_dropout:
            x1 = self.dropout(x1)
        
        x1 = F.relu(self.conv2(x1))
        x1 = F.relu(self.conv22(x1))
        x1 = F.relu(self.conv22(x1))
        x1 = self.pool2(x1)
        x1 = x1.view(x1.size(0), -1)
        
        # Second CNN path
        x2 = F.relu(self.conv1b(x))
        x2 = self.pool1b(x2)
        if self.use_dropout:
            x2 = self.dropout(x2)
        
        x2 = F.relu(self.conv2(x2))
        x2 = F.relu(self.conv22(x2))
        x2 = F.relu(self.conv22(x2))
        x2 = self.pool2(x2)
        x2 = x2.view(x2.size(0), -1)
        
        # Concatenate
        x = torch.cat((x1, x2), dim=1)
        if self.use_dropout:
            x = self.dropout(x)
#         print(x.shape)
        # Fully connected layer
        x = self.fc(x)
        return x

class DeepSleepNet(DeepFeatureNet):
    def __init__(self, input_dims, n_classes, seq_length, n_rnn_layers, use_dropout_feature, use_dropout_sequence):
        super(DeepSleepNet, self).__init__(input_dims, n_classes, use_dropout_feature)
        
        self.seq_length = seq_length
        self.n_rnn_layers = n_rnn_layers
        self.use_dropout_sequence = use_dropout_sequence
        
        hidden_size = 512
        self.lstm = nn.LSTM(input_size=1024, hidden_size=hidden_size, num_layers=n_rnn_layers, bidirectional=True, batch_first=True, dropout=0.5 if use_dropout_sequence else 0)
        
        self.fc_sequence = nn.Linear(hidden_size * 2, 1024)
        self.fc_final = nn.Linear(1024, n_classes)

    def forward(self, x):
        batch_size = x.size(0)
#         print(batch_size)
        # Reshape input for CNN
#         x = x.view(batch_size * self.seq_length, 1, -1)
        x = super(DeepSleepNet, self).forward(x)
#         print(x.shape)
        
        
        # LSTM
        x_lstm, _ = self.lstm(x)
#         print(x.shape)
        
        # Fully connected sequence
        x_lstm = F.relu(x_lstm)
        x_all = x_lstm+x
        if self.use_dropout_sequence:
            x_all = self.dropout(x_all)
        
        # Final layer
        x = self.fc_final(x_all)
        
        return x
    
model = DeepSleepNet(input_dims=3000, n_classes=5, seq_length=30, n_rnn_layers=2, use_dropout_feature=True, use_dropout_sequence=True)