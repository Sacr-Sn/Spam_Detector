import torch
import torch.nn as nn


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, kernel_size, max_sequence_length,
                 embedding_matrix=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            self.embedding.weight.requires_grad = False
        self.conv = nn.Conv1d(embedding_dim, num_filters, kernel_size)
        self.fc1 = nn.Linear(num_filters, 32)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, text, temperature=1.0):
        text = self.embedding(text).permute(0, 2, 1)
        text = self.relu(self.conv(text))
        text = torch.max(text, dim=2)[0]
        text = self.relu(self.fc1(text))
        text = self.dropout(text)
        return self.fc2(text) / temperature

    def save_model(self, vocab_size, max_sequence_length, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'vocab_size': vocab_size,
                'max_sequence_length': max_sequence_length,
                'embedding_dim': self.embedding.embedding_dim,
                'num_filters': self.conv.out_channels,
                'kernel_size': self.conv.kernel_size[0],
            }
        }, path)

    @classmethod
    def load_model(cls, path):
        checkpoint = torch.load(path, weights_only=True)
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
