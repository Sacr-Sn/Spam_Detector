import torch
import torch.nn as nn


class SpamLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout=0.5):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size+1, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0  # 仅在多层时使用dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, temperature=1.0):
        # text shape: [batch_size, seq_len]
        embedded = self.embedding(text)  # [batch_size, seq_len, emb_dim]

        # LSTM处理
        output, (hidden, _) = self.lstm(embedded)  # output: [batch_size, seq_len, hidden_dim]

        # 取最后一个时间步的隐藏状态
        last_hidden = output[:, -1, :]  # [batch_size, hidden_dim]

        # 应用dropout
        last_hidden = self.dropout(last_hidden)

        # 输出层
        logits = self.fc(last_hidden)  # [batch_size, output_dim]
        # print("logits:", logits)
        # print("logits shape:", logits.shape)
        return logits / temperature

    def save_model(self, vocab_size, max_sequence_length, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'vocab_size': self.embedding.num_embeddings,
                'embedding_dim': self.embedding.embedding_dim,
                'hidden_dim': 128,
                'output_dim': 1,  # 表示二分类
                'n_layers': 2,
                'dropout': 0.3
            }
        }, path)

    @classmethod
    def load_model(cls, path):
        checkpoint = torch.load(path, weights_only=True)
        model = cls(**checkpoint['config'])

        # =============
        # 加载预训练权重
        pretrained_weights = torch.load(path, weights_only=True)

        # 获取当前模型的词汇表大小
        current_vocab_size = 144559

        # 动态调整预训练权重的形状
        if 'embedding.weight' in pretrained_weights:
            pretrained_embedding = pretrained_weights['embedding.weight']
            new_embedding = torch.zeros((current_vocab_size, pretrained_embedding.shape[1]))
            new_embedding[:pretrained_embedding.shape[0], :] = pretrained_embedding
            pretrained_weights['embedding.weight'] = new_embedding
        model.load_state_dict(pretrained_weights, strict=False)
        # =============

        # model.load_state_dict(checkpoint['model_state_dict'])
        return model
