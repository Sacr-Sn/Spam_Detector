import os
import pickle
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm
import jieba
from gensim.models import Word2Vec
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import seaborn as sb
from matplotlib import pyplot as plt
from collections import defaultdict


class ChineseEmailClassification():
    def __init__(self):
        # 模型参数
        self.vocab = None
        self.word2vec_model = None
        self.model = None
        self.max_sequence_length = 300  # 最大序列长度
        self.embedding_dim = 64  # 词向量维度
        self.epochs = 20  # 训练轮数
        self.batch_size = 2048  # 每批样本数
        self.num_filters = 64  # 卷积核数目
        self.kernel_size = 5  # 卷积核尺寸
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 路径设置
        self.data_cache_dir = Path("../data/data_cache")
        self.model_dir = Path("../models")
        self.w2v_model_path = self.model_dir / "word2vec" / "word2vec.model"
        self.cnn_model_dir = self.model_dir / "cnn"

        # 创建目录（如果不存在）
        self.data_cache_dir.mkdir(parents=True, exist_ok=True)
        self.w2v_model_path.parent.mkdir(parents=True, exist_ok=True)
        self.cnn_model_dir.mkdir(parents=True, exist_ok=True)

    # 数据缓存相关方法
    def _get_cache_path(self, name):
        return self.data_cache_dir / f"{name}.pkl"

    def load_or_process_data(self, force_update=False):
        """智能加载或处理数据"""
        cache_files = {
            'path_list': self._get_cache_path('path_list'),
            'label_list': self._get_cache_path('label_list'),
            'content_list': self._get_cache_path('content_list')
        }

        # 检查是否需要重新处理
        if not force_update and all(f.exists() for f in cache_files.values()):
            print("加载缓存数据...")
            with open(cache_files['path_list'], 'rb') as f:
                path_list = pickle.load(f)
            with open(cache_files['label_list'], 'rb') as f:
                label_list = pickle.load(f)
            with open(cache_files['content_list'], 'rb') as f:
                content_list = pickle.load(f)
        else:
            print("处理原始数据...")
            path_list, label_list = self.get_paths_labels()
            content_list = self._process_content(path_list)

            # 保存缓存
            with open(cache_files['path_list'], 'wb') as f:
                pickle.dump(path_list, f)
            with open(cache_files['label_list'], 'wb') as f:
                pickle.dump(label_list, f)
            with open(cache_files['content_list'], 'wb') as f:
                pickle.dump(content_list, f)

        return path_list, label_list, content_list

    def _process_content(self, path_list):
        """处理邮件内容（带进度条）"""
        content_list = []
        for path in tqdm(path_list, desc="Processing Emails"):
            try:
                content_list.append(self.get_mail_text(path))
            except Exception as e:
                print(f"Error processing {path}: {str(e)}")
                content_list.append("")
        return content_list

    # 原始数据处理方法（保持不变）
    def get_mail_text(self, mailPath):
        with open(mailPath, "r", encoding="gb2312", errors="ignore") as mail:
            mailTestList = [text for text in mail]
            XindexList = [mailTestList.index(i) for i in mailTestList if re.match("[a-zA-Z0-9]", i)]
            textBegin = int(XindexList[-2]) + 1
            text = "".join(mailTestList[textBegin:])
            chinese_chars = re.findall(r'[\u4e00-\u9fa5]', text)
            text = ''.join(chinese_chars)
            seg_list = jieba.cut(text, cut_all=False)
            return " ".join(seg_list)

    def get_paths_labels(self):
        with open("../dataset/trec06c/full/index", "r", encoding="gb2312", errors="ignore") as f:
            targetList = [line.strip().split() for line in f if len(line.strip().split()) == 2]
        pathList = [path[1].replace("..", "../dataset/trec06c") for path in targetList]
        labelList = [label[0] for label in targetList]
        return pathList, labelList

    def preprocess_email_text(self, text):
        chinese_chars = re.findall(r'[\u4e00-\u9fa5]', text)
        text = ''.join(chinese_chars)
        seg_list = jieba.cut(text, cut_all=False)
        preprocessed_text = " ".join(seg_list)
        return preprocessed_text

    def transform_label(self, labelList):
        return [1 if label == "spam" else 0 for label in labelList]

    def build_vocab_and_sequences(self, content_list):
        word_counts = defaultdict(int)
        for text in content_list:
            for word in text.split():
                word_counts[word] += 1

        # 创建词汇表
        vocab = {word: idx + 1 for idx, (word, count) in enumerate(word_counts.items())}

        # 将文本转换为序列
        sequences = []
        for text in content_list:
            sequence = [vocab[word] for word in text.split() if word in vocab]
            sequences.append(sequence)

        return sequences, vocab

    def pad_sequences(self, sequences, maxlen):
        padded_sequences = []
        for seq in sequences:
            if len(seq) > maxlen:
                padded_seq = seq[:maxlen]
            else:
                padded_seq = seq + [0] * (maxlen - len(seq))
            padded_sequences.append(padded_seq)
        return torch.tensor(padded_sequences, dtype=torch.long)

    def prepare_data_for_cnn(self, content_list, labels):
        print("prepare_data_for_cnn 执行中...")
        sequences, word_index = self.build_vocab_and_sequences(content_list)
        X_cnn = self.pad_sequences(sequences, self.max_sequence_length)
        y_cnn = torch.tensor(labels, dtype=torch.long)
        return X_cnn, y_cnn, word_index

    class EmailDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    # 模型相关方法（保持不变）
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

        def forward(self, x):
            x = self.embedding(x).permute(0, 2, 1)
            x = self.relu(self.conv(x))
            x = torch.max(x, dim=2)[0]
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            return self.fc2(x)

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
            checkpoint = torch.load(path)
            model = cls(**checkpoint['config'])
            model.load_state_dict(checkpoint['model_state_dict'])
            return model

    def train_model(self, train_loader, test_loader):
        self.model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                print(predicted)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_acc = 100 * correct / total
            train_loss = total_loss / len(train_loader)

            # 验证集评估
            val_loss, val_acc = self.evaluate_model(test_loader, epoch)

            print(f'Epoch {epoch + 1}/{self.epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    # 评估模型
    '''
        功能：计算验证集表现
        输出：
            混淆矩阵（通过plot_confusion_matrix可视化）
            分类报告（precision/recall/F1）
            准确率和平均损失
    '''

    def evaluate_model(self, test_loader, epoch):
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算混淆矩阵和分类报告
        conf_matrix = confusion_matrix(all_labels, all_preds)
        class_report = classification_report(all_labels, all_preds)

        # 绘制混淆矩阵
        # self.plot_confusion_matrix(conf_matrix)

        print("Confusion Matrix:\n", conf_matrix)
        print("\nClassification Report:\n", class_report)

        # 保存模型
        self.model.save_model(len(self.vocab), self.max_sequence_length,
                              os.path.join(self.cnn_model_dir, f"cnn_{epoch + 1}.pth"))

        return total_loss / len(test_loader), 100 * correct / total

    def predict(self, email_texts):
        preprocessed_texts = [self.preprocess_email_text(text) for text in email_texts]

        # 转换为序列
        sequences = []
        for text in preprocessed_texts:
            sequence = [self.vocab[word] for word in text.split() if word in self.vocab]
            sequences.append(sequence)

        # 填充序列
        X_pred = self.pad_sequences(sequences, self.max_sequence_length)
        X_pred = X_pred.to(self.device)

        # 预测
        self.model = self.TextCNN.load_model("../models/cnn/cnn_10.pth")
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_pred)
            _, predictions = torch.max(outputs, 1)

        return predictions.cpu().numpy().tolist()

    # 绘制混淆矩阵(混淆矩阵可视化)
    '''
        使用：seaborn.heatmap绘制
        坐标轴：
            x：预测标签
            y：真实标签
    '''

    def plot_confusion_matrix(self, conf_matrix):
        plt.figure(figsize=(8, 6))
        sb.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Spam', 'Spam'],
                   yticklabels=['Non-Spam', 'Spam'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()

    def train(self):
        # 加载或处理数据
        path_list, label_list, content_list = self.load_or_process_data()
        label_list = self.transform_label(label_list)

        # 训练或加载Word2Vec模型
        if self.w2v_model_path.exists():
            self.word2vec_model = Word2Vec.load(str(self.w2v_model_path))
            print("加载预训练的Word2Vec模型")
        else:
            sentences = [jieba.lcut(text) for text in content_list]
            self.word2vec_model = Word2Vec(
                sentences,
                vector_size=self.embedding_dim,
                window=5,
                min_count=1,
                workers=4,
                sg=1,
                epochs=10
            )
            self.word2vec_model.save(str(self.w2v_model_path))
            print("Word2Vec模型训练完成")

        # 准备CNN数据
        sequences, self.vocab = self.build_vocab_and_sequences(content_list)
        X_cnn = self.pad_sequences(sequences, self.max_sequence_length)
        y_cnn = torch.tensor(label_list, dtype=torch.long)

        # 准备嵌入矩阵
        embedding_matrix = np.zeros((len(self.vocab) + 1, self.embedding_dim))
        for word, i in self.vocab.items():
            if word in self.word2vec_model.wv:
                embedding_matrix[i] = self.word2vec_model.wv[word]

        # 初始化CNN模型
        self.model = self.TextCNN(
            vocab_size=len(self.vocab),
            embedding_dim=self.embedding_dim,
            num_filters=self.num_filters,
            kernel_size=self.kernel_size,
            max_sequence_length=self.max_sequence_length,
            embedding_matrix=embedding_matrix
        ).to(self.device)

        # 训练模型
        X_train, X_test, y_train, y_test = train_test_split(X_cnn, y_cnn, test_size=0.3, random_state=42)
        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size,
            shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(X_test, y_test),
            batch_size=self.batch_size  # 每批样本数：2048
        )

        self.train_model(train_loader, test_loader)

    def test(self):
        # 加载或处理数据
        path_list, label_list, content_list = self.load_or_process_data()
        label_list = self.transform_label(label_list)

        # 准备CNN数据
        sequences, self.vocab = self.build_vocab_and_sequences(content_list)

        # 测试预测
        test_emails = [
            """尊敬的贵公司(财务/经理)负责人您好！  
            我你深圳拿的海阳有限曲线（广州。东莞）等省市地震发财。  
            我司有人社会关系企业，因每日多出项少现有一部分学生  
            发票开不出来，烟雨江湖国税，一点含光万丈芒.地税.     
            运输.西安电子科技大学，点击就送，还可以根据数目大小来衡  
            量优惠的多少，希望导师捞捞我.一问三不知，拟录取通知欢迎合作。""",
            "你好，今天天气晴转多云，请注意防护",
            """希望这封邮件能找到你一切安好。感谢你一直的支持和友谊。
            最近天气转暖，我计划组织一次小型聚会，希望你能参加。我们可以分享一些美食，畅谈近况，共度愉快时光。请告诉我你的方便时间，期待和你见面。""",
            """在实际应用中，你可能需要检查模型的训练数据，尝试调整模型的架构或超参数，以及考虑更多的特征工程，
            以提高模型的性能和分类准确度。这可能包括调整词向量维度、卷积核大小等""",
            """啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊""",
            """你好""",
            """矩阵的秩是指非零子式的最高阶数""",
            """今天过得怎么样？""",
            """好久不见了，你今天好吗""",
            """打客服寄快递今飞凯达剪发卡说法卡拉季打卡飞机啊咖色激发颇大MV扩大没法v看老大妈刚发打卡没打，v"""
        ]

        predictions = self.predict(test_emails)
        print("Predictions:", predictions)
        for email, pred in zip(test_emails, predictions):
            print(f"Email: {email[:50]}... -> {'Spam' if pred == 1 else 'Non-Spam'}")


if __name__ == "__main__":
    classifier = ChineseEmailClassification()
    classifier.train()
    # classifier.test()
