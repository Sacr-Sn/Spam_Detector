import os.path

import jieba
import re
import numpy as np
from gensim.models import Word2Vec
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict


class ChineseEmailClassification():
    # 初始化模型参数和硬件设置
    def __init__(self):
        self.vocab = None
        self.word2vec_model = None
        self.model = None
        self.max_sequence_length = 300  # 最大序列长度
        self.embedding_dim = 64  # 词向量维度
        self.epochs = 10  # 训练次数
        self.batch_size = 2048  # 每批样本数
        self.num_filters = 64  # 卷积核数目
        self.kernel_size = 5  # 卷积核尺寸
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 路径参数
        self.w2v_model_dir = "models\\word2vec"
        os.makedirs(self.w2v_model_dir, exist_ok=True)  # 自动创建目录（如果不存在）
        self.w2v_model_path = os.path.join(self.w2v_model_dir, "word2vec.model")
        self.cnn_model_dir = "models\\cnn"

    """   数据预处理【得到完整的路径，将邮件信息转为0∪1】   """

    # 获取jieba分词后的文本，如" 我 爱 中文 ..."
    # 关键处理步骤：
    # 1. 使用GB2312编码读取邮件
    # 2. 定位正文起始位置(跳过邮件头)
    # 3. 提取中文字符
    # 4. 使用jieba分词
    # 返回格式："单词1 单词2 ..."
    def get_mail_text(self, mailPath):
        print("get_mail_text 执行中...")
        mail = open(mailPath, "r", encoding="gb2312", errors="ignore")
        mailTestList = [text for text in mail]
        # 去除文件开头部分
        XindexList = [mailTestList.index(i) for i in mailTestList if re.match("[a-zA-Z0-9]", i)]
        textBegin = int(XindexList[-2]) + 1
        text = "".join(mailTestList[textBegin:])
        # 匹配汉字
        chinese_chars = re.findall(r'[\u4e00-\u9fa5]', text)
        text = ''.join(chinese_chars)
        # jieba分词
        seg_list = jieba.cut(text, cut_all=False)
        text = " ".join(seg_list)
        return text

    # 通过index文件获取所有文件路径与标签值
    '''
        输入：TREC06C数据集索引文件
        处理：
            解析index文件中的路径和标签
            将相对路径转为绝对路径
        输出：
            pathList：所有邮件完整路径列表
            labelList：对应标签列表（"spam"/"ham"）
    '''

    def get_paths_labels(self):
        targets = open("../dataset/trec06c/full/index", "r", encoding="gb2312", errors="ignore")
        targetList = [t for t in targets]
        newTargetList = [target.split() for target in targetList if len(target.split()) == 2]
        pathList = [path[1].replace("..", "../dataset/trec06c") for path in newTargetList]  # 完整路径
        labelList = [label[0] for label in newTargetList]
        return pathList, labelList

    # 对准备识别的文本预处理
    '''
        输入：原始文本字符串
        处理：
            移除非中文字符
            jieba分词
        输出：同get_mail_text格式的分词结果
    '''

    def preprocess_email_text(self, text):
        chinese_chars = re.findall(r'[\u4e00-\u9fa5]', text)
        text = ''.join(chinese_chars)
        seg_list = jieba.cut(text, cut_all=False)
        preprocessed_text = " ".join(seg_list)
        return preprocessed_text

    # 标签转化，spam，ham 分别对应 1，0
    def transform_label(self, labelList):
        print("transform_label 执行中...")
        return [1 if label == "spam" else 0 for label in labelList]

    """                            数据向量化与搭建模型                        """

    # 构建词汇表和文本序列化
    '''
        输入：分词后的邮件列表 ["单词1 单词2", ...]
        处理：
            统计词频生成词汇表 {"单词": 索引}
            将文本转为数字序列 [[12,34,...], ...]
        输出：
            sequences：数字序列列表
            vocab：词汇表字典
    '''

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

    # 填充序列
    '''
        输入：
            sequences：不等长数字序列
            maxlen：填充长度
        处理：
            截断超过300的序列
            不足的补0
        输出：固定长度的Tensor [batch_size, 300]
    '''

    def pad_sequences(self, sequences, maxlen):
        padded_sequences = []
        for seq in sequences:
            if len(seq) > maxlen:
                padded_seq = seq[:maxlen]
            else:
                padded_seq = seq + [0] * (maxlen - len(seq))
            padded_sequences.append(padded_seq)
        return torch.tensor(padded_sequences, dtype=torch.long)

    # 准备数据
    '''
        流程：
            调用build_vocab_and_sequences
            调用pad_sequences
            转换标签为Tensor
        输出：
            X_cnn：填充后的特征矩阵
            y_cnn：标签Tensor
            word_index：词汇表
    '''

    def prepare_data_for_cnn(self, content_list, labels):
        print("prepare_data_for_cnn 执行中...")
        sequences, word_index = self.build_vocab_and_sequences(content_list)
        X_cnn = self.pad_sequences(sequences, self.max_sequence_length)
        y_cnn = torch.tensor(labels, dtype=torch.long)
        return X_cnn, y_cnn, word_index

    # 自定义Dataset类
    '''
        功能：封装数据供DataLoader使用
        关键方法：
            __getitem__：返回(特征序列, 标签)
            __len__：返回样本数
    '''

    class EmailDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    # 搭建CNN模型
    # 网络结构：Embedding -> Conv1d -> ReLU -> MaxPool -> FC1 -> Dropout -> FC2
    '''
        关键参数：
            embedding：加载预训练词向量
            conv1d：in_channels=emb_dim, out_channels=num_filters
            fc2：输出2维（spam/ham）
    '''

    class TextCNN(nn.Module):
        def __init__(self, vocab_size, embedding_dim, num_filters, kernel_size, max_sequence_length,
                     embedding_matrix=None):
            super().__init__()

            # 嵌入层
            self.embedding = nn.Embedding(vocab_size + 1, embedding_dim)
            if embedding_matrix is not None:
                self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
                self.embedding.weight.requires_grad = False  # 冻结嵌入层

            # 卷积层
            self.conv = nn.Conv1d(embedding_dim, num_filters, kernel_size)

            # 全连接层
            self.fc1 = nn.Linear(num_filters, 32)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(32, 2)

            # 激活函数
            self.relu = nn.ReLU()

        def forward(self, x):
            # 嵌入层
            x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
            x = x.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_len] 适合1D卷积

            # 卷积层和池化
            x = self.conv(x)  # [batch_size, num_filters, seq_len - kernel_size + 1]
            x = self.relu(x)
            x = torch.max(x, dim=2)[0]  # 全局最大池化 [batch_size, num_filters]

            # 全连接层
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)

            return x

        # 保存模型
        def save_model(self, path):
            torch.save({
                'model_state_dict': self.state_dict(),  # 核心参数
                'vocab': self.vocab,  # 自定义数据
                'config': {  # 结构配置
                    'embedding_dim': self.embedding_dim,
                    'num_filters': self.num_filters,
                    'kernel_size': self.kernel_size
                    # 注意：max_sequence_length不需要保存，因其不影响模型结构
                }
            }, path)

        # 加载模型
        @classmethod
        def load_model(cls, path):
            checkpoint = torch.load(path)
            model = cls(**checkpoint['config'])  # 先用config初始化结构
            model.load_state_dict(checkpoint['model_state_dict'])  # 再加载参数
            model.vocab = checkpoint['vocab']  # 恢复自定义数据
            return model

    # 训练模型
    '''
        流程：
            设置交叉熵损失和Adam优化器
            每个epoch：
                前向传播计算损失
                反向传播更新参数
                计算训练集准确率
                在验证集评估
        输出：实时打印训练指标
    '''

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
        self.plot_confusion_matrix(conf_matrix)

        print("Confusion Matrix:\n", conf_matrix)
        print("\nClassification Report:\n", class_report)

        # 保存模型
        self.model.save_model(os.path.join(self.cnn_model_dir, f"cnn_{epoch+1}.pth"))

        return total_loss / len(test_loader), 100 * correct / total

    # 预测函数
    '''
        流程：
            预处理输入文本
            转换为数字序列
            填充序列
            模型预测
        输出：预测标签列表 [0, 1, ...]
    '''

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

    # 主函数
    '''
        执行顺序：
            加载数据和标签
            训练Word2Vec词向量
            准备CNN输入数据
            初始化模型
            训练和评估
            测试预测
    '''

    def main(self):
        # 数据加载
        path_list, label_list = self.get_paths_labels()
        print("数据加载完成")

        # 获得分词文本
        content_list = [self.get_mail_text(file_path) for file_path in path_list]
        print("分词文本获取完成")

        # 转换标签
        label_list = self.transform_label(label_list)
        print("标签转换完成")

        # 训练word2vec模型
        if os.path.exists(self.w2v_model_path):
            # 加载已有模型
            self.word2vec_model = Word2Vec.load(self.w2v_model_path)
            print(f"从 {self.w2v_model_path} 加载模型成功")
        else:
            # 首次训练并保存
            sentences = [jieba.lcut(text) for text in content_list]
            self.word2vec_model = Word2Vec(sentences, vector_size=self.embedding_dim, window=5, min_count=1, workers=4)
            self.word2vec_model.save(self.w2v_model_path)
            print(f"word2vec模型训练完成，并保存至：{self.w2v_model_path}")

        # 准备数据
        X_cnn, y_cnn, word_index = self.prepare_data_for_cnn(content_list, label_list)
        self.vocab = word_index
        print("数据准备完成")

        # 准备嵌入矩阵
        embedding_matrix = np.zeros((len(word_index) + 1, self.embedding_dim))
        for word, i in word_index.items():
            if word in self.word2vec_model.wv:
                embedding_matrix[i] = self.word2vec_model.wv[word]
        print("矩阵嵌入完成")

        # 搭建CNN模型
        self.model = self.TextCNN(
            vocab_size=len(word_index),
            embedding_dim=self.embedding_dim,
            num_filters=self.num_filters,
            kernel_size=self.kernel_size,
            max_sequence_length=self.max_sequence_length,
            embedding_matrix=embedding_matrix
        )
        print("CNN模型搭建完成")

        # 划分训练集与验证集
        X_train, X_test, y_train, y_test = train_test_split(
            X_cnn, y_cnn, test_size=0.3, random_state=42)
        print("数据集划分完成")

        # 创建DataLoader
        train_dataset = self.EmailDataset(X_train, y_train)
        test_dataset = self.EmailDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        print("dataloader搭建完成")

        # 训练模型
        self.train_model(train_loader, test_loader)
        print("模型训练完成")

        # 测试预测
        test_emails = [
            """尊敬的贵公司(财务/经理)负责人您好！  
            我是深圳金海实业有限公司（广州。东莞）等省市有分公司。  
            我司有良好的社会关系和实力，因每月进项多出项少现有一部分  
            发票可优惠对外代开税率较低，增值税发票为5%其它国税.地税.     
            运输.广告等普通发票为1.5%的税点，还可以根据数目大小来衡  
            量优惠的多少，希望贵公司.商家等来电商谈欢迎合作。""",
            "港澳台代表团参加亚运会",
            """希望这封邮件能找到你一切安好。感谢你一直的支持和友谊。
            最近天气转暖，我计划组织一次小型聚会，希望你能参加。我们可以分享一些美食，畅谈近况，共度愉快时光。请告诉我你的方便时间，期待和你见面。""",
            """在实际应用中，你可能需要检查模型的训练数据，尝试调整模型的架构或超参数，以及考虑更多的特征工程，
            以提高模型的性能和分类准确度。这可能包括调整词向量维度、卷积核大小等"""
        ]

        predictions = self.predict(test_emails)
        print("Predictions:", predictions)
        for email, pred in zip(test_emails, predictions):
            print(f"Email: {email[:50]}... -> {'Spam' if pred == 1 else 'Non-Spam'}")


# 运行主程序
if __name__ == "__main__":
    classifier = ChineseEmailClassification()
    classifier.main()
