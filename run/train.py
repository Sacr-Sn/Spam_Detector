import os

import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

from net.SpamLSTM import SpamLSTM
from net.TextCNN import TextCNN
from tools.data_cache import *
from tools.raw_data_handle import *


# 训练（并同步调用验证）
def train(model_cate, max_sequence_length, embedding_dim,
          epochs, batch_size, num_filters, kernel_size, device,
          w2v_model_path, data_cache_dir, all_model_dir):
    # 加载或处理数据
    path_list, label_list, content_list = load_or_process_data(data_cache_dir=data_cache_dir)
    label_list = transform_label(label_list)

    # 训练或加载Word2Vec模型
    if w2v_model_path.exists():
        word2vec_model = Word2Vec.load(str(w2v_model_path))
        print("加载预训练的Word2Vec模型")
    else:
        sentences = [jieba.lcut(text) for text in content_list]
        word2vec_model = Word2Vec(
            sentences,
            vector_size=embedding_dim,
            window=5,
            min_count=1,
            workers=4,
            sg=1,
            epochs=10
        )
        word2vec_model.save(str(w2v_model_path))
        print("Word2Vec模型训练完成")

    # 准备CNN数据
    sequences, vocab = build_vocab_and_sequences(content_list)
    X_cnn = pad_sequences(sequences, max_sequence_length)
    y_cnn = torch.tensor(label_list, dtype=torch.long)

    # 准备嵌入矩阵
    embedding_matrix = np.zeros((len(vocab) + 1, embedding_dim))
    for word, i in vocab.items():
        if word in word2vec_model.wv:
            embedding_matrix[i] = word2vec_model.wv[word]

    # 训练模型
    # 三种数据集：
    #   训练集：用于模型参数的训练和优化。每轮
    #   验证集：用于超参数调优和模型选择，不参与参数训练。每轮
    #   测试集：最终评估模型在真实场景中的表现，模拟未知数据.仅在最终阶段使用一次（避免数据泄漏）
    # 第一次分割，分离测试集（20%）
    # 这是随机种子（random seed），用于确保每次运行代码时数据划分的结果是相同的。
    # 指定了测试集的比例
    X_train, X_test, y_train, y_test = train_test_split(X_cnn, y_cnn, test_size=0.2, random_state=42)
    # 第二次分割，训练集（75%）和验证集（25%）
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    print(f"训练集：{len(X_train)}条，验证集：{len(X_val)}条，测试集：{len(X_test)}条")

    # 训练集
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )
    # 验证集
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size  # 每批样本数：2048
    )

    # TODO 初始化模型(根据model_cate)
    model = None
    if model_cate == "TextCNN":
        model = TextCNN(
            vocab_size=len(vocab),
            embedding_dim=embedding_dim,
            num_filters=num_filters,
            kernel_size=kernel_size,
            max_sequence_length=max_sequence_length,
            embedding_matrix=embedding_matrix
        ).to(device)
    elif model_cate == "SpamLSTM":
        model = SpamLSTM(
            vocab_size=len(vocab),
            embedding_dim=embedding_dim,
            hidden_dim=128,
            output_dim=2,  # 二分类
            n_layers=2,
            dropout=0.3
        )

    train_model(model=model, model_cate=model_cate, device=device, epochs=epochs,
                vocab=vocab, max_sequence_length=max_sequence_length, batch_size=batch_size,
                all_model_dir=all_model_dir, train_loader=train_loader, val_loader=val_loader)


# 训练模型
def train_model(model, model_cate, device, epochs, vocab, max_sequence_length, batch_size,
                all_model_dir, train_loader, val_loader):
    model.to(device)
    class_weights = torch.tensor([1.0, 0.5], device=device)  # 调整权重
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs[:, :max_sequence_length]  # 确保不超过最大长度
            # print("inputs shape:", inputs.shape)  # 检查输入形状

            optimizer.zero_grad()
            outputs = model(inputs)
            # print("outputs shape:", outputs.shape)  # 检查模型输出形状

            labels = labels.long()  # 确保标签是整数
            # print("labels shape:", labels.shape)
            # print("labels type:", labels.dtype)
            loss = criterion(outputs, labels)
            # print(f"labels: {labels}")
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            # print(predicted)  # 本批预测结果
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        train_loss = total_loss / len(train_loader)

        # 验证集评估
        val_loss, val_acc = evaluate_model(model=model, model_cate=model_cate, device=device, vocab=vocab,
                                           max_sequence_length=max_sequence_length, batch_size=batch_size,
                                           all_model_dir=all_model_dir, val_loader=val_loader,
                                           epoch=epoch)

        print(f'Epoch {epoch + 1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')


def evaluate_model(model, model_cate, device, vocab, max_sequence_length, batch_size,
                   all_model_dir, val_loader, epoch):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs[:, :max_sequence_length]  # 确保不超过最大长度
            outputs = model(inputs)

            labels = labels.long()  # 确保标签是整数
            # print("labels shape:", labels.shape)
            # print("labels type:", labels.dtype)
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

    model_name_pre = None
    if model_cate == "TextCNN":
        model_name_pre = "cnn"
    elif model_cate == "SpamLSTM":
        model_name_pre = "lstm"

    # 保存模型(每5轮保存一次)
    if (epoch+1) % 5 == 0:
        model.save_model(len(vocab), max_sequence_length,
                         os.path.join(all_model_dir, f"{model_name_pre}_{epoch + 1}.pth"))

    return total_loss / len(val_loader), 100 * correct / total
