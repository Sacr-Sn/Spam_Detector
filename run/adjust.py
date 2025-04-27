import os

import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # 导入F模块

from net.SpamLSTM import SpamLSTM
from net.TextCNN import TextCNN
from tools.data_cache import *
from tools.raw_data_handle import *

''' 微调训练 '''

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 模型路径 TODO
model_path = "models/lstm/lstm_10.pth"

# 微调后的名称
model_name_pre = "lstm_wt"
# 模型
# model = TextCNN.load_model(model_path).to(device)
model = SpamLSTM.load_model(model_path).to(device)

# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 动态温度函数（线性衰减）
def get_temperature(epoch, max_epochs, initial_temp, final_temp):
    return initial_temp - (initial_temp - final_temp) * (epoch / max_epochs)


# 动态权重函数（线性变化）
def get_weight(epoch, max_epochs, initial_weight, final_weight):
    return initial_weight - (initial_weight - final_weight) * (epoch / max_epochs)


# 损失函数计算
def compute_loss(outputs, labels, temperature=1.0, weight=0.0):
    # 交叉熵损失（硬标签）
    class_weights = torch.tensor([1.0, 0.5], device=device)  # 调整权重
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights)
    loss_ce = criterion_ce(outputs, labels)

    # 如果权重大于 0，计算 KL 散度损失（软标签）
    if weight > 0:
        with torch.no_grad():
            soft_labels = torch.softmax(outputs.detach() / temperature, dim=1)  # 使用 torch.softmax
        criterion_kl = nn.KLDivLoss(reduction="batchmean")
        # loss_kl = criterion_kl(torch.softmax(outputs / temperature, dim=1).log(), soft_labels)
        # 使用 log_softmax 替代 softmax().log()
        loss_kl = F.kl_div(
            F.log_softmax(outputs / temperature, dim=1),
            soft_labels,
            reduction='batchmean'
        ) * (temperature ** 2)  # 温度缩放
    else:
        loss_kl = 0.0

    # 总损失
    loss = loss_ce + weight * loss_kl
    return loss


# 训练（并同步调用验证）
def adjust(max_sequence_length, embedding_dim,
           epochs, batch_size,
           w2v_model_path, data_cache_dir, wt_model_dir):
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

    finetune_with_kd(epochs=epochs,
                     vocab=vocab, max_sequence_length=max_sequence_length,
                     wt_model_dir=wt_model_dir, train_loader=train_loader, val_loader=val_loader)


# 微调阶段（使用知识自蒸馏）
def finetune_with_kd(epochs, vocab, max_sequence_length,
                     wt_model_dir, train_loader, val_loader,
                     initial_temp=3.0, final_temp=1.0, initial_weight=0.5, final_weight=0.1):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    loss = None
    for epoch in range(epochs):
        # 计算当前温度和权重
        temperature = get_temperature(epoch, epochs, initial_temp, final_temp)
        weight = get_weight(epoch, epochs, initial_weight, final_weight)
        total_loss = 0
        for texts, labels in train_loader:
            optimizer.zero_grad()
            # 计算模型输出（加入温度参数）
            texts = texts.to(device)
            labels = labels.to(device)
            outputs = model(texts, temperature=temperature)
            # 计算损失（使用交叉熵损失和 KL 散度损失）
            loss = compute_loss(outputs, labels, temperature=temperature, weight=weight)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()  # 累加损失
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        train_loss = total_loss / len(train_loader)

        # 验证集评估
        # 验证集评估
        val_loss, val_acc = evaluate_model(vocab=vocab,
                                           max_sequence_length=max_sequence_length,
                                           wt_model_dir=wt_model_dir, val_loader=val_loader,
                                           epoch=epoch, save=(epoch+1 == epochs))
        model.train()  # 评估后切换回训练模式
        print(f'Epoch {epoch + 1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(
            f"Finetune Epoch {epoch + 1}/{epochs}, Temperature: {temperature:.2f}, Weight: {weight:.2f}, Loss: {loss.item()}")


def evaluate_model(vocab, max_sequence_length, wt_model_dir, val_loader, epoch, save):
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
            outputs = model(inputs)
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

    # 保存模型（只保留最后一次）
    if save:
        model.save_model(len(vocab), max_sequence_length,
                         os.path.join(wt_model_dir, f"{model_name_pre}_{epoch + 1}.pth"))

    return total_loss / len(val_loader), 100 * correct / total
