import os

from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

from net.SpamLSTM import SpamLSTM
from net.TextCNN import TextCNN
from tools.data_cache import *
from tools.raw_data_handle import *


def test(model_cate, model_path, max_sequence_length, batch_size, device, data_cache_dir):
    # 加载或处理数据
    path_list, label_list, content_list = load_or_process_data(data_cache_dir=data_cache_dir)
    label_list = transform_label(label_list)

    # 准备CNN数据
    sequences, vocab = build_vocab_and_sequences(content_list)
    X_cnn = pad_sequences(sequences, max_sequence_length)
    y_cnn = torch.tensor(label_list, dtype=torch.long)

    # 指定了测试集的比例
    X_train, X_test, y_train, y_test = train_test_split(X_cnn, y_cnn, test_size=0.2, random_state=42)
    # 第二次分割，训练集（75%）和验证集（25%）
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    print(f"训练集：{len(X_train)}条，验证集：{len(X_val)}条，测试集：{len(X_test)}条")

    # 测试集
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size  # 每批样本数：2048
    )

    # 测试
    test_model(model_cate, model_path, test_loader, device)


def test_model(model_cate, model_path, test_loader, device):
    # 1. 模型加载安全检查
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在，请先训练模型")
    model = None
    # TODO
    if model_cate == "TextCNN":
        model = TextCNN.load_model(model_path)
    elif model_cate == "SpamLSTM":
        model = SpamLSTM.load_model(model_path)
    model.to(device)
    model.eval()

    # 2. 测试数据校验
    if len(test_loader.dataset) == 0:
        raise ValueError("测试数据集为空")

    # 3. 测试过程
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
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
    # 4. 计算指标
    test_loss = total_loss / len(test_loader)
    test_acc = 100 * correct / total
    conf_matrix = confusion_matrix(all_labels, all_preds)  # # 混淆矩阵
    class_report = classification_report(all_labels, all_preds, output_dict=True)  # 分类报告

    # 绘制混淆矩阵
    # self.plot_confusion_matrix(conf_matrix)

    # 5. 关键指标断言（示例）
    # assert test_acc > 95.0, f"测试准确率{test_acc:.2f}%低于阈值95%"
    # assert class_report['1']['recall'] > 0.95, "垃圾邮件召回率不足"

    # 6. 结果输出
    print(f"\n测试结果: Loss={test_loss:.4f}, Acc={test_acc:.2f}%")
    print("Confusion Matrix:\n", conf_matrix)
    print("\nClassification Report:", class_report)
    print(classification_report(all_labels, all_preds))

    return test_loss, test_acc
