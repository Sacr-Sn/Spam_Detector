import os

from net.SpamLSTM import SpamLSTM
from net.TextCNN import TextCNN
from tools.data_cache import *
from tools.raw_data_handle import *


def pred(email_texts, model_cate, model_path, data_cache_dir, max_sequence_length,
         device):
    # 加载或处理数据
    path_list, label_list, content_list = load_or_process_data(data_cache_dir)
    label_list = transform_label(label_list)

    # 准备CNN数据
    sequences, vocab = build_vocab_and_sequences(content_list)

    # 测试预测
    predictions = predict(model_cate, model_path, email_texts, vocab,
                          max_sequence_length, device)
    print("Predictions:", predictions)
    for email, prediction in zip(email_texts, predictions):
        print(f"Email: {email[:50]}... -> {'Spam' if prediction == 1 else 'Non-Spam'}")


def predict(model_cate, model_path, email_texts, vocab, max_sequence_length, device):
    preprocessed_texts = [preprocess_email_text(text) for text in email_texts]

    # 转换为序列
    sequences = []
    for text in preprocessed_texts:
        sequence = [vocab[word] for word in text.split() if word in vocab]
        sequences.append(sequence)

    # 填充序列
    X_pred = pad_sequences(sequences, max_sequence_length)
    X_pred = X_pred.to(device)

    # 预测
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
    with torch.no_grad():
        outputs = model(X_pred)
        _, predictions = torch.max(outputs, 1)

    return predictions.cpu().numpy().tolist()
