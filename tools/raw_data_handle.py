import re
import jieba
import torch
from collections import defaultdict


# 原始数据处理方法（保持不变）
def get_mail_text(mailPath):
    with open(mailPath, "r", encoding="gb2312", errors="ignore") as mail:
        mailTestList = [text for text in mail]
        XindexList = [mailTestList.index(i) for i in mailTestList if re.match("[a-zA-Z0-9]", i)]
        textBegin = int(XindexList[-2]) + 1
        text = "".join(mailTestList[textBegin:])
        chinese_chars = re.findall(r'[\u4e00-\u9fa5]', text)
        text = ''.join(chinese_chars)
        seg_list = jieba.cut(text, cut_all=False)
        return " ".join(seg_list)


def get_paths_labels():
    with open("../dataset/trec06c/full/index", "r", encoding="gb2312", errors="ignore") as f:
        targetList = [line.strip().split() for line in f if len(line.strip().split()) == 2]
    pathList = [path[1].replace("..", "../dataset/trec06c") for path in targetList]
    labelList = [label[0] for label in targetList]
    return pathList, labelList


def preprocess_email_text(text):
    chinese_chars = re.findall(r'[\u4e00-\u9fa5]', text)
    text = ''.join(chinese_chars)
    seg_list = jieba.cut(text, cut_all=False)
    preprocessed_text = " ".join(seg_list)
    return preprocessed_text


def transform_label(labelList):
    return [1 if label == "spam" else 0 for label in labelList]


def build_vocab_and_sequences(content_list):
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


def pad_sequences(sequences, maxlen):
    padded_sequences = []
    for seq in sequences:
        if len(seq) > maxlen:
            padded_seq = seq[:maxlen]
        else:
            padded_seq = seq + [0] * (maxlen - len(seq))
        padded_sequences.append(padded_seq)
    return torch.tensor(padded_sequences, dtype=torch.long)


def prepare_data_for_cnn(content_list, labels, max_sequence_length):
    print("prepare_data_for_cnn 执行中...")
    sequences, word_index = build_vocab_and_sequences(content_list)
    X_cnn = pad_sequences(sequences, max_sequence_length)
    y_cnn = torch.tensor(labels, dtype=torch.long)
    return X_cnn, y_cnn, word_index
