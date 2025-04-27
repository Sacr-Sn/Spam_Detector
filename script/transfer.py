# -*- coding: utf-8 -*-
"""
邮件数据处理工具
功能：
1. 自动检测文件编码
2. 生成Spam/Ham分类语料
3. 生成Word2Vec训练语料
"""
import os
import sys
import jieba
import chardet


def check_contain_chinese(check_str):
    """检查字符串是否包含中文"""
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False


def detect_file_encoding(file_path):
    """检测文件编码"""
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)  # 读取前10000字节检测编码
    result = chardet.detect(raw_data)
    return result['encoding'] or 'gbk'  # 默认回退到gbk


def load_label_files(label_file):
    """加载标签文件"""
    label_dict = {}
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split("..")
            if len(parts) >= 2:
                label_dict[parts[1].strip()] = parts[0].strip()
    return label_dict


def load_stopwords(stop_word_path):
    """加载停用词表(自动处理编码)"""
    stopwords = set()
    with open(stop_word_path, 'rb') as f:
        encoding = detect_file_encoding(stop_word_path)
        f.seek(0)
        content = f.read().decode(encoding, errors='ignore')
        for word in content.splitlines():
            stopwords.add(word.strip())
    return stopwords


def process_file(file_path, stopwords):
    """处理单个文件返回分词结果"""
    encoding = detect_file_encoding(file_path)
    temp_list = []

    try:
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            for line in f:
                line_clean = line.strip()
                if not line_clean:  # 跳过空行
                    continue
                if not check_contain_chinese(line_clean):
                    continue
                seg_list = jieba.cut(line_clean, cut_all=False)
                temp_list.extend([word for word in seg_list if word not in stopwords])
    except Exception as e:
        print(f"处理文件 {file_path} 时发生错误: {str(e)}")

    return temp_list


def generate_corpus(input_dir, label_dict, stopwords, output_spam, output_ham):
    """生成分类语料(spam.txt/ham.txt)"""
    with open(output_spam, 'wb') as spam_f, open(output_ham, 'wb') as ham_f:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_name = os.path.basename(file_path)

                # 获取标签
                label = label_dict.get(file_name, "unk")
                if label not in ("spam", "ham"):
                    continue

                # 处理文件
                words = process_file(file_path, stopwords)
                if not words:
                    continue

                # 写入结果
                line = ' '.join(words) + '\n'
                if label == "spam":
                    spam_f.write(line.encode('utf-8', 'ignore'))
                else:
                    ham_f.write(line.encode('utf-8', 'ignore'))


def generate_word2vec_corpus(input_dir, stopwords, output_path):
    """生成Word2Vec语料(word2vec.txt)"""
    with open(output_path, 'wb') as out_f:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                file_path = os.path.join(root, file)
                words = process_file(file_path, stopwords)
                if words:
                    line = ' '.join(words) + '\n'
                    out_f.write(line.encode('utf-8', 'ignore'))


if __name__ == "__main__":
    # 参数设置
    if len(sys.argv) != 3:
        print("使用方法: python transfer.py <邮件目录> <标签文件>")
        sys.exit(1)

    EMAIL_DIR = sys.argv[1]
    LABEL_FILE = sys.argv[2]
    STOPWORD_FILE = "../dataset/ret_file/stop_words.txt"
    SPAM_FILE = "../dataset/ret_file/spam.txt"
    HAM_FILE = "../dataset/ret_file/ham.txt"
    WORD2VEC_FILE = "../dataset/ret_file/word2vec.txt"

    # 初始化组件
    print("正在加载停用词表...")
    stopwords = load_stopwords(STOPWORD_FILE)

    print("正在加载邮件标签...")
    label_dict = load_label_files(LABEL_FILE)

    # 生成语料
    print("生成分类语料...")
    generate_corpus(EMAIL_DIR, label_dict, stopwords, SPAM_FILE, HAM_FILE)

    print("生成Word2Vec语料...")
    generate_word2vec_corpus(EMAIL_DIR, stopwords, WORD2VEC_FILE)

    print("处理完成！")
    print(f"生成文件: {SPAM_FILE}, {HAM_FILE}, {WORD2VEC_FILE}")