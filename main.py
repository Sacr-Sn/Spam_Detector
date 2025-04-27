from pathlib import Path
import torch

from run.adjust import adjust
from run.predict import pred
from run.test import test
from run.train import train

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

''' ========================= 参数定义 ========================= '''

# 模型参数
vocab = None
word2vec_model = None
model = None
max_sequence_length = 128  # 最大序列长度(lstm:128  cnn:300)
embedding_dim = 64  # 词向量维度
epochs = 20  # 训练轮数
batch_size = 128  # 每批样本数（lstm:128  cnn:2048）
num_filters = 64  # 卷积核数目
kernel_size = 5  # 卷积核尺寸
hidden_dim = 128
output_dim = 2  # 表示二分类
n_layers = 2
dropout = 0.3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# 模型类别，根据类别名称定义具体模型
# model_cate = "TextCNN"
model_cate = "SpamLSTM"

# 路径设置
model_dir = Path("./models")
data_cache_dir = Path("./data/data_cache")
w2v_model_path = model_dir / "word2vec" / "word2vec.model"
all_model_dir = model_dir / "lstm"  # 模型存储全路径
wt_model_dir = model_dir / "lstm_wt"  # 微调后的模型存储路径

# 被测试的模型路径
test_model_path = "models/lstm_wt/lstm_wt_10.pth"

# 创建目录（如果不存在）
data_cache_dir.mkdir(parents=True, exist_ok=True)
w2v_model_path.parent.mkdir(parents=True, exist_ok=True)
all_model_dir.mkdir(parents=True, exist_ok=True)  # 模型存储全路径
wt_model_dir.mkdir(parents=True, exist_ok=True)


def to_train():
    train(model_cate=model_cate, max_sequence_length=max_sequence_length,
          embedding_dim=embedding_dim, epochs=epochs, batch_size=batch_size,
          num_filters=num_filters, kernel_size=kernel_size, device=device,
          w2v_model_path=w2v_model_path, data_cache_dir=data_cache_dir,
          all_model_dir=all_model_dir)


def to_test():
    test(model_cate=model_cate, model_path=test_model_path,
         max_sequence_length=max_sequence_length, batch_size=batch_size,
         device=device, data_cache_dir=data_cache_dir)


def to_predict(model_path):
    email_texts = [
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
    pred(email_texts=email_texts, model_cate=model_cate, model_path=model_path,
         data_cache_dir=data_cache_dir, max_sequence_length=max_sequence_length,
         device=device)


def to_adjust():
    adjust(max_sequence_length=max_sequence_length, embedding_dim=embedding_dim,
           epochs=10, batch_size=batch_size,
           w2v_model_path=w2v_model_path, data_cache_dir=data_cache_dir,
           wt_model_dir=wt_model_dir)


if __name__ == '__main__':
    to_train()
    # to_test()
    # to_adjust()
    # to_predict("models/cnn_wt/cnn_wt_10.pth")
