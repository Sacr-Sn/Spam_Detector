import seaborn as sb
from matplotlib import pyplot as plt

''' ========================= 工具函数 ========================= '''


# 绘制混淆矩阵
def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(8, 6))
    sb.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Spam', 'Spam'],
               yticklabels=['Non-Spam', 'Spam'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
