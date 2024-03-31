import os
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table

# 定义存放实验数据的根目录
log_dir = 'log'

# 初始化一个空的列表来存放所有实验数据
data = []

# 遍历log目录下的所有实验文件夹
for experiment in os.listdir(log_dir):
    # 构建means.txt文件的完整路径
    means_file_path = os.path.join(log_dir, experiment, 'imgs_test_all', 'mean.txt')
    
    # 检查means.txt文件是否存在
    if os.path.exists(means_file_path):
        # 读取means.txt文件的内容
        with open(means_file_path, 'r') as file:
            lines = file.readlines()
            # 假设每个文件中恰好有四行数据，对应PSNR、SSIM、lpipsv、lpipsa
            data.append([experiment] + [float(line.strip()) for line in lines])

# 将数据转换为DataFrame
df = pd.DataFrame(data, columns=['Experiment', 'PSNR', 'SSIM', 'LPIPSV', 'LPIPSA'])

# 确保DataFrame按某个列排序，这里假设按照'Experiment'列排序
df = df.sort_values(by='Experiment')

# 创建一个绘图窗口
fig, ax = plt.subplots(figsize=(12, len(df) * 0.3)) # 窗口大小可能需要根据实际数据量调整
ax.axis('tight')
ax.axis('off')
ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

plt.subplots_adjust(left=0.2, bottom=0.2)

# 检查是否存在一个用于保存图片的目录，如果不存在则创建
save_dir = 'visualization_results'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 保存图表到指定的文件
save_path = os.path.join(save_dir, 'experiment_metrics_table.png')
plt.savefig(save_path, dpi=300)
print(f"表格已保存至：{save_path}")

# 如果不需要在脚本运行后显示图表，可以注释掉下一行
# plt.show()
