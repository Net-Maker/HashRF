import os
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table

# 定义存放实验数据的根目录
log_dir = 'log'

nsvf = ["Bike","Lifesytle","Palace","Toad","Spaceship", "Steamtrain",  "Wineholder", "Robot"]
synthetic = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
tanks = ["barn", "caterpillar", "family", "ignatius", "truck"]
llff = ["fern", "flower" ,"fortress", "horns", "leaves", "orchids", "room", "trex"]

# 初始化一个空的列表来存放所有实验数据
data = []

for experiment in os.listdir(log_dir):
    # 只处理实验文件夹名包含"CP"的情况
    for item in synthetic:
        if item == experiment[8:-3]:
            # print(item)
        # 构建means.txt文件的完整路径
            means_file_path = os.path.join(log_dir, experiment, 'imgs_test_all', 'mean.txt')
        
        # 检查means.txt文件是否存在
            if os.path.exists(means_file_path):
            # 读取means.txt文件的内容
                with open(means_file_path, 'r') as file:
                    lines = file.readlines()
                    data.append([experiment] + [float(line.strip()) for line in lines])
            continue

# 将数据转换为DataFrame
df = pd.DataFrame(data, columns=['Experiment', 'PSNR', 'SSIM', 'LPIPSV', 'LPIPSA'])

# 计算PSNR、SSIM、LPIPSV和LPIPSA列的平均值
mean_values = df[['PSNR', 'SSIM', 'LPIPSA', 'LPIPSV']].mean()

# 将平均值添加为DataFrame的新行
df.loc['Average'] = ['Average'] + mean_values.tolist()

# 确保DataFrame按某个列排序，这里假设按照'Experiment'列排序
# 注意：当添加平均值行后可能不需要再次排序
# df = df.sort_values(by='Experiment')

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
