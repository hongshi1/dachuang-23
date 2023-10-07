import pandas as pd
from scipy.stats import wilcoxon

# 1. 读取Excel文件
data = pd.read_excel("../output/total_result.xlsx")  # 请将"your_file_path.xlsx"替换为你的文件路径

# 2. 提取“论文代码”列的数据
paper_code_data = data["论文代码"]

# 3. 对比其余的模型
models = ["随机森林", "决策树", "svr", "dpnn"]
for model in models:
    model_data = data[model]
    stat, p = wilcoxon(paper_code_data, model_data)
    print(f"Comparing 论文代码 vs {model}:")
    print(f"W-statistic: {stat}, P-value: {p}")
    if p < 0.05:
        print(f"There is a significant difference between 论文代码 and {model}")
    else:
        print(f"There is no significant difference between 论文代码 and {model}")
    print("----------")
