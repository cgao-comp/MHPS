
# 输入文件和输出文件路径
input_file = '/sda/home/wkjing/GRASS-main/data/christianity/edges.txt'
output_file = 'edges_comma_separated.txt'

# 读取文件并处理数据
with open(input_file, 'r') as f:
    data = f.read().strip().split('\n')

# 按空格替换为逗号
data_with_commas = [line.replace(' ', ',') for line in data]

# 保存处理后的结果到文件
with open(output_file, 'w') as f:
    for line in data_with_commas:
        f.write(line + '\n')

print(f"Data has been saved to {output_file}")
