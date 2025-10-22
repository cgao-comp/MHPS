# 这个就是构造网络推演图的文件


import numpy as np
import networkx as nx
from collections import defaultdict

def read_cascade_file(file_path):
    cascades = []
    with open(file_path, 'r') as f:
        for line in f:
            pairs = line.strip().split()
            cascade = {}
            for pair in pairs:
                node, time = map(float, pair.split(','))
                cascade[int(node)] = time
            cascades.append(cascade)
    return cascades

def get_unique_nodes(cascades):
    unique_nodes = set()
    for cascade in cascades:
        unique_nodes.update(cascade.keys())
    return sorted(unique_nodes)

def DANI(cascades, K):
    # 映射节点ID到连续索引
    unique_nodes = get_unique_nodes(cascades)
    node_id_to_index = {node: idx for idx, node in enumerate(unique_nodes)}
    N = len(unique_nodes)
    
    # 初始化概率矩阵（稀疏存储）
    P = defaultdict(lambda: defaultdict(float))
    node_status_set = defaultdict(set)
    
    # 处理每个级联
    for c_idx, cascade_dict in enumerate(cascades):
        # 转换节点ID为索引，并按时间排序
        cascade = {node_id_to_index[node]: time for node, time in cascade_dict.items()}
        sorted_nodes = sorted(cascade.keys(), key=lambda x: cascade[x])
        CV = {node: i+1 for i, node in enumerate(sorted_nodes)}
        
        # 计算D矩阵
        D = defaultdict(lambda: defaultdict(float))
        for i in range(len(sorted_nodes)):
            u = sorted_nodes[i]
            for j in range(i+1, len(sorted_nodes)):
                v = sorted_nodes[j]
                D[u][v] = 1 / (CV[v] * (CV[v] - CV[u]))
        
        # 归一化并累加至P
        for u in D:
            total = sum(D[u].values())
            for v in D[u]:
                P[u][v] += D[u][v] / total
        
        # 记录节点级联参与情况
        for node in sorted_nodes:
            node_status_set[node].add(c_idx)
    
    # 构建边权重A
    A = defaultdict(float)
    for u in range(N):
        for v in range(u+1, N):
            if u in P and v in P[u]:
                intersection = len(node_status_set[u] & node_status_set[v])
                union = len(node_status_set[u] | node_status_set[v])
                if union > 0:
                    A[(u, v)] = (intersection / union) * P[u][v]
    
    # 生成最终网络
    result = sorted(A.items(), key=lambda x: x[1], reverse=True)
    IG = nx.Graph()
    for (u, v), _ in result[:K]:
        IG.add_edge(u, v)
    
    return IG, result, A

def save_network_to_file(graph, output_file):
    with open(output_file, 'w') as f:
        for edge in graph.edges():
            f.write(f"{edge[0]} {edge[1]}\n")

# 示例用法
file_path = "/sda/home/wkjing/GODEN_jing/GODEN-main/data/memetracker/cascade.txt"
output_file = "output_network_meme1.txt"
cascades = read_cascade_file(file_path)
# K = 47992
# K = 17992
K = 155017
IG, result, A = DANI(cascades, K)
save_network_to_file(IG, output_file)
print(f"潜在的传播网络已保存到 {output_file}")
