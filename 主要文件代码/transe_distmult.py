# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd


# ------------------- 全局配置（用户需修改） -------------------
TRANSE_CKPT_PATH = "./checkpoint/transe_WN18RR.ckpt"    # TransE 模型路径
DISTMULT_CKPT_PATH = "./checkpoint/distmult.ckpt"  # DistMult 模型路径
TEST_TRIPLES_PATH = "./benchmarks/WN18RR/test2id.txt"  # 你的测试集路径（格式：h_id t_id r_id）
ENT_TOT = 40943  # 实体总数（根据你的数据集调整）
REL_TOT = 11      # 关系总数（根据你的数据集调整，或通过模型自动获取）
EMBEDDING_DIM = 200  # 嵌入维度（与训练时一致）


# ------------------- 工具函数 -------------------
def load_embeddings(ckpt_path):
    """加载 .ckpt 文件中的实体/关系嵌入矩阵"""
    try:
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        ent_emb = checkpoint['ent_embeddings.weight']  # 实体嵌入矩阵 [ENT_TOT, DIM]
        rel_emb = checkpoint['rel_embeddings.weight']  # 关系嵌入矩阵 [REL_TOT, DIM]
        return ent_emb, rel_emb
    except Exception as e:
        raise RuntimeError(f"加载模型失败：{str(e)}")


def load_triples(triple_path):
    """
    加载三元组文件（你的数据集格式：h_id t_id r_id）
    转换为代码需要的格式：(h_id, r_id, t_id)
    """
    triples = []
    with open(triple_path, "r") as f:
        lines = f.readlines()
        total = int(lines[0].strip())  # 首行为三元组总数
        for line in lines[1:]:  # 跳过首行
            h_id, t_id, r_id = map(int, line.strip().split())  # 解析为 h_id, t_id, r_id
            triples.append((h_id, r_id, t_id))  # 转换为 (h_id, r_id, t_id) 供后续计算使用
    return triples


def compute_transe_score(h_id, r_id, t_id, ent_emb, rel_emb):
    """计算 TransE 模型对三元组 (h, r, t) 的得分（尾预测）"""
    h = ent_emb[h_id]
    r = rel_emb[r_id]
    t = ent_emb[t_id]
    return -torch.norm(h + r - t, p=2).item()  # 得分越高，尾实体越合理


def compute_distmult_score(h_id, r_id, t_id, ent_emb, rel_emb):
    """计算 DistMult 模型对三元组 (h, r, t) 的得分（尾预测）"""
    h = ent_emb[h_id]
    r = rel_emb[r_id]
    t = ent_emb[t_id]
    return torch.dot(h * r, t).item()  # 得分越高，尾实体越合理


def compute_mrr_and_hits(scores_df, k_list=[1, 3, 10]):
    """计算 MRR 和 Hits@k 指标（针对尾预测任务）"""
    mrr = 0.0
    hits = {k: 0.0 for k in k_list}

    for idx, row in scores_df.iterrows():
        true_t = row['true_t_id']
        all_scores = row['scores']  # 所有候选尾实体的得分列表

        # 生成（候选尾实体ID，得分）的列表，并按得分降序排序
        candidates = list(enumerate(all_scores))
        candidates.sort(key=lambda x: -x[1])  # 得分越高，排名越靠前

        # 找到真实尾实体的排名（从1开始计数）
        rank = None
        for pos, (t_id, score) in enumerate(candidates):
            if t_id == true_t:
                rank = pos + 1  # 排名从1开始
                break
        if rank is None:
            continue  # 理论上不会发生（真实尾实体必在候选列表中）

        # 更新 MRR
        mrr += 1.0 / rank

        # 更新 Hits@k
        for k in k_list:
            if rank <= k:
                hits[k] += 1

    # 归一化
    total = len(scores_df)
    mrr /= total
    for k in hits:
        hits[k] /= total

    return mrr, hits


# ------------------- 主流程 -------------------
def main():
    # ------------------- 步骤1：加载模型嵌入 -------------------
    try:
        ent_emb_transe, rel_emb_transe = load_embeddings(TRANSE_CKPT_PATH)
        ent_emb_distmult, rel_emb_distmult = load_embeddings(DISTMULT_CKPT_PATH)
    except RuntimeError as e:
        print(f"模型加载错误：{e}")
        return

    # 打印模型实际关系总数（关键验证）
    print(f"TransE 关系总数（模型实际值）：{rel_emb_transe.shape[0]}")
    print(f"DistMult 关系总数（模型实际值）：{rel_emb_distmult.shape[0]}")

    # ------------------- 步骤2：加载测试集三元组（你的数据集格式：h_id t_id r_id） -------------------
    try:
        test_triples = load_triples(TEST_TRIPLES_PATH)
        print(f"测试集三元组数量（尾预测）：{len(test_triples)}")
        # 验证前3个三元组的格式（h_id, r_id, t_id）
        print("前3个三元组（h_id, r_id, t_id）:", test_triples[:3])
    except Exception as e:
        print(f"测试集加载错误：{e}")
        return

    # ------------------- 步骤3：计算所有候选尾实体的得分（向量化优化） -------------------
    print("计算测试集所有候选尾实体得分...")
    test_results = []

    for h_id, r_id, true_t_id in test_triples:
        # 检查 r_id 是否在模型关系范围内
        if r_id < 0 or r_id >= rel_emb_transe.shape[0]:
            print(f"警告：三元组 (h_id={h_id}, r_id={r_id}, t_id={true_t_id}) 的关系ID超出模型范围！")
            continue

        # ------------------- TransE 得分（向量化计算） -------------------
        h_transe = ent_emb_transe[h_id]  # [DIM]
        r_transe = rel_emb_transe[r_id]  # [DIM]
        all_t_transe = ent_emb_transe    # 所有候选尾实体的嵌入 [ENT_TOT, DIM]
        s_transe_all = -torch.norm(h_transe + r_transe - all_t_transe, p=2, dim=1)  # [ENT_TOT]

        # ------------------- DistMult 得分（向量化计算） -------------------
        h_distmult = ent_emb_distmult[h_id]  # [DIM]
        r_distmult = rel_emb_distmult[r_id]  # [DIM]
        all_t_distmult = ent_emb_distmult    # 所有候选尾实体的嵌入 [ENT_TOT, DIM]
        s_distmult_all = torch.sum(h_distmult * r_distmult * all_t_distmult, dim=1)  # [ENT_TOT]

        # ------------------- 集成策略：加权平均（假设 alpha=0.7） -------------------
        alpha = 0.7  # 可替换为验证集调优后的最优值
        s_final_all = alpha * s_transe_all + (1 - alpha) * s_distmult_all  # [ENT_TOT]

        # 保存结果（真实尾实体ID，所有候选得分）
        test_results.append({
            "true_t_id": true_t_id,
            "scores": s_final_all.tolist()  # 转换为 Python 列表
        })

    if not test_results:
        print("无有效测试数据，程序终止。")
        return

    # 转换为 DataFrame
    df_test = pd.DataFrame(test_results)

    # ------------------- 步骤4：计算 MRR 和 Hits@k -------------------
    print("计算评估指标...")
    mrr, hits = compute_mrr_and_hits(df_test, k_list=[1, 3, 10])

    # ------------------- 输出结果 -------------------
    print("\n===== 集成模型评估结果 =====")
    print(f"MRR（Mean Reciprocal Rank）: {mrr:.4f}")
    print(f"Hits@1: {hits[1]:.4f}")
    print(f"Hits@3: {hits[3]:.4f}")
    print(f"Hits@10: {hits[10]:.4f}")


if __name__ == "__main__":
    main()