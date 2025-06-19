# -*- coding: utf-8 -*-
import torch
import numpy as np
from typing import Tuple, Optional

# ------------------- 全局配置（用户需根据实际路径修改） -------------------
ANALOGY_CKPT_PATH = "./checkpoint/analogy.ckpt"  # Analogy 模型检查点路径
SIMPLE_CKPT_PATH = "./checkpoint/simple.ckpt"     # SimplE 模型检查点路径
TEST_TRIPLES_PATH = "./benchmarks/WN18RR/test2id.txt"  # WN18RR 测试集路径
ENT_TOT = 40943  # WN18RR 实体总数（固定值）
REL_TOT = 11      # WN18RR 关系总数（固定值）
EMBEDDING_DIM = 400  # 嵌入维度（根据检查点实际维度设置，Analogy 通常为400）
HITS_K = [1, 3, 10]  # 评估的 Hits@k 指标


# ------------------- Analogy 模型加载函数 -------------------
def load_analogy_embeddings(
    ckpt_path: str,
    ENT_TOT: int,
    REL_TOT: int,
    EMBEDDING_DIM: int,
    device: Optional[torch.device] = None  # 允许手动指定设备（可选）
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    加载 Analogy 模型的实体嵌入、关系嵌入、类比矩阵 M（支持 GPU/CPU）
    
    Args:
        ckpt_path (str): 模型检查点路径（.pth 文件）
        ENT_TOT (int): 知识图谱的实体总数（需与检查点中实体嵌入行数一致）
        REL_TOT (int): 知识图谱的关系总数（需与检查点中关系嵌入行数一致）
        EMBEDDING_DIM (int): 嵌入维度（需与检查点中嵌入列数一致）
        device (Optional[torch.device]): 目标设备（可选，默认自动选择 GPU 或 CPU）
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 实体嵌入、关系嵌入、类比矩阵 M
    """
    try:
        # 设备初始化（支持手动指定或自动选择）
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"加载 Analogy 模型到设备：{device}")

        # 参数合法性校验
        if not isinstance(ENT_TOT, int) or ENT_TOT <= 0:
            raise ValueError(f"ENT_TOT 必须为正整数（当前值：{ENT_TOT}）")
        if not isinstance(REL_TOT, int) or REL_TOT <= 0:
            raise ValueError(f"REL_TOT 必须为正整数（当前值：{REL_TOT}）")
        if not isinstance(EMBEDDING_DIM, int) or EMBEDDING_DIM <= 0:
            raise ValueError(f"EMBEDDING_DIM 必须为正整数（当前值：{EMBEDDING_DIM}）")

        # 加载检查点到目标设备
        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"检查点文件未找到：{ckpt_path}")
        except Exception as e:
            raise RuntimeError(f"加载检查点失败：{str(e)}")

        # 提取嵌入（处理键名可能的变化）
        try:
            ent_emb = checkpoint['ent_embeddings.weight']  # 实体嵌入 [ENT_TOT, EMBEDDING_DIM]
            rel_emb = checkpoint['rel_embeddings.weight']  # 关系嵌入 [REL_TOT, EMBEDDING_DIM]
        except KeyError as e:
            raise KeyError(f"检查点中缺少键：{e}。请确认检查点文件格式是否正确（可能键名不同）")

        # 创建类比矩阵 M（与设备同步）
        M = torch.eye(EMBEDDING_DIM, device=device)  # 类比矩阵 [EMBEDDING_DIM, EMBEDDING_DIM]

        # 维度与设备验证
        if ent_emb.shape != (ENT_TOT, EMBEDDING_DIM):
            raise ValueError(
                f"实体嵌入维度不匹配：实际 {ent_emb.shape}，预期 ({ENT_TOT}, {EMBEDDING_DIM})\n"
                f"可能原因：ENT_TOT 或 EMBEDDING_DIM 参数错误，或检查点文件与参数不匹配"
            )
        if rel_emb.shape != (REL_TOT, EMBEDDING_DIM):
            raise ValueError(
                f"关系嵌入维度不匹配：实际 {rel_emb.shape}，预期 ({REL_TOT}, {EMBEDDING_DIM})\n"
                f"可能原因：REL_TOT 或 EMBEDDING_DIM 参数错误，或检查点文件与参数不匹配"
            )
        if M.shape != (EMBEDDING_DIM, EMBEDDING_DIM):
            raise ValueError(f"矩阵 M 维度错误：实际 {M.shape}，预期 ({EMBEDDING_DIM}, {EMBEDDING_DIM})")
        if M.device != device:
            raise RuntimeError(f"矩阵 M 未在目标设备 {device} 上（当前设备：{M.device}）")

        return ent_emb, rel_emb, M

    except Exception as e:
        raise RuntimeError(f"加载 Analogy 失败：{str(e)}") from e  # 保留原始异常栈


# ------------------- SimplE 模型加载函数（适配当前检查点） -------------------
def load_simple_embeddings(ckpt_path) -> Tuple[torch.Tensor, torch.Tensor]:
    """加载 SimplE 模型的实体嵌入（h和h'）、关系嵌入（r和r'）"""
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        
        # 从检查点中提取实际键名的嵌入（用户提供的键名）
        ent_emb = checkpoint['ent_embeddings.weight']  # 实体嵌入 [ENT_TOT, 200]（h和h'合并）
        rel_emb = checkpoint['rel_embeddings.weight']  # 关系嵌入 [REL_TOT, 200]（r和r'合并）

        # 验证维度（SimplE 要求实体嵌入为 2×100 维，合并为200维）
        assert ent_emb.shape == (ENT_TOT, 200), \
            f"SimplE 实体嵌入维度错误：实际 {ent_emb.shape}，预期 ({ENT_TOT}, 200)"
        assert rel_emb.shape == (REL_TOT, 200), \
            f"SimplE 关系嵌入维度错误：实际 {rel_emb.shape}，预期 ({REL_TOT}, 200)"

        return ent_emb, rel_emb

    except Exception as e:
        raise RuntimeError(f"加载 SimplE 失败：{e}")


# ------------------- Analogy 得分计算函数（修正维度） -------------------
def compute_analogy_score(h_id: int, r_id: int, t_id: int,
                         ent_emb: torch.Tensor, rel_emb: torch.Tensor,
                         M: torch.Tensor) -> float:
    """计算 Analogy 模型对三元组 (h, r, t) 的得分（适配400维嵌入）"""
    h = ent_emb[h_id]  # 实体 h 嵌入 [400]
    r = rel_emb[r_id]  # 关系 r 嵌入 [400]
    t = ent_emb[t_id]  # 实体 t 嵌入 [400]
 
    # Analogy 得分公式（无需修改，维度自动适配）
    term1 = torch.dot(torch.matmul(h, M) * r, t)  # (h^T M r) · t
    term2 = torch.dot(torch.matmul(t, M) * r, h)  # (t^T M r) · h
    return (term1 + term2).item()


# ------------------- SimplE 得分计算函数（适配合并嵌入） -------------------
def compute_simple_score(h_id: int, r_id: int, t_id: int,
                        ent_emb: torch.Tensor, rel_emb: torch.Tensor) -> float:
    """计算 SimplE 模型对三元组 (h, r, t) 的得分（适配200维合并嵌入）"""
    # 拆分实体嵌入为 h（前100维）和 h'（后100维）
    h = ent_emb[h_id, :100]       # [100]
    h_prime = ent_emb[h_id, 100:]  # [100]

    # 拆分关系嵌入为 r（前100维）和 r'（后100维）
    r = rel_emb[r_id, :100]       # [100]
    r_prime = rel_emb[r_id, 100:]  # [100]

    # 拆分实体 t 嵌入为 t（前100维）和 t'（后100维）
    t = ent_emb[t_id, :100]       # [100]
    t_prime = ent_emb[t_id, 100:]  # [100]

    # SimplE 得分公式：(h^T r t + h'^T r' t') / 2
    term1 = torch.dot(h * r, t)    # h·(r⊙t)
    term2 = torch.dot(h_prime * r_prime, t_prime)  # h'·(r'⊙t')
    return ((term1 + term2) / 2).item()


# ------------------- 集成评估函数（优化性能提示） -------------------
def evaluate_integration(analogy_ent_emb, analogy_rel_emb, analogy_M,
                         simple_ent_emb, simple_rel_emb,
                         test_triples, k_list: list):
    """集成 Analogy 和 SimplE 模型，评估 Hits@k 指标（完整逻辑）"""
    total = len(test_triples)
    hits = {k: 0 for k in k_list}
    all_entities = torch.arange(ENT_TOT)  # 所有实体ID：0~ENT_TOT-1
 
    for idx, (h_id, r_id, t_id) in enumerate(test_triples):
        # 注意：遍历所有实体计算得分在 ENT_TOT 较大时（如40943）会非常慢，
        # 实际使用时建议用矩阵运算优化（例如批量计算候选实体得分）
        t_candidates = all_entities
 
        scores = []
        for t_candidate in t_candidates:
            score_analogy = compute_analogy_score(h_id, r_id, t_candidate, 
                                                analogy_ent_emb, analogy_rel_emb, analogy_M)
            score_simple = compute_simple_score(h_id, r_id, t_candidate, 
                                              simple_ent_emb, simple_rel_emb)
            final_score = (score_analogy + score_simple) / 2
            scores.append(final_score)
 
        scores = torch.tensor(scores)
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
 
        rank = (sorted_indices == t_id).nonzero().item() + 1  # 排名从1开始
 
        for k in k_list:
            if rank <= k:
                hits[k] += 1
 
        if (idx + 1) % 100 == 0:
            print(f"评估进度：{idx+1}/{total}")
 
    for k in k_list:
        hits[k] /= total
    return hits


# ------------------- 主流程（修复路径调用错误） -------------------
def main():
    # 加载 Analogy 模型（修复：使用全局变量 ANALOGY_CKPT_PATH）
    device = torch.device("cuda:1") if torch.cuda.device_count() > 1 else torch.device("cuda")
    try:
        analogy_ent_emb, analogy_rel_emb, analogy_M = load_analogy_embeddings(
            ckpt_path=ANALOGY_CKPT_PATH,  # 修正：使用全局配置的路径
            ENT_TOT=ENT_TOT,
            REL_TOT=REL_TOT,
            EMBEDDING_DIM=EMBEDDING_DIM,
            device=device
        )
        print("Analogy 模型加载成功！")
    except Exception as e:
        print(f"Analogy 模型加载失败：{e}")
        return

    # 加载 SimplE 模型
    try:
        simple_ent_emb, simple_rel_emb = load_simple_embeddings(SIMPLE_CKPT_PATH)
        print("SimplE 模型加载成功！")
    except Exception as e:
        print(f"SimplE 模型加载失败：{e}")
        return

    # 加载测试集（假设文件格式为：第一行总数，后续每行 h_id t_id r_id）
    test_triples = []
    try:
        with open(TEST_TRIPLES_PATH, 'r') as f:
            lines = f.readlines()
            # 跳过第一行（总数），读取后续行并调整顺序为 (h_id, r_id, t_id)
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 3:
                    print(f"警告：无效行格式，跳过：{line}")
                    continue
                h_id, t_id_in_file, r_id_in_file = map(int, parts)
                test_triples.append((h_id, r_id_in_file, t_id_in_file))  # 调整为 (h, r, t)
        print(f"成功加载 {len(test_triples)} 个测试三元组")
    except Exception as e:
        print(f"加载测试集失败：{e}")
        return

    # 集成评估
    print("\n开始集成评估...")
    results = evaluate_integration(analogy_ent_emb, analogy_rel_emb, analogy_M,
                                   simple_ent_emb, simple_rel_emb,
                                   test_triples, HITS_K)

    # 输出结果
    print("\n评估结果：")
    for k in HITS_K:
        print(f"Hits@{k}: {results[k]:.4f}")


if __name__ == "__main__":
    main()