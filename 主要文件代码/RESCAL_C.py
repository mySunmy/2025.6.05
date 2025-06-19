# -*- coding: utf-8 -*-
import torch
from typing import Tuple, Optional

# ------------------- 全局配置（用户需根据实际修改） -------------------
RESCAL_CKPT_PATH = "./checkpoint/rescal_wn18rr.ckpt"    # Rescal 模型检查点路径
COMPLEX_CKPT_PATH = "./checkpoint/complEx.ckpt"  # ComplEx 模型检查点路径
TEST_TRIPLES_PATH = "./benchmarks/WN18RR/test2id.txt"  # 测试集路径
ENT_TOT = 40943  # WN18RR 实体总数（固定值）
REL_TOT = 11      # WN18RR 关系总数（固定值）
EMBEDDING_DIM = 200  # Rescal/ComplEx 嵌入维度（根据模型实际设置）
HITS_K = [1, 3, 10]  # 评估指标


# ------------------- Rescal 模型加载函数（适配检查点键名） -------------------
def load_rescal_embeddings(ckpt_path: str, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    加载 Rescal 模型的实体嵌入和关系张量（适配检查点实际键名）
    
    Args:
        ckpt_path (str): 模型检查点路径
        device (torch.device): 目标设备（GPU/CPU）
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 实体嵌入 [ENT_TOT, EMBEDDING_DIM], 关系张量 [REL_TOT, EMBEDDING_DIM, EMBEDDING_DIM]
    """
    try:
        # 设备初始化
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载检查点到目标设备
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        print(f"成功加载 Rescal 检查点：{ckpt_path}")

        # 提取实体嵌入（键名与检查点一致：ent_embeddings.weight）
        ent_emb = checkpoint['ent_embeddings.weight']  # [ENT_TOT, EMBEDDING_DIM]

        # 提取关系张量（检查点实际键名为 rel_matrices.weight）
        rel_tensor = checkpoint['rel_matrices.weight']  # 关键修改：使用实际键名

        # 验证实体嵌入维度
        if ent_emb.shape != (ENT_TOT, EMBEDDING_DIM):
            raise ValueError(
                f"Rescal 实体嵌入维度错误：实际 {ent_emb.shape}，预期 ({ENT_TOT}, {EMBEDDING_DIM})\n"
                f"请检查 ENT_TOT 或 EMBEDDING_DIM 参数是否与检查点匹配"
            )

        # 验证关系张量维度（Rescal 要求三维张量）
        if rel_tensor.ndim != 3 or rel_tensor.shape != (REL_TOT, EMBEDDING_DIM, EMBEDDING_DIM):
            # 若检查点中关系张量是二维的（例如 [REL_TOT, EMBEDDING_DIM^2]），尝试重塑为三维
            if rel_tensor.ndim == 2 and rel_tensor.shape == (REL_TOT, EMBEDDING_DIM ** 2):
                rel_tensor = rel_tensor.reshape(REL_TOT, EMBEDDING_DIM, EMBEDDING_DIM)
                print(f"警告：关系张量从二维重塑为三维，新形状：{rel_tensor.shape}")
            else:
                raise ValueError(
                    f"Rescal 关系张量维度错误：实际 {rel_tensor.shape}，预期 ({REL_TOT}, {EMBEDDING_DIM}, {EMBEDDING_DIM})\n"
                    f"请检查检查点是否为 Rescal 模型，或调整参数 REL_TOT/EMBEDDING_DIM"
                )

        return ent_emb.to(device), rel_tensor.to(device)

    except KeyError as e:
        raise KeyError(f"检查点缺少关键参数：{e}。请确认检查点是否为 Rescal 模型，或键名是否匹配（如 'rel_matrices.weight'）")
    except Exception as e:
        raise RuntimeError(f"加载 Rescal 失败：{e}")


# ------------------- ComplEx 模型加载函数（无需修改） -------------------
def load_complex_embeddings(ckpt_path: str, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    try:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        ent_real = checkpoint['ent_real.weight']
        ent_imag = checkpoint['ent_imag.weight']
        rel_real = checkpoint['rel_real.weight']
        rel_imag = checkpoint['rel_imag.weight']
        ent_emb = torch.complex(ent_real, ent_imag)
        rel_emb = torch.complex(rel_real, rel_imag)
        assert ent_emb.shape == (ENT_TOT, EMBEDDING_DIM), \
            f"ComplEx 实体嵌入维度错误：实际 {ent_emb.shape}，预期 ({ENT_TOT}, {EMBEDDING_DIM})"
        assert rel_emb.shape == (REL_TOT, EMBEDDING_DIM), \
            f"ComplEx 关系嵌入维度错误：实际 {rel_emb.shape}，预期 ({REL_TOT}, {EMBEDDING_DIM})"
        return ent_emb.to(device), rel_emb.to(device)
    except Exception as e:
        raise RuntimeError(f"加载 ComplEx 失败：{e}")


# ------------------- Rescal 得分计算函数（无需修改） -------------------
def compute_rescal_score(h_id: int, r_id: int, t_id: int,
                         ent_emb: torch.Tensor, rel_tensor: torch.Tensor) -> float:
    h = ent_emb[h_id]
    R_r = rel_tensor[r_id]
    t = ent_emb[t_id]
    score = torch.matmul(torch.matmul(h, R_r), t)
    return score.item()


# ------------------- ComplEx 得分计算函数（无需修改） -------------------
def compute_complex_score(h_id: int, r_id: int, t_id: int,
                          ent_emb: torch.Tensor, rel_emb: torch.Tensor) -> float:
    h = ent_emb[h_id]
    r = rel_emb[r_id]
    t = ent_emb[t_id]
    score = torch.real(torch.sum(h * torch.conj(r) * torch.conj(t)))
    return score.item()


# ------------------- 集成评估函数（无需修改） -------------------
def evaluate_integration(rescal_ent_emb, rescal_rel_tensor,
                         complex_ent_emb, complex_rel_emb,
                         test_triples, k_list: list):
    total = len(test_triples)
    hits = {k: 0 for k in k_list}
    all_entities = torch.arange(ENT_TOT)
    for idx, (h_id, r_id, t_id) in enumerate(test_triples):
        scores = []
        for t_candidate in all_entities:
            score_rescal = compute_rescal_score(h_id, r_id, t_candidate, rescal_ent_emb, rescal_rel_tensor)
            score_complex = compute_complex_score(h_id, r_id, t_candidate, complex_ent_emb, complex_rel_emb)
            final_score = (score_rescal + score_complex) / 2
            scores.append(final_score)
        scores = torch.tensor(scores)
        _, sorted_indices = torch.sort(scores, descending=True)
        rank = (sorted_indices == t_id).nonzero().item() + 1
        for k in k_list:
            if rank <= k:
                hits[k] += 1
        if (idx + 1) % 100 == 0:
            print(f"评估进度：{idx+1}/{total}")
    for k in k_list:
        hits[k] /= total
    return hits


# ------------------- 主流程（无需修改） -------------------
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    try:
        rescal_ent_emb, rescal_rel_tensor = load_rescal_embeddings(RESCAL_CKPT_PATH, device)
        print("Rescal 模型加载成功！")
    except Exception as e:
        print(f"Rescal 模型加载失败：{e}")
        return

    try:
        complex_ent_emb, complex_rel_emb = load_complex_embeddings(COMPLEX_CKPT_PATH, device)
        print("ComplEx 模型加载成功！")
    except Exception as e:
        print(f"ComplEx 模型加载失败：{e}")
        return

    test_triples = []
    try:
        with open(TEST_TRIPLES_PATH, 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                h_id, t_id_in_file, r_id_in_file = map(int, line.split())
                test_triples.append((h_id, r_id_in_file, t_id_in_file))
        print(f"成功加载 {len(test_triples)} 个测试三元组")
    except Exception as e:
        print(f"加载测试集失败：{e}")
        return

    print("\n开始集成评估...")
    results = evaluate_integration(rescal_ent_emb, rescal_rel_tensor,
                                   complex_ent_emb, complex_rel_emb,
                                   test_triples, HITS_K)

    print("\n评估结果：")
    for k in HITS_K:
        print(f"Hits@{k}: {results[k]:.4f}")


if __name__ == "__main__":
    main()