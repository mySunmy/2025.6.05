import openke
from openke.config import Trainer, Tester
from openke.module.model import RESCAL
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

# -----------------------------
# 1. 加载训练数据
# -----------------------------
train_dataloader = TrainDataLoader(
    in_path = "./benchmarks/WN18RR/",   # 数据集路径
    nbatches = 100,                     # 每轮划分的 batch 数
    threads = 8,                        # 多线程数
    sampling_mode = "normal",          # 负采样模式
    bern_flag = 0,                      # 是否使用 Bernoulli trick
    filter_flag = 1,                    # 是否过滤正例
    neg_ent = 25,                       # 每个正例采样的负实体数
    neg_rel = 0                         # 每个正例采样的负关系数
)

# -----------------------------
# 2. 定义 RESCAL 模型
# -----------------------------
rescal = RESCAL(
    ent_tot = train_dataloader.get_ent_tot(),
    rel_tot = train_dataloader.get_rel_tot(),
    dim = 200                          # 向量维度（可调）
)

# -----------------------------
# 3. 定义损失函数 + 负采样策略
# -----------------------------
model = NegativeSampling(
    model = rescal,
    loss = MarginLoss(),
    batch_size = train_dataloader.get_batch_size()
)

# -----------------------------
# 4. 模型训练
# -----------------------------
trainer = Trainer(
    model = model,
    data_loader = train_dataloader,
    train_times = 1000,               # 训练轮数
    alpha = 0.05,                    # 学习率
    use_gpu = True
)

trainer.run()
rescal.save_checkpoint("./checkpoint/rescal_wn18rr.ckpt")  # 保存模型

# -----------------------------
# 5. 测试评估
# -----------------------------
test_dataloader = TestDataLoader("./benchmarks/WN18RR/", "link")

rescal.load_checkpoint("./checkpoint/rescal_wn18rr.ckpt")

tester = Tester(model=rescal, data_loader=test_dataloader, use_gpu=True)
tester.run_link_prediction(type_constrain=False)  # 不使用类型约束
