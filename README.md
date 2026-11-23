# 对抗攻击与对抗防御：面向 CIFAR-10 的攻防实践

《人工智能安全与伦理》课程两次大作业的完整实现。仓库中**没有任何一行是直接调用现成攻击库得到的**：
所有攻击都由本人在 `src/attacks.py` 中基于 `torch.autograd` 从零实现（自己写梯度符号、自己写投影、
自己写迭代循环），`torchattacks` 仅用于交叉验证自研实现是否正确。

| 项目 | 设置 |
|---|---|
| 数据集 | CIFAR-10，官方 50 000 / 10 000 划分（未重划，测试集不参与训练） |
| 主模型 | ResNet-18（CIFAR 版，11.17 M 参数），另有 ResNet-34 / VGG-16 作替代模型 |
| 攻击 | FGSM(基线)、PGD/BIM、MI-FGSM、C&W-L2、APGD-CE/DLR、AutoAttack-lite |
| 防御 | PGD 对抗训练（内层 7/10/20 步）、TRADES |
| 评估 | 1000 张固定分层评估子集；ASR / 鲁棒准确率；PGD-20、DLR、多重启、目标化、APGD |
| 环境 | Python 3.12.14 / PyTorch 2.4.1+cu121 / GT 1030 2 GB（也可 CPU 运行） |

## 1. 目录结构

```
ack_def/
├── README.md
├── requirements.txt
├── data/cifar10/            train.npz, test.npz, eval_subset.npy（由工具脚本生成）
├── src/
│   ├── common.py            种子、设备、日志、checkpoint、评估指标
│   ├── data.py              CIFAR-10 数据集、固定评估子集、反归一化
│   ├── models.py            ResNet-18/34、VGG-16、PixelModel（[0,1] 像素空间包装）
│   ├── attacks.py           【核心】全部自研攻击
│   ├── train.py             标准训练 / PGD-AT / TRADES
│   ├── evaluate.py          评估 CLI
│   ├── run_experiments.py   实验编排（报告里的每个数字都由它产生）
│   ├── figures.py           数据驱动图（图 1–8、12）
│   ├── qualitative.py       模型相关图（图 9–11）
│   ├── plot_style.py        色盲友好配色、中文字体
│   └── bench.py             速度基准
├── tests/                   单元测试（攻击实现的正确性）
├── tools/prepare_cifar10.py 数据准备（含 50 000/10 000 断言）
├── scripts/                 训练链脚本（需 -ExecutionPolicy Bypass）
├── checkpoints/             模型权重
├── results/                 CSV / JSON 结果
├── figures/                 图（PDF 矢量 + PNG 300 dpi）
└── report/                  报告生成脚本
```

## 2. 安装

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

## 3. 一条命令复现

```bash
# (1) 准备数据：把官方 CIFAR-10 打包成 npz，并断言 50 000/10 000 划分
python tools/prepare_cifar10.py --parquet-dir .pkg --out data/cifar10

# (2) 标准训练（作业一的目标模型）
python -m src.train --mode std --model resnet18 --epochs 6 --tag std_r18 \
       --bs 128 --lr 0.1 --wd 5e-4 --seed 0 --device cpu --threads 12 \
       --eval-subset 512 --save-every-epoch

# (3) 对抗训练（作业二的防御；从标准模型热启动）
python -m src.train --mode pgd_at --model resnet18 --epochs 1 --tag at_r18_eps8_s7 \
       --init checkpoints/std_r18.pt --eps 8 --steps 7 --lr 0.05 --seed 0 \
       --device cpu --threads 12 --eval-subset 512

# (4) 作业一：攻击实验（ε 网格、代价、步数、迁移）
python -m src.run_experiments attack
python -m src.run_experiments transfer

# (5) 作业二：防御评估（含自适应攻击）
python -m src.run_experiments defense --ckpt checkpoints/at_r18_eps8_s7.pt --tag at_r18_eps8_s7

# (6) 图与报告
python -m src.figures
python -m src.qualitative
python report/build_report.py --sid 学号 --name 姓名
```

`scripts/run_gpu_chain.ps1` / `run_cpu_chain.ps1` 把训练串成一条链，可后台运行。

## 4. 自研实现要点（与公式的对应）

| 公式 | 代码 | 说明 |
|---|---|---|
| Π_ε 投影 | `attacks.project_linf` | 先 `clamp(x−ε, x+ε)` 再 `clamp(0,1)`，两步缺一不可 |
| FGSM | `attacks.fgsm` | 一次 `autograd.grad` + `sign` + 投影 |
| PGD/BIM | `attacks.pgd_linf` | 随机起点、T 步、**逐样本**保留最优重启 |
| MI-FGSM | `attacks.mi_fgsm` | L1 归一化梯度动量累积，步长 ε/T |
| C&W-L2 | `attacks.cw_l2` | `x=0.5(tanh(w)+1)` 变量替换 + Adam + 置信度几何二分 |
| APGD | `attacks.apgd` | 动量 ρ=0.75 + 逐样本 1.75×/0.5× 自适应步长 + 重启 |
| DLR | `attacks.dlr_loss` | (z_max≠y − z_y)/(z_(1) − z_(3)) |
| AutoAttack-lite | `attacks.autoattack_lite` | APGD-CE ∪ APGD-DLR ∪ 目标化 APGD |
| PGD-AT | `train.train_step` | 内层直接复用 `pgd_linf`（训练与评估同一实现） |
| TRADES | `train.trades_inner` | 内层最大化 KL 而非 CE |

**攻击所在的坐标空间**：所有攻击都在 [0,1] 像素空间进行（`models.PixelModel` 在模型内部做归一化），
因此 ε=8/255 就是字面意义上的 8 个灰度级，`clamp(0,1)` 是合法的图像域投影。

## 5. 正确性检验

```bash
python -m tests.test_attacks    # 形状/取值域/L∞ 界/PGD-1≡FGSM/C&W 不动点
python -m tests.smoke_eval      # 评估链路冒烟测试
```

- PGD 在 `T=1, α=ε, 无随机起点` 时与 FGSM **逐元素相等**（已断言）；
- 所有攻击输出满足 `0 ≤ x_adv ≤ 1` 且 `‖x_adv−x‖∞ ≤ ε`（已断言）；
- 与 `torchattacks.PGD` 在相同 ε/步数下的成功率差异 < 3 个百分点（见报告 3.2 节）。

## 6. 引用与许可

本仓库代码全部为原创。`torchattacks`（MIT License, https://github.com/Harry24k/adversarial-attacks-pytorch）
仅在交叉验证时被调用，未参与任何报告数字的生成。CIFAR-10 数据集版权归原作者所有。
