# SnowflakeNet + PointNet 条件扩散实验

基于 linacao-325/SnowflakeNet main，commit 0880ba1e1bf1c135d276aa0044f0acd38c44d94e。此目录覆盖到完整仓库根目录即可，原训练入口不变。

## 实现

教师输入 partial xyz+标记0、GT xyz+标记1，拼接后经共享 MLP 4→64→128→256；max/mean pooling 拼接为512维，再经 MLP 512→256→256 与 LayerNorm 得到 F。学生采用相同结构，仅用 partial+标记0产生 F'。

第一阶段联合训练教师、去噪器和 SPD。第二阶段加载第一阶段全部权重，冻结教师，训练学生、去噪器和 SPD；特征约束为 mean MSE(F', stopgrad(F))，默认权重1。学生随机初始化，去噪器与 SPD 继承教师阶段权重。

用轻量、点序等变的残差点 MLP 预测 epsilon，各层注入 F/F' 和正弦时间编码，通过全局 max 汇聚交互。余弦噪声调度1000步；推理从独立高斯噪声以 DDIM eta=0 生成256个完整粗点。保留 partial 拼接+FPS到512点，以及原 SPD [1,4,8]，最终16384点。SPD全局条件通道改为256。

训练损失为 epsilon MSE + 多尺度CD-L1 + partial matching + λKD。粗目标由GT FPS取得。SPD使用 t∈[0,99] 的预测 x0 训练，epsilon损失覆盖全部1000步。这是明确的初始实验选择：低噪声训练与自由采样间存在分布差异，需要通过验证集评估；尚未验证最优。

验证/测试只调用 student(partial)，GT仅用于计算指标。教师阶段不报告学生验证指标；第二阶段按val的真实采样CD保存best.pt。实现了独立训练入口，尚未支持断点续训（checkpoint保存优化器状态但不自动恢复）。

## 运行

先按原仓库README安装 CUDA/PyTorch、pointnet2_ops 和数据依赖。将 PCN 数据放入 data/PCN 或修改 experiments/pcn.yaml；category_file_path 使用原仓库的 completion/category_files/ShapeNet.json。需要train、val、test划分。所有命令从仓库根目录执行。

```bash
python -m experiments.smoke
python -m experiments.run --config experiments/pcn.yaml --stage teacher --out work/teacher
python -m experiments.run --config experiments/pcn.yaml --stage student --checkpoint work/teacher/last.pt --out work/student
python -m experiments.run --config experiments/pcn.yaml --stage eval --checkpoint work/student/best.pt --split test --out work/test50 --steps 50
```

默认每阶段400 epoch，Adam lr=.001，50 epoch衰减0.5；batch等来自配置。第一阶段采用固定epoch，不用GT条件的测试分数挑选模型。CD使用精确分块torch.cdist以免强制加载EMD扩展，计算开销可能较大；完整GPU实验应确认损失耗时，并对所有候选使用同一损失实现测训练吞吐。

## 实验矩阵与判定

| 实验 | 训练 | 目的 |
|---|---|---|
| 原SnowflakeNet | 原仓库 completion/train.py | 基线精度与速度 |
| 条件扩散，无KD | 相同教师checkpoint，student --kd-weight 0 | 分离特征约束收益 |
| 条件扩散，有KD | student --kd-weight 1 | 主实验 |
| 采样步数 | 固定学生checkpoint，eval --steps 10/20/50 | 精度/推理耗时权衡 |

先用一个seed进行短训排错，再用2022/2023/2024完整重复，报告test CD-L1×1000的均值/标准差、每形状推理时间、训练迭代耗时和GPU显存。不要用test选权重或采样步数。计时必须同GPU、batch、点数、精度，并统一预热条件；当前runner时间包括首批、排除数据载入与CD，不能直接与其他口径比较。基线最终点数和输入点数应保持一致。

此设计减少编码器开销，但扩散有多次去噪，不能预先断言整体比Simba或原SnowflakeNet更快。

## 已执行与尚未执行

CPU PyTorch 2.3.1上 smoke 检查已通过：256维输出、池化点序不变性、教师阶段梯度、学生阶段教师冻结/无梯度、学生反传、有限梯度、固定随机种子的DDIM采样一致性、采样步数边界。Python编译检查和CLI --help通过。

当前环境无GPU；PCN目录仅包含索引，无实际点云；Git克隆和pip下载代理连接失败。通过GitHub插件读取源码并在本地实现。本次尚未运行完整SPD前后向、真实数据训练或评估，没有CD提升、速度提升或收敛结论。


## 双卡 DDP

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m experiments.run --config experiments/pcn.yaml --stage teacher --out work/teacher
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m experiments.run --config experiments/pcn.yaml --stage student --checkpoint work/teacher/last.pt --out work/student
```

每个进程绑定LOCAL_RANK对应GPU，NCCL同步梯度。训练使用DistributedSampler并每epoch调用set_epoch；样本数不整除卡数时按PyTorch默认补齐。训练指标在所有rank汇总，仅rank0写日志和保存不带DDP前缀的checkpoint。验证/测试由rank0遍历完整划分，其他rank等待，避免重复统计。

配置 train.batch_size 是**每卡**batch size，默认32，双卡总batch=64。若保持原单卡总batch=32，改为每卡16；学习率默认仍为.001。train.num_workers同样是每进程设置。单卡启动命令仍然可用。

双卡补丁已通过Python编译、核心CPU smoke和训练objective的CPU前后向检查（FPS/SPD使用替代实现）。本环境禁止创建本地socket，torchrun的双进程检查在rendezvous初始化时受阻；尚未实测双GPU/NCCL。可在训练机执行双CPU检查：

```bash
torchrun --standalone --nproc_per_node=2 -m experiments.ddp_smoke
```

该检查使用真实教师/学生和扩散模块，替代FPS/SPD，不代表完整CUDA算子的验证。

## OOM 修复

Chamfer已改为无梯度分块查找最近邻，再通过匹配点对计算距离和反传，避免旧版分块cdist保留全部距离矩阵的反向图。搜索仍是精确全点搜索，耗时仍需实测。随机无并列最近邻输入的距离值、双方梯度已与完整cdist对照通过；零距离梯度有限。并列最近邻采用首个索引。

新增 --batch-size（每卡）、--eval-batch-size（仅rank0验证）和 --accum-steps。先从每卡1、验证1开始排错：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m experiments.run --config experiments/pcn.yaml --stage teacher --out work/teacher --batch-size 1 --eval-batch-size 1 --accum-steps 16
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m experiments.run --config experiments/pcn.yaml --stage student --checkpoint work/teacher/last.pt --out work/student --batch-size 1 --eval-batch-size 1 --accum-steps 16
```

双卡正常完整累积组的有效batch=1×2×16=32。非最后微批次使用DDP.no_sync，最后不足一组仍会更新，按实际微批次数归一化。各微批次的loss取平均；若最后微批次样本更少，并非按样本数加权。此实现不使用AMP，保持原CUDA扩展的float32输入。没有GPU峰值显存或完整训练验证结果；SPD本身仍可能在小显存卡上OOM。

## CUDA Chamfer 提速

默认 --chamfer-backend auto：CUDA训练优先使用环境中已安装的Simba `chamfer` 扩展，否则直接加载SnowflakeNet的Chamfer3D文件（避免同时导入EMD），必要时首次JIT编译。要求CUDA toolkit、兼容编译器和ninja。CUDA算子只提供最近邻索引，距离与梯度仍由匹配点对计算，避免sqrt(0)梯度问题。首次编译耗时不代表每批训练耗时。

可以用 --chamfer-backend cuda 强制GPU算子；--chamfer-backend chunked 为显式慢速回退。CUDA加载失败会报错，不会静默回退。GT目标由16384→2048→512→256分层FPS获取并复用，取代各尺度分别从完整GT采样；与基线层级采样方式一致，但目标点集相对先前版本有所改变，应重新训练作公平对比。

执行 `python -m experiments.chamfer_smoke` 对照CUDA与分块参考的距离和双方梯度。开发环境只验证CPU匹配点对路径，CUDA算子、显存与提速比尚未实测。正式训练先维持每卡1确认运行，再试每卡2/累积8或每卡4/累积4，以保持双卡有效batch32。
