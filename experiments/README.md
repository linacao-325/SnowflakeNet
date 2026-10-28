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
