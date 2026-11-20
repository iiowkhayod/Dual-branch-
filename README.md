# DBMF 太阳辐照度临近预报复现

对应 Zhang 等发表于 *Solar Energy* 316 的 [论文](https://doi.org/10.1016/j.solener.2026.114864)。论文的 Data availability 声明作者应请求提供处理后数据和模型代码。本项目依据论文第 2–4 节独立实现，使用 [Folsom 原始数据](https://zenodo.org/records/2826939)。

## 验收范围

用真实 Folsom 2014 年数据完成梯度更新，2015 年验证集选权重，重载权重后在 2016 年输出 10、20、30 分钟 GHI 和 Smart Persistence 预测。每个时效输出 RMSE、MAE、MBE、R² 和相对 Smart Persistence 的 skill。验收要求有完整检查点、预测和指标文件，三个时效均有有限误差指标。这是端到端小样本运行，Table 4 的全年、三种子模型比较属于论文原实验。

P100-node1 已完成该验收，详见 [服务器运行记录](RESULTS.md)。

## 实现与论文对应

| 论文位置 | 实现 |
| --- | --- |
| §2.2–2.3 | 12 个数值输入、单张 64×64 RGB、30 步数值窗、2014 年拟合标准化 |
| §3.2–3.4 | 四块 CNN、两层 128 维 LSTM、LayerNorm 和拼接融合 |
| §3.6、§4.1 | Adam、MSE、梯度裁剪，预测晴朗指数并乘地外水平辐照度还原 GHI |
| §5 | 按时效计算 DBMF 和 Smart Persistence 的测试指标 |

独立实现采用完整连续的 30 个 UTC 分钟，删除缺测或重复观测，要求历史及三个监督目标的 GHI 大于 10 W/m²。论文允许窗口内相邻记录最多间隔 2 分钟。图像使用归档 mtime，按 UTC 向后匹配，默认最大帧龄 60 秒，可切换到文件名时钟。小时和年内日周期按 UTC 编码，RGB 缩放到 [0,1]。小样本运行提取每年归档中的前 2500 个 JPEG，在匹配结果上按时间均匀取至多 1024 个起报点，固定种子 42，训练 1 轮。各项选择及时间匹配写入运行目录。

## 在服务器运行

复用服务器现有 `folsom-lab` Python 环境和此前实验的 pvlib 依赖目录。环境或数据目录不同可覆盖 `DBMF_PYTHON` 和 `DBMF_RAW`。

```bash
export PYTHONPATH=/root/data1/igarss_t18_ramp_pilot_20261114/vendor
bash run_smoke.sh "$PWD/runs/smoke-01"
```

`run.log` 记录数据准备及训练，`result/` 保存配置、逐年样本清单、检查点、逐起报时刻的预测及指标。不覆盖已有运行目录。已有完整解包图像时，可通过 `python dbmf.py --help` 指定全年图像目录、训练轮数和运行参数。
