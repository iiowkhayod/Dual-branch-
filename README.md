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

## 单次全量主模型实验

使用三年归档中所有符合筛选条件的样本，2014 年训练，2015 年按验证 MSE 选择权重，2016 年计算 DBMF 和 Smart Persistence 的 GHI 指标。固定种子 42，最多 50 轮，早停耐心 8 轮。

全量实验依据 [Folsom 时钟对照研究 §3.5–3.6](https://arxiv.org/html/2503.21966)，以归档 mtime（DM）为起报时刻。对于重复 DM，按文件路径排序保留一帧。未来 10、20、30 分钟的标签按目标时刻相邻两条分钟观测线性插值，缺少任一所需观测则丢弃样本。三个目标均须满足白天阈值。地外水平辐照度按目标的秒级时间直接计算。历史窗口结束在 DM 向下取整后的整分钟，保留 30 个有效观测，相邻观测间隔最多 2 分钟。Smart Persistence 使用同一窗口内的最后一个已知晴朗指数。文件名时间仅用于逐日审计，运行不施加额外秒级平移。DBMF 论文未披露共同测试集名单，本次测试集由上述规则生成。

```bash
raw=/root/data1/folsom_multimodal/data/raw
cache="$PWD/full-cache-v1"
for year in 2014 2015 2016; do
    python cache_images.py "$raw/Folsom_sky_images_${year}.tar.bz2" "$cache/$year"
done
bash run_full.sh "$cache" "$PWD/runs/full-seed42"
```

缓存从归档依次解码 JPEG，并保存双线性缩放后的 `64×64` RGB uint8 及原始帧时间。无法解码的 JPEG 记在 `unreadable.csv`。缓存目录只有在整份归档成功读取后才会生成 `complete.json`。全量运行每 250 个训练批次打印进度，最终写入 `exit_code` 和完整评估产物。

可用 `dbmf.py --issue-clock image --max-history-gap 2 --prepare-only` 生成逐年样本清单和逐日 FN−DM 审计，另需指定数据路径和输出目录。样本清单包括实际起报时间及历史观测最早、最晚的记录时间，可据此核对起报时的数值输入。
