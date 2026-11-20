# P100-node1 运行记录

2026 年 11 月 20 日 17:28–17:31（北京时间），Tesla P100-SXM2-16GB。运行 ID 为 `smoke-01`，使用源码提交 `b60288667438628b6ea677561df2bf031da9ccee`。服务器部署归档与本地 Git 归档的 SHA-256 均为 `1c7bdd40180922d5bbfacd630a2a1dd00ccd84c3a6f1f58f73fcb2a58144ef6f`。

```bash
cd /root/data1/igarss_dbmf/b602886
PYTHONPATH=/root/data1/igarss_t18_ramp_pilot_20261114/vendor \
  bash run_smoke.sh /root/data1/igarss_dbmf/b602886/runs/smoke-01
```

运行环境为 PyTorch 2.5.1+cu118、CUDA 11.8、NVIDIA 驱动 510.47.03、pandas 2.2.3、NumPy 1.26.4、pvlib 0.11.2 和 Pillow 11.1.0。固定种子 42，批量 64，训练 1 轮，共 16 个训练批次。晴朗指数域训练 MSE 为 0.056950144，验证 MSE 为 0.016703243。测试前重新载入第 1 轮的最优检查点。

| 切分 | UTC 起报日期范围 | 样本 | 帧龄范围（秒） |
| --- | --- | ---: | ---: |
| 2014 训练 | 1 月 2–5 日 | 1,024 | 0–60 |
| 2015 验证 | 1 月 1–5 日 | 1,024 | 4–37 |
| 2016 测试 | 1 月 1–5 日 | 1,024 | 18–41 |

| 时效（分钟） | DBMF RMSE | DBMF MAE | DBMF MBE | R² | Smart Persistence RMSE | Skill |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 119.20 | 97.91 | 43.79 | 0.1536 | 62.84 | −89.68% |
| 20 | 126.34 | 104.14 | 61.72 | 0.0456 | 72.94 | −73.21% |
| 30 | 129.60 | 106.83 | 63.99 | 0.0197 | 77.14 | −68.01% |

误差单位为 W/m²。已从 `predictions.csv` 独立重算全部指标，并核对三个切分的时间年份、帧龄、唯一性和 2016 年预测行对齐。单轮年初样本满足端到端验收，三个时效的 RMSE 均高于 Smart Persistence。论文 Table 4 使用全年共同测试集和三种子汇总。

运行日志、配置、样本清单、预测和检查点已回传到本地 `runs/smoke-01/`。服务器保留相同运行目录和抽取的 JPEG。配置保存两份 Folsom 输入 CSV 的 SHA-256，三个 `extraction.json` 保存各年来源归档、字节数及抽取数量。采用的正式论文 PDF（DOI 10.1016/j.solener.2026.114864）SHA-256 为 `375adb46cf7a595d8b3d8a47ccf13a6a8c536c326c3686fb97a143237a9de244`。

| 产物 | SHA-256 |
| --- | --- |
| `best.pt` | `9341fbb2667837b2d3c63a212eacc5b30d67bde0dd0700e7ac807702379a1f85` |
| `metrics.csv` | `c3b1c495e81b3a072b4307fa7baff414ea98738af714d7f59655d702ca4d5f31` |
| `predictions.csv` | `ba9b90e287cf8957a355e719c82caf9f6b9a53f27e5996a267721de6ff7f70bb` |
