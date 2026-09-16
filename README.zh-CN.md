# SAIP：面向稠密视频描述的可摘要性感知迭代伪标注

论文 **Summarization-Aware Iterative Pseudo-Labeling for Dense Video Captioning**
的伪标签生成流程参考实现。

SAIP 把*完全无标注的原始视频*转换成可直接用于稠密视频描述（DVC）训练的标注集：
每段视频由若干时间事件组成，每个事件对应一句自然语言描述，格式与下游模型所用的
ActivityNet Captions / Charades 标注完全一致。

本仓库对应论文的第 3.1–3.4 节，即**到伪标签数据集为止**的全部流程。
下游模型（PDVC、Vid2Seq）的训练代码不在本仓库范围内。

```
 原始视频 ──► BLIP 特征 ──► 过生成候选事件池                      §3.1
                                  │
                                  ▼
                     SFS 打分  +  贪心筛选  ──► L(0)             §3.2
                                  │
                                  ▼
                        跨视频统计校准                            §3.4
                                  │
                                  ▼
              边界校准网络 → 边界重定位 → SFS 重算 → 重筛选       §3.3
              重复至满足式 (17)  ──► L*                          §3.3
                                  │
                                  ▼
                    伪标签数据集（train_pseudo.json）             §3.4
```

---

## 1. 仓库内容

| 路径 | 内容 |
|---|---|
| `saip/candidates.py` | **第 3.1 节** — 式 (2) 帧间距离、峰值检测、两两事件提议、候选池规模控制 |
| `saip/sfs.py` | **第 3.2 节** — 式 (4)–(10) 四个 SFS 维度，以及第 3.2.5 节的贪心筛选 |
| `saip/bcnet.py` | **第 3.3 节** — 式 (11)–(14) 网络、式 (15)–(16) 损失、边界重定位、式 (17) 停机条件 |
| `saip/calibration.py` | **第 3.4 节** — 类别校准、$$P(\text{type}\mid\text{category})$$、核心/背景事件分析 |
| `saip/pipeline.py` | 完整迭代流程 $$\mathcal{L}^{(0)} \to \mathcal{L}^{(1)} \to \dots \to \mathcal{L}^{*}$$ |
| `saip/features.py` | BLIP 适配层与视频解码；产出 $$h_t$$、$$c_t$$、$$q_i$$ 的接口 |
| `saip/dataset_io.py` | manifest 读取与两种伪标签数据集写出 |
| `saip/synthetic.py` | **仅供测试**的合成语料生成器（注意模块文档字符串中的警告） |
| `docs/paper_mapping.md` | 论文公式 ↔ 代码的逐条对照，含全部实现选择与论文中留白之处 |
| `configs/` | 可直接修改的 YAML 配置 |
| `tests/` | 单元测试 + CPU 端到端冒烟测试 |

## 2. 安装

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

跑通第 3.1、3.2、3.4 节只需要 `numpy`、`scikit-learn`、`PyYAML`。
第 3.3 节的边界校准网络需要 `torch`；若环境中没有 torch，流程仍会正常跑完并输出
仅由 SFS 得到的 $$\mathcal{L}^{(0)}$$，同时在报告里注明跳过原因。
只有在**进程内**提取特征（`blip` 后端）时才需要 `ffmpeg-python` 与 BLIP 源码；
默认的 `cache` 后端直接读磁盘上的数组。

## 3. 快速开始（无需下载权重、无需 GPU）

```bash
python scripts/make_demo_data.py --out demo_data --videos 8

python -m saip \
    --manifest demo_data/manifest.json \
    --feat-dir demo_data/feats \
    --caption-dir demo_data/captions \
    --text-feat-dir demo_data/text_feats \
    --out-dir demo_data/out \
    --device cpu
```

将写出 `demo_data/out/train_pseudo.json`、`pseudo_label_scores.json`、
`saip_report.json`。演示语料是合成的，用来验证代码链路可以跑通，其中的数字没有
实验意义。

## 4. 在真实数据上运行

### 4.1 三份缓存数组（推荐）

特征提取是一次性的 GPU 作业，因此流程从磁盘读取输入。对每个视频 id 提供：

```
feats/<video_id>.npy        float32 (T, 768)     式 (1) 的 h_t，每 stride 帧取一帧
captions/<video_id>.json    list[str]，长度 T      c_t，每帧的描述
text_feats/<video_id>.npy   float32 (T, 768)     式 (10) 的 q_i，即 c_t 的 BLIP 文本嵌入
```

`captions/<video_id>.json` 也可以是「列表的列表」（每个描述采样一条）；此时取每帧
的第一条。数组来自两个冻结的 BLIP 模型：

| 量 | 模型 |
|---|---|
| $$h_t$$、$$q_i$$ | BLIP 检索模型 `blip-itm-base-coco`（ViT-B，384×384） |
| $$c_t$$ | BLIP 描述模型 `blip-caption-large-coco` |

**权重不随本仓库分发。** 将 `FeatureConfig.itm_ckpt` 与
`FeatureConfig.caption_ckpt` 指向本地副本并加 `--backend blip` 即可在进程内运行；
或者用你自己的脚本离线提特征，然后使用默认的 `cache` 后端。

### 4.2 manifest

```json
{"videos": [
  {"id": "v_abc123",
   "duration": 185.2,
   "category": "Sports",
   "path": "/data/activitynet/videos/v_abc123.mp4"}
]}
```

`duration` 与 `category` 可选但建议填写：第 3.4(1) 节按类别归一化 $$S_{density}$$，
ActivityNet 写出器用 duration 归一化时间戳。若没有 `category`，则改为对视频级平均
特征聚类，与第 3.2.3 节的描述一致。

### 4.3 运行

```bash
python -m saip --config configs/activitynet.yaml \
    --manifest /data/activitynet/manifest_train.json \
    --out-dir runs/activitynet
```

任意单项都可以在不改配置文件的前提下覆盖：

```bash
python -m saip --config configs/activitynet.yaml \
    --num-events 20 --max-iters 10 --set candidates.fps=2.0
```

`--set` 接受 `section.field=value`，节名与字段名就是 YAML 文件里的那一套
（`candidates`、`sfs`、`bcnet`、`calibration`、`features`、`output`）。

Charades 使用同一套代码，加 `--dataset charades`，输出
`charades_sta_train_pseudo.txt`（格式为 `<video_id> <start> <end>##<description>`）。

## 5. 输出文件

| 文件 | 内容 |
|---|---|
| `train_pseudo.json` | `{video_id: {duration, timestamps: [[s, e], …]（归一化到 [0,1]）, sentences: [...]}}`，按时间排序 |
| `charades_sta_train_pseudo.txt` | 每行一个事件：`vid start end##description` |
| `pseudo_label_scores.json` | 每个入选事件的四个 SFS 原始维度与 SFS 总分——第 4.6 节消融实验切换的正是这里 |
| `saip_report.json` | 式 (17) 的逐轮历史、第 3.4(2)–(3) 节的统计、最终生效的配置，以及各类提示（例如 torch 不可用） |

## 6. 关键超参数

全部定义在 `saip/config.py` 与 `configs/*.yaml` 中，默认值即论文取值。

| 设置 | 默认值 | 出处 |
|---|---|---|
| `candidates.fps`（γ） | 3.0 | 第 3.1 节，「3 fps 关键帧」 |
| `candidates.min_span_sec` | 2.0 | 第 3.1 节，「不少于 2 秒（即 6 帧）」 |
| `candidates.pool_min` / `pool_max`（M） | 20 / 80 | 第 3.1 节 |
| `sfs.weights` | 各 0.25 | 式 (4) |
| `sfs.phi_near` / `phi_far` / `phi_floor` | 0.05 / 0.25 / 0.3 | 式 (7) |
| `sfs.tioi_thresh` | 0.5 | 第 3.2.5 节 |
| `sfs.num_events`（K） | 10 | 见 `docs/paper_mapping.md` |
| `bcnet.d_attn` / `n_heads` / `conv_kernel` / `hidden` | 256 / 4 / 3 / 128 | 式 (11)–(14) |
| `bcnet.lambda_diou` | 1.0 | $$\mathcal{L}_{total} = \mathcal{L}_{BCE} + \mathcal{L}_{DIoU}$$ |
| `bcnet.delta`（Δ） | 4 帧 | 式 (16)、第 3.3.2(2) 节 |
| `bcnet.delta_jaccard` / `delta_boundary_frames` | 0.98 / 0.5 | 式 (17) |
| `bcnet.max_iters` | 8 | 高于表 3 的 T = 6 的上限 |

## 7. 消融实验

消融表的每一行都是一个配置开关：

```bash
python -m saip --config configs/activitynet.yaml --ablate uniq     # 去掉 S_uniq
python -m saip --config configs/activitynet.yaml --ablate density  # 去掉 S_density
python -m saip --config configs/activitynet.yaml --ablate narr     # 去掉 S_narr
python -m saip --config configs/activitynet.yaml --ablate conf     # 去掉 S_conf
```

把某一维权重置零后重新归一化其余权重，就是对「从式 (4) 中移除该维度」的直接实现。
论文中与 SPL 对应的「仅 $$S_{conf}$$」配置为 `--set sfs.weights=0,0,0,1`。

## 8. 测试

```bash
pytest -q                        # 16 项测试，纯 CPU，数秒
python tests/test_pipeline.py    # 不用 pytest 的同一组检查
```

测试覆盖式 (2)、候选池上下界与式 (3) 的均值特征、式 (7) 两个折点、SFS 各维取值范围、
贪心规则、式 (15) 标签、式 (16) 损失的可微性、式 (17) 停机判据、
$$P(\text{type}\mid\text{category})$$ 的文档频率形式、随仓库提供的 YAML 配置能否正确
加载，以及合成语料上的完整流程。

## 9. 引用

```bibtex
@article{saip,
  title  = {Summarization-Aware Iterative Pseudo-Labeling for Dense Video Captioning},
  author = {...},
  year   = {...}
}
```

## 10. 许可

MIT，见 [LICENSE](LICENSE)。
