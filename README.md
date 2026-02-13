# 关于全向麦克风构成小型线性阵列在端射方向 DOA 估计退化的环节
**Endfire DOA degradation with a small linear omnidirectional microphone array**

本仓库包含一套基于**实录英语语音多通道 WAV**的数据与 **MATLAB** 评测脚本，用于研究与复现：当使用**4 元全向麦克风小型线性阵列**进行声源定位（DOA, 0°–180°）时，端射（endfire）附近多种算法出现退化的原因与表现，并比较不同算法在各方位角条件下的误差分布。

本仓库评测的 4 种算法为：
- **SRP-MVDR（MVDR 扫描）**：成熟基线算法（baseline）
- **SRP-PHAT**：成熟基线算法（baseline）
- **W-SRP-PHAT**：在 SRP-PHAT 基础上的加权/稳健改进（proposed）
- **GCC-WLS（Stable+）**：基于 GCC 曲线 + 鲁棒加权最小二乘融合的改进（proposed）

> 代码实现与批量评测入口：`measure.m`

---

## 1. 数据集（WAV）要求

### 1.1 多通道格式要求（脚本强依赖）
`measure.m` 读取每个 `.wav` 后会检查通道数，并**只取前 4 个通道**作为阵列输入：
- 需要 `>= 4` 通道，否则该文件会被跳过（skip）
- 采样率：脚本内部统一到 **16 kHz**（如文件采样率不同会 `resample` 到 16 kHz）
- 单段时长：你的数据为 1 s；脚本默认最多处理每条音频前 **3.0 s**（对 1 s 数据无影响）

> ⚠️ 因此：公开数据时请明确说明“WAV 是 4 通道阵列录音（每通道对应一个麦克风）”，并给出通道顺序。

### 1.2 命名规则与真值角解析
脚本通过文件名解析真值角度（TrueAngle），优先匹配：
- `(\d{1,3})d` 这样的模式（例如 `120d2m_014.wav` → TrueAngle=120）

示例：
- `120d2m_014.wav`：方位角 120°、距离 2 m、第 14 段（距离信息不参与当前统计）

> 建议：确保所有文件名都包含 `"{azimuth}d{distance}m_{index}.wav"` 这种格式，以避免角度解析歧义。

关于完整数据：公开于https://pan.baidu.com/s/wwwhy

---

## 2. 阵列与实验配置（来自 measure.m）

### 2.1 阵列几何（Linear array, M=4）
- 麦克风数量：**4**
- 相邻间距：**0.035 m**
- 阵列坐标（单位 m，沿 x 轴线性排布）：
  - `[0, 0, 0]`
  - `[-0.035, 0, 0]`
  - `[-0.070, 0, 0]`
  - `[-0.105, 0, 0]`

`W-SRP-PHAT` 与 `GCC-WLS` 使用**中心化坐标**（减去几何中心），基线方法使用原始坐标。

### 2.2 DOA 扫描范围
- 角度范围：**0°–180°**
- 默认假设：**远场（far-field）**、二维平面扫描（脚本内部等效固定仰角 `theta_s = 90°`）

> 如果你希望支持 0°–360° 或近场模型，需要扩展方向向量/传播模型与真值定义。

---

## 3. 4 种算法说明（实现细节对齐 measure.m）

### 3.1 基线 1：MVDR 扫描（SRP-MVDR）
实现位置：`measure.m` 中 `doa_two_methods_code2(...)`  
核心思路：
- 对每个频点构造协方差矩阵 `Rxx`
- 扫描方位网格 `phiV2 = 0:1:180`
- MVDR 空间谱累加：`Pmvdr(phi) += 1 / (a^H Rxx^{-1} a)`

关键配置（code2）：
- STFT：`win=512`，`overlap=0.5`（hop=256），Kaiser 窗
- 频段：约 **1 kHz – 4 kHz**
- 扫描步长：**1°**

### 3.2 基线 2：SRP-PHAT
实现位置：同 `doa_two_methods_code2(...)`  
特点（按脚本实现）：
- 只使用 3 组麦克风对：`(1,2) (1,3) (1,4)`
- 对每个频点做 PHAT 加权互谱，并累加到各候选角度的延时一致性得分

关键配置（code2）同 MVDR。

---

### 3.3 改进 1：W-SRP-PHAT（Weighted SRP-PHAT）
实现位置：`doa_srp_phat_wideband_farfield(...)`  
核心改进点（相对标准 SRP-PHAT）：
1) **全麦克风对融合**：使用全部 `nchoosek(4,2)=6` 对  
2) **长基线加权**：`pair_w = (baseline_length / max)^2`  
   - 直觉：端射方向更依赖较长基线提供更可分辨的时延差  
3) **频率加权**：`wf = (f/fmax)^(freq_alpha)`  
4) **相干性（coherence）加权**：`coh^(coh_alpha)`  
5) **能量门控帧选择（Energy VAD）**：保留参考通道在目标频带中能量排名前 **30%** 的帧，低能量帧自动丢弃

关键配置（code1）：
- STFT：`win=1024`，`overlap=0.75`（hop=256），Kaiser 窗
- 频段：**800 Hz – 4500 Hz**
- 扫描网格：`phiV = 0:0.2:180`（步长 **0.2°**）
- 权重指数：`freq_alpha=2`，`coh_alpha=1`

---

### 3.4 改进 2：GCC-WLS（Stable+）
实现位置：`doa_gcc_wls_farfield_v3(...)`  
该方法可理解为：
- 先构造每对麦克风的 **GCC(-PHAT) 时延相关曲线**（含频率/相干性加权与 oversampling）
- 再用 **SRP-like** 在角度网格上做一次粗搜索得到若干候选初值
- 最后进行 **guided peak picking**（在预测 tau 周围找峰）并把每对麦克风得到的 `u=cos(theta)` 观测用**鲁棒 WLS/Huber-IRLS**融合成最终角度

Stable+ 的“稳定性增强”（脚本注释明确指出避免“过度端射特化”导致曲线不自然）：
- **Top-L 多初值**：从 SRP 得分中选 Top-3 且间隔 ≥ 5° 的候选起点，减少混响/多径导致的错误初始化
- **自适应 tau 搜索窗**：在宽侧（broadside, u≈0）适当放大搜索窗；端射（|u|≈1）相对收紧
- **MAD 离群剔除**：在 IRLS 融合前对 u-measurements 做 MAD outlier rejection，避免单个坏麦对主导结果
- **Huber IRLS 融合**：对残差大的观测降低权重，提升鲁棒性

关键配置（code1, GCC-WLS）：
- GCC 频段：**1000 Hz – 4200 Hz**
- STFT：同 W-SRP-PHAT（`win=1024, hop=256`）
- 帧选择：能量 Top 30% + MSC Top 60%
- GCC 参数：
  - `gcc_beta = 0.7`（PHAT 幂次）
  - `gcc_freq_alpha = 3`，`gcc_coh_alpha = 1`
  - `gcc_oversample = 16`
- 角度网格（端射更密）：
  - `0:0.10:25`
  - `25.25:0.25:154.75`
  - `155:0.10:180`
- 鲁棒融合参数：
  - `irls_iter = 8`
  - `guided_iters = 2`
  - `huber_k_u = 0.08`
  - `psr_delta = 1.0`（PSR 权重指数）
  - `tau_win_ratio = 0.18`（基础 tau 搜索窗比例）

---

## 4. 快速运行（Batch Evaluation）

### 4.1 运行前修改路径
打开 `measure.m`，修改：
```matlab
audio_dir = "...\wav_split_1m";   % 你的 wav 数据目录
out_dir   = "...\End-Fire";       % 输出结果目录
```

---

## 5. 目录结构（Directory layout）

下面是一个推荐目录结构（你也可以按现有结构上传，但建议保持清晰）：

```text

├── data/

│ └── wav/ # 音频数据（.wav）

├── matlab/

│ ├── measure.m # 主评测脚本

│ ├── csv2mat.m # 工具：CSV -> MAT

│ ├── stas\_result.m # 工具：统计

│ └── draw.m # 工具：绘图

├── results/ # 输出结果（运行后生成）

├── LICENSE

└── README.md
```

> 如果你不想移动文件，也可以在 `measure.m` 里配置数据路径；README 里给了通用的运行方式。

---

## 6. 环境依赖（Requirements）

- MATLAB：建议 R2025
- 可能用到的工具箱（按需填写）：

---

## 7. 快速开始（Quick Start）

### 7.1 获取代码

```bash

git clone https://github.com/<your-username>/<repo-name>.git

cd <repo-name>
```

### 7.2 在 MATLAB 中运行

* 打开 MATLAB，将当前路径切到仓库根目录（或 matlab/ 目录）

* 添加路径：

```bash

addpath(genpath(pwd));
```

* 运行主评测脚本measure.m

如果 measure.m 需要你配置音频路径/输出路径，通常在脚本开头会有类似 DATA\_DIR / OUT\_DIR 的变量；请按你的实际目录修改。

---

## 8.实验流程建议（Suggested pipeline）

### 8.1 一个典型的流程如下（按需执行）：
算法评测（measure.m） --> 结果格式转换（如果 measure 输出为 CSV， csv2mat.m） --> 全局统计(stas\_result.m) --> 数据可视化 (draw.m)

---

## 9. 输出与指标（Outputs \& metrics）

本仓库的评测通常会关心（示例，可按你脚本实际输出调整）：

角度误差：|θ\_est - θ\_gt|

平均/中位数误差、RMSE

命中率：误差 ≤ 某阈值（例如 5° / 10°）

端射区域 vs 非端射区域的性能差异曲线/热力图

---

## 10. 联系方式（Contact）

Maintainer: WwHhYy666

Email: 2540160476@qq.com,3191479712@qq.com

Issues: 欢迎在 GitHub Issues 中反馈问题/复现实验差异/改进建议



