# Endfire DOA Degradation with a Small Linear Omnidirectional Microphone Array

This repository contains a comprehensive dataset and MATLAB evaluation suite based on **real recorded multi-channel English speech WAV files** for investigating and reproducing the degradation phenomena of multiple DOA (Direction of Arrival) estimation algorithms when using a **4-element omnidirectional small linear microphone array** for acoustic source localization (DOA, 0°–180°). It also enables comparative analysis of error distributions across different algorithms and azimuth angles.

This repository evaluates **4 DOA estimation algorithms**:
- **SRP-MVDR (MVDR Scanning)**: Mature baseline algorithm
- **SRP-PHAT**: Mature baseline algorithm  
- **W-SRP-PHAT**: Weighted/robust improvement over SRP-PHAT (proposed)
- **GCC-WLS**: GCC curve + weighted least squares fusion (proposed)

> **Main entry point**: `measure.m` contains the core implementation and batch evaluation framework

---

## 1. Dataset (WAV) Requirements

### 1.1 Multi-channel Format Requirements (Script Dependency)
The `measure.m` script checks the number of channels after loading each `.wav` file and **only uses the first 4 channels** as array inputs:
- Requires `>= 4` channels; files with fewer channels will be skipped
- Sample rate: Internally standardized to **16 kHz** (files with different rates will be resampled to 16 kHz)
- Duration per segment: Your data is 1 second; the script processes up to **3.0 seconds** per file (no effect on 1-second data)

> ⚠️ **Important**: When sharing the public dataset, clearly specify "WAV files are 4-channel microphone array recordings (each channel corresponds to one microphone)" and provide the channel order.

### 1.2 Naming Convention and Ground Truth Angle Parsing
The script parses ground truth angle (TrueAngle) from filenames using the following priority:
- Pattern `(\d{1,3})d` (e.g., `120d2m_014.wav` → TrueAngle=120)

Examples:
- `120d2m_014.wav`: Azimuth angle 120°, distance 2 m, segment 14 (distance information not used in current analysis)

> **Recommendation**: Ensure all filenames follow the format `"{azimuth}d{distance}m_{index}.wav"` to avoid angle parsing ambiguity.

**Complete dataset**: Available via cloud storage - Link: https://pan.baidu.com/s/1cphwmxwBzt-lslcU0amuiQ?pwd=s8z7

---

## 2. Array Configuration and Experimental Setup (from measure.m)

### 2.1 Array Geometry (Linear array, M=4)
- Number of microphones: **4**
- Adjacent spacing: **0.035 m**
- Array coordinates (in meters, linear along x-axis):
  - `[0, 0, 0]`
  - `[-0.035, 0, 0]`
  - `[-0.070, 0, 0]`
  - `[-0.105, 0, 0]`

W-SRP-PHAT and GCC-WLS use **centered coordinates** (minus geometric center); baseline methods use original coordinates.

### 2.2 DOA Scanning Range
- Angular range: **0°–180°**
- Default assumption: **Far-field**, 2D planar scanning (script internally fixes elevation angle at `theta_s = 90°`)

> To support 0°–360° or near-field models, you need to extend the steering vectors/propagation model and ground truth definitions.

---

## 3. Four Algorithm Descriptions (Implementation Details Aligned with measure.m)

> **Baseline Algorithm Reference**: The implementation of **SRP-MVDR** and **SRP-PHAT** baseline algorithms is based on code adapted from the [sound-source-localization-algorithm_DOA_estimation](https://github.com/WenzheLiu-Speech/sound-source-localization-algorithm_DOA_estimation) project. We reimplemented and integrated these methods into our evaluation framework.

### 3.1 Baseline 1: MVDR Scanning (SRP-MVDR)
Implementation location: `doa_two_methods_code2(...)` in `measure.m`  
Core approach:
- Constructs covariance matrix $R_{xx}$ for each frequency bin
- Scans azimuth grid $\phi = 0°:0.2°:180°$
- Accumulates MVDR spatial spectrum: $$P_{mvdr}(\phi) = \sum_f \frac{1}{\mathbf{a}^H R_{xx}^{-1} \mathbf{a}}$$

Key configuration (code2):
- STFT: `win=1024`, `overlap=0.75` (hop=256), Kaiser window
- Frequency range: **800 Hz – 4500 Hz** (same as code1)
- Scanning grid: **0° : 0.2° : 180°** (same as code1)

### 3.2 Baseline 2: SRP-PHAT
Implementation location: `doa_two_methods_code2(...)`  
Characteristics (per script implementation):
- Uses only 3 microphone pairs: $(1,2), (1,3), (1,4)$
- For each frequency bin, applies PHAT weighting to cross-correlation and accumulates DOA consistency scores at candidate angles

Key configuration (code2): Same as MVDR.

---

### 3.3 Improvement 1: W-SRP-PHAT (Weighted SRP-PHAT)
Implementation location: `doa_srp_phat_wideband_farfield(...)`  
Core improvement points (vs. standard SRP-PHAT):
1) **Full microphone pair fusion**: Uses all $\binom{4}{2}=6$ pairs
2) **Long baseline weighting**: $$w_{pair} = \left(\frac{b}{b_{max}}\right)^2$$
   - Rationale: Endfire directions depend more heavily on longer baselines for better TDOA resolvability
3) **Frequency weighting**: $$w_f = \left(\frac{f}{f_{max}}\right)^{\alpha_f}$$
4) **Coherence weighting**: $$w_{coh} = \rho^{\alpha_{coh}}$$
5) **Energy-based frame selection (Energy VAD)**: Retains frames in the reference channel with energy ranking in the top **15%** within the target band; low-energy frames are automatically discarded

Key configuration (code1):
- STFT: $\text{win}=1024$, $\text{overlap}=0.75$ (hop=256), Kaiser window
- Frequency range: **800 Hz – 4500 Hz**
- Scanning grid: $\phi = 0°:0.2°:180°$ (step size **0.2°**)
- Weight exponents: $\alpha_f=2$, $\alpha_{coh}=2$

---

### 3.4 Improvement 2: GCC-WLS
Implementation location: `doa_gcc_wls_farfield(...)`  
This method:
- Constructs **GCC(-PHAT) time-delay correlation curves** for each microphone pair (with frequency/coherence weighting and oversampling)
- Applies per-pair TDOA extraction via oversampled cross-correlation
- Fuses each pair's direction estimate using **Weighted Least Squares (WLS)**

Key features:
- **Baseline-dependent weighting**: $$w_{pair} = \left(\frac{b}{b_{max}}\right)^2$$ - longer baselines have higher weight
- **Frequency weighting**: $$w_f = \left(\frac{f}{f_{max}}\right)^{\alpha_f}$$
- **Coherence weighting**: $$w_{coh} = \rho^{\alpha_{coh}}$$

Key configuration (code1, GCC-WLS):
- Frequency range: **800 Hz – 4500 Hz** (same as W-SRP-PHAT)
- STFT: Same as W-SRP-PHAT ($\text{win}=1024$, hop=256)
- Frame selection: Top 15% energy frames
- GCC parameters:
  - $\beta_{PHAT} = 0.8$ (PHAT exponent)
  - $\alpha_f = 2$ (frequency weighting exponent)
  - $\alpha_{coh} = 2$ (coherence weighting exponent)
  - $\text{oversample} = 16$ (oversampling factor for time-delay search)
- Angular grid: **0° : 0.2° : 180°** (0.2° resolution)

---

## 4. Quick Start (Batch Evaluation)

### 4.1 Modify Paths Before Running
Open `measure.m` and update:
```matlab
audio_dir = "...\wav_split_1m";   % Your WAV data directory
out_dir   = "...\End-Fire";       % Output results directory
```

---

## 5. Directory Structure

Below is a recommended directory layout (you can also keep your current structure, but maintaining clarity is recommended):

```
.
├── data/
│   └── wav/                 # Audio data (.wav files)
├── matlab/
│   ├── experiment/
│   │   └── measure.m        # Main evaluation script
│   ├── func/
│   │   ├── stft_1.m
│   │   ├── STFT.m
│   │   ├── STFTInv.m
│   │   └── synchronize_signal.m
│   └── tools/
│       ├── csv2mat.m        # Tool: CSV -> MAT conversion
│       ├── figure/
│       │   └── draw.m       # Tool: Visualization
│       └── statistic/
│           ├── compute_paper_metrics.m
│           └── stas_result.m # Tool: Statistics
├── python/
│   └── tools/
│       ├── datapre/
│       │   ├── pcm2wav.py
│       │   ├── split_merged_to_1s.py
│       │   └── trim_silence.py
│       └── figure/
│           └── plot_scene.py
├── runs/
│   └── results/             # Output results (generated after running)
├── LICENSE
└── README.md
```

> If you prefer not to move files, you can configure data paths directly in `measure.m`.

---

## 6. Environment Requirements

- **MATLAB**: R2018b or later (R2025+ recommended)
- Optional toolboxes (as needed):
  - Signal Processing Toolbox
  - DSP System Toolbox (for advanced filtering if needed)

---

## 7. Quick Start Guide

### 7.1 Clone Repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 7.2 Run in MATLAB

1. Open MATLAB and set the current working directory to the repository root (or `matlab/` subdirectory)
2. Add paths:
```matlab
addpath(genpath(pwd));
```
3. Run the main evaluation script:
```matlab
measure
```

If `measure.m` requires you to configure audio/output paths, these are typically defined as variables at the script's beginning (e.g., `audio_dir`, `out_dir`). Modify these according to your directory structure.

---

## 8. Suggested Experimental Pipeline

A typical workflow (execute as needed):
```
Algorithm Evaluation (measure.m) 
    ↓
Result Format Conversion (csv2mat.m if needed)
    ↓
Global Statistics (stas_result.m)
    ↓
Data Visualization (draw.m)
```

---

## 9. Outputs and Metrics

The evaluation conducted in this repository (computed via `compute_paper_metrics.m`) produces two sets of metrics:

### 9.1 Summary Metrics (per method, per condition)
- **E_all**: Mean RMSE across all azimuth angles (global performance)
- **E_EF**: Mean RMSE in end-fire regions [20°-40°, 140°-160°] (critical region performance)
- **S_ACC_EF**: Mean Soft-Accuracy (S-ACC) in end-fire regions
  - Formula: $\text{S-ACC}(e) = \dfrac{1}{1 + (e/6)^2}$ where $e$ is angular error in degrees
  - Provides continuous accuracy metric (0 to 1) instead of hard threshold
- **Deg_Span**: Total angular span where performance degraded (S-ACC < 0.50)
  - Measured in degrees, accounts for non-uniform angle spacing

### 9.2 Per-Angle Metrics (for each azimuth angle)
- **AngleDeg**: Azimuth angle in degrees
- **RMSE**: Root Mean Square Error at this angle
- **S_ACC**: Soft-Accuracy at this angle
- **N**: Number of test samples at this angle
- **IsEndFire**: Binary flag indicating if angle is in end-fire region
- **AngleWidthDeg**: Angular width (span) represented by this angle point

### 9.3 Soft-Accuracy Metric Definition
The **S-ACC** metric provides a continuous measure of estimation accuracy:
$$\text{S-ACC}(e) = \frac{1}{1 + (e/6)^2}$$
where $e$ is the angular error in degrees.
- At $e=0°$: S-ACC = 1.0 (perfect estimation)
- At $e=6°$: S-ACC = 0.5 (half accuracy)
- At $e=12°$: S-ACC ≈ 0.07 (poor estimation)

The **Deg_Span** metric measures the total angular range where $\text{S-ACC} < 0.50$, accounting for non-uniform angle spacing.

### 9.4 Output Files
The script generates two CSV files:
- `paper_metrics_summary.csv`: Summary metrics for each method × condition
- `paper_metrics_by_angle.csv`: Detailed per-angle breakdown

---

## 10. Contact Information

**Maintainers**: WwHhYy666

**Email**: 2540160476@qq.com, 3191479712@qq.com

**Issues**: Welcome to report issues, reproduce experiments, or suggest improvements via GitHub Issues.

---