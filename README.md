# 阻尼谐振子的数值方法比较（Euler vs Velocity-Verlet-type vs RK4）

本科计算物理论文原型：比较三种时间积分方法在阻尼谐振子问题上的精度、收敛阶、色散（相位）误差、包络衰减误差、计算成本与大步长稳定性。

## 1. 模型

无量纲阻尼谐振子方程：

```
x'' + 2ζx' + x = 0,   x(0) = 1, v(0) = 0
```

其中 ζ 为阻尼比。三种阻尼情形：

| ζ |  regime  | 解析解 x(t) |
|---|----------|-------------|
| 0.1 | 欠阻尼 | e^(−ζt)·[cos(ω_d t) + (ζ/ω_d)·sin(ω_d t)]，ω_d = √(1−ζ²) |
| 1 | 临界阻尼 | (1+t)·e^(−t) |
| 2 | 过阻尼 | [(ζ+s)e^(r₁t) − (ζ−s)e^(r₂t)]/(2s)，s = √(ζ²−1)，r₁,₂ = −ζ±s |

（`model.py`，含自检：ζ=0 退化为 cos(t)；初值；v 为 x 的导数；过阻尼长时尾部跟随慢根。）

## 2. 数值方法

记加速度 `a(x, v) = −x − 2ζv`。统一接口 `solve(zeta, h, t_end) -> (t, x, v)`（`solvers.py`）。

- **Euler（显式）**：一阶
  ```
  x_{n+1} = x_n + h·v_n
  v_{n+1} = v_n + h·a(x_n, v_n)
  ```
- **Velocity-Verlet-type（二阶，阻尼修正版）** ⚠️ 见下文局限说明
  ```
  a_n     = −x_n − 2ζ·v_n
  x_{n+1} = x_n + h·v_n + ½h²·a_n
  v_pred  = v_n + h·a_n
  a_{n+1} = −x_{n+1} − 2ζ·v_pred
  v_{n+1} = v_n + ½h·(a_n + a_{n+1})
  ```
- **RK4（经典四阶）**：对状态向量 y = (x, v) 的标准四阶 Runge-Kutta。

### 关于 velocity-Verlet-type 的说明与局限（重要）

标准 velocity-Verlet 是为**加速度仅依赖位置**（a = a(x)，保守系统）推导的；它在无阻尼谐振子上有辛性质、长期能量行为好。本问题的阻尼项含速度（a = a(x, v)），**不能直接套用标准 Verlet**。此处采用的变体用显式预测 `v_pred` 来计算 `a_{n+1}` 中的阻尼项，因此：

1. 它**不是**辛算法，不继承标准 Verlet 的长期能量守恒特性；
2. 预测步对 v 只是一阶的，整体格式仍为二阶，但误差常数比保守系统的真 Verlet 大；
3. 本文中统一标注为 **"velocity-Verlet-type scheme for velocity-dependent damping"**，不简称为 Verlet。

## 3. 实验矩阵

- ζ ∈ {0.1, 1, 2}（欠阻尼 / 临界 / 过阻尼）
- h ∈ {0.2, 0.1, 0.05, 0.025, 0.0125}，t ∈ [0, 20]
- 3 方法 × 3 阻尼 × 5 步长 = 45 组，每组另测 CPU 时间（3 次取最小）

## 4. 误差指标（`analysis.py`）

- **E_inf** = max|x_n − x(t_n)|：最大位移误差（主指标）
- **E_RMS**：均方根误差
- **相位漂移**（欠阻尼）：数值峰值时刻相对解析峰值 t_k = kT（T = 2π/ω_d）的漂移；对峰序号 k 做线性拟合，斜率即每周期漂移（秒/周期）——色散误差的干净度量
- **包络衰减误差**（欠阻尼）：对数值峰值包络做 ln A_k = ln A₀ − ζ_fit·t_k 拟合，报告 |ζ_fit − ζ|/ζ
- **CPU 时间**：`time.perf_counter` 计时

## 5. 如何复现

```bash
pip install numpy matplotlib
python3 model.py      # 解析解自检
python3 solvers.py    # 求解器自检
python3 analysis.py   # 收敛阶速查（期望斜率 ≈ 1 / 2 / 4）
python3 experiments/run_all.py  # 跑完整 45 组实验，保存 data/*.csv
python3 plots.py                # 当前绘图入口；CSV-only 入口将在下一提交加入
```

只用 numpy/matplotlib；未用 scipy 的 ODE 求解器（解析解为手写闭式）。

## 6. 主要结论（2026-09-29 实测）

- **收敛阶**（E_inf 的 log-log 斜率）：Euler ≈ 1.0–1.3，Verlet-type ≈ 2.0，RK4 ≈ 4.0–4.2，全部符合理论预期。ζ=0.1 时 Euler 为 1.27，系 h=0.2 尚未进入渐近区（见下条）。
- **Euler 在欠阻尼 h=0.2 已 marginal 不稳定**：线性稳定性要求 |1+hλ|<1，对应 h<0.2；h=0.2 时数值解振幅不衰减反增，包络拟合给出 100% 衰减率误差。h=0.5 时彻底发散（见 fig6）。
- **色散误差**：Euler 每周期快约 0.1·h 秒（一阶）；Verlet-type 与 RK4 的相位漂移在 h≤0.1 时已低于峰值检测噪声 floor，可忽略。
- **成本-精度**：同等 CPU 下 RK4 精度高 3–6 个量级；Verlet-type 是"性价比"中点；Euler 只在极粗精度要求下可考虑。
- **临界/过阻尼**：三种方法行为单调干净，无振荡相关病态；RK4 在 h=0.0125 时 E_inf ~ 1e−9–1e−10，已接近双精度 floor。

## 7. 文件结构

```
model.py        解析解 + 自检
solvers.py      Euler / velocity-Verlet-type / RK4 + 自检
analysis.py     误差指标、收敛阶、CPU 计时、全矩阵实验
experiments/run_all.py  保存完整实验矩阵及原有绘图输入
data/           CSV 数据、列定义与运行环境（见 data/README.md）
plots.py        生成 paper_figures/ 6 张图
paper_figures/  fig1–fig6（见下）
README.md       本文件
```

| 图 | 内容 |
|---|---|
| fig1 | 欠阻尼轨道对比（h=0.2，含首峰放大） |
| fig2 | 临界阻尼轨道对比 |
| fig3 | 过阻尼轨道对比 |
| fig4 | E_inf vs h log-log，三阻尼分栏，含 1/2/4 阶参考线 |
| fig5 | CPU 时间 vs E_inf（成本-精度） |
| fig6 | h=0.5 大步长稳定性：Euler 发散，另两者有界但失真 |

数据列定义及既有相位指标实现的说明见 [data/README.md](data/README.md)。
