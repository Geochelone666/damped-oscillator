# 阻尼谐振子的数值方法比较（Euler vs Velocity-Verlet-type vs RK4）

拿阻尼谐振子当试验场，比较三种时间积分方法的精度、收敛阶、相位误差、包络衰减误差、计算成本和大步长稳定性。跟我的 RLC 电路仓是姊妹篇，数学上很多相通，互相验算过。

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

1. 在有阻尼时它**不是**辛算法；ζ=0 时退化为标准 velocity Verlet，能量误差有界；
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

### 无阻尼极限 (zeta = 0)

无阻尼时，解析解为 `x=cos(t)`、`v=-sin(t)`，周期为 `T=2π`。
真实包络平坦，高精度数值解的 `zeta_fit` 应接近零，允许很小的数值残差。
`envelope_rel_err` 恒为 `NaN`，不会为 `inf`，因为对零衰减率的相对误差无定义。
`phase_err` 仍表示相对 `T=2π` 的每周期峰值漂移；相位和包络拟合都需至少两个可用峰，否则为 `NaN`。
显式 Euler 每步将幅值乘以 `sqrt(1+h²)≈1+h²/2`，能量则乘以 `1+h²`，因此拟合得到负的衰减率。
Euler 长区间误差由能量增长主导，这是方法的固有性质；短区间收敛阶仍为一阶，Verlet 为二阶，RK4 为四阶。

## 5. 如何复现

从仓库根目录运行；先创建虚拟环境（避免系统 Python 的 externally-managed 限制）：

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt   # Python 3.11–3.13
python3 model.py      # 解析解自检
python3 solvers.py    # 求解器自检
python3 analysis.py   # 收敛阶速查（期望斜率 ≈ 1 / 2 / 4）
python3 experiments/run_all.py  # 跑完整 45 组实验，保存 data/*.csv
python3 experiments/stability_scan.py # ζ=0.1，h=0.2 附近的独立稳定性扫描
python3 experiments/stability_theory.py # 理论稳定边界 vs 求解器实测（stability.py）
python3 experiments/energy_drift.py # ζ=0，长时间能量漂移实验（t≤100）
python3 figures/make_figures.py # 仅读取 CSV，生成 paper_figures/ 下 8 张图
python3 plots.py                # 兼容旧命令，同样仅读取 CSV
```

只用 numpy/matplotlib；未用 scipy 的 ODE 求解器（解析解为手写闭式）。

## 6. 主要结论（2026-09-29 实测）

- **收敛阶**（E_inf 的 log-log 斜率）：Euler ≈ 1.0–1.3，Verlet-type ≈ 2.0–2.2，RK4 ≈ 4.0–4.2，全部符合理论预期（粗步长会影响拟合）。ζ=0.1 时 Euler 为 1.27，系 h=0.2 尚未进入渐近区（见下条）。
- **Euler 在欠阻尼 h=0.2 处于中性稳定边界**：线性稳定性要求 |1+hλ|<1，对应 h<0.2；h=0.2 时数值包络既不衰减也不增长，包络拟合给出 100% 衰减率误差。h=0.5 时彻底发散（见 fig6）。
- **理论稳定边界（stability.py）**：三种方法对模型方程都是线性的，单步即 2×2 放大矩阵 M，长时间行为由谱半径 ρ(M) 决定。`stability.py` 从文档化的单步递推直接推导 M（测试保证 M^k 与求解器循环逐位一致），`experiments/stability_theory.py` 用求解器实测验证预测边界（0.95h* 有界、1.05h* 发散，见 data/stability_boundaries.csv）。结论：Euler 边界恰为 h=2ζ（0<ζ≤1），ζ=0 时每步能量乘子恰为 1+h²；Verlet-type 与 RK4 的边界**不是** ζ 的单调函数——Verlet-type 在 ζ≈0.3 处降到约 1.73，ζ=0.5 时回到 2.0；RK4 在 ζ=0.1 处升到约 2.95，ζ=0.5 时降到约 2.62。ζ=0 时 RK4 每步能量乘子 det(M)=1−h⁶/72+h⁸/576，与 data/README.md 引用的级数一致。
- **色散误差**：相位指标返回峰值时刻漂移的拟合斜率（秒/周期）；例如 ζ=0.1、h=0.2 时 Euler / Verlet-type / RK4 约为 −0.042 / −0.0097 / +0.00031。区间内仅有三个正峰，小漂移受峰值插值误差影响，不能据此断言可忽略。`stability.period_drift` 给出同一量的理论值（放大矩阵特征值角度推出数值频率），与实测对拍见 data/dispersion_check.csv：Euler 吻合到约 1e-6，Verlet-type 约 1e-4，RK4 预测值低于峰值拟合噪声下限。顺带解释了 metrics.csv 中 Euler 在 h=0.1 与 h=0.2 的 phase_slope 完全相同并非脏数据：h=ζ 与 h=2ζ 处数值频率有精确恒等式。
- **成本-精度**：同等 CPU 下 RK4 精度高 3–6 个量级；Verlet-type 是"性价比"中点；Euler 只在极粗精度要求下可考虑。
- **临界/过阻尼**：三种方法行为单调干净，无振荡相关病态；RK4 在 h=0.0125 时 E_inf ~ 1e−9–1e−10，仍明显高于双精度机器精度；这些步长未显示舍入误差平台。

## 7. 文件结构

```
model.py        解析解 + 自检
solvers.py      Euler / velocity-Verlet-type / RK4 + 自检
analysis.py     误差指标、收敛阶、CPU 计时、全矩阵实验
stability.py    单步放大矩阵、谱半径与稳定边界（理论）+ 自检
experiments/run_all.py  保存完整实验矩阵及原有绘图输入
data/           CSV 数据、列定义与运行环境（见 data/README.md）
figures/make_figures.py  仅读取 data/*.csv，生成 8 张图
plots.py        向后兼容的薄封装（调用 figures/make_figures.py）
paper_figures/  fig1–fig8（见下）
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
| fig7 | ζ=0.1：包络衰减误差 vs h，标出理论边界 h=2ζ=0.2 |
| fig8 | ζ=0，h=0.1：相对能量误差 vs t（对数纵轴，t≤100） |

数据列定义及既有相位指标实现的说明见 [data/README.md](data/README.md)。

图像生成不再运行求解器；可直接使用仓库中的 CSV 重绘。重新运行实验后，CPU 计时会变化，因此 fig5 随测量结果变化。原有 fig6 的 h=0.5 轨道一并存入 CSV，新增 fig7 的稳定性扫描由 `experiments/stability_scan.py` 独立生成（ζ=0.1，t≤40）。

## 8. 自动回归检查

GitHub Actions 在推送到 `main` 和提交 PR 时，使用 Python 3.11、3.12、3.13 安装 `requirements.txt` 后执行 unittest 测试、解析解和求解器自检，并从已保存的 CSV 重绘图像（检查绘图脚本可用）。测试不重新生成实验数据，也不对机器相关的 CPU 性能设定阈值。

在仓库根目录复现同一测试命令：

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
python3 model.py
python3 solvers.py
```

stability.py 的理论模块由 tests/test_stability.py 覆盖（含 M^k 与求解器循环的逐步一致性）；另可单独运行 `python3 stability.py` 做快速自检。

```bash
python3 stability.py
```
