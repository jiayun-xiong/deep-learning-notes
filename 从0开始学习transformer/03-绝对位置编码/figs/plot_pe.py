# 绘制正弦位置编码的两张直观图（曲线 + 热图），风格对齐笔记仓库
# 运行: python3 plot_pe.py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

import os
FIGS = os.path.dirname(os.path.abspath(__file__))
INK = '#2C2C2A'
AMBER = '#BA7517'
BLUE = '#185FA5'
GREEN = '#639922'
RED = '#B03A2E'
PURPLE = '#7D3C98'

# P[i, 2j] = sin(i / 10000^(2j/d))，与 notebook 的 PositionalEncoding 一致
d, L = 32, 60
i = np.arange(L)[:, None]                       # (L, 1) 位置
j = np.arange(d // 2)[None, :]                  # (1, d/2) 维度对
theta = 10000.0 ** (-2 * j / d)                 # (1, d/2) 每对的角频率
P_sin = np.sin(i * theta)                       # (L, d/2)
P_cos = np.cos(i * theta)

# ---------- 图 1：不同维度对 j 的正弦/余弦曲线（1x5 小面板） ----------
sel = [0, 1, 2, 4, 7]                            # 低 -> 高 j
def _speed(jj):
    th = 10000.0 ** (-2 * jj / d)          # theta_j，单位：弧度/步
    turn = 2 * np.pi / th                   # 转一圈需要的步数
    th_s = '1' if jj == 0 else f'{th:.2f}'.rstrip('0').rstrip('.')
    return f'{th_s} 弧度/步\n约 {turn:.0f} 步一圈'

speed = [_speed(jj) for jj in sel]
fig, axes = plt.subplots(1, 5, figsize=(6.8, 2.9), dpi=300, sharey=True)
x = np.arange(L)
for k, (ax, jj) in enumerate(zip(axes, sel)):
    c = [RED, AMBER, GREEN, BLUE, PURPLE][k]
    ax.plot(x, P_sin[:, jj], color=c, lw=1.4, label='sin')
    ax.plot(x, P_cos[:, jj], color=c, lw=1.0, ls='--', alpha=0.55, label='cos')
    ax.set_title(f'j={jj}', fontsize=10, color=c, pad=4)
    ax.text(0.5, -0.24, speed[k], transform=ax.transAxes, fontsize=7,
            color=INK, ha='center', va='top')
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlim(0, L - 1)
    ax.grid(True, lw=0.35, alpha=0.3)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#9A9894')
    ax.tick_params(labelsize=7, colors=INK)
    if k == 0:
        ax.set_ylabel('编码值', fontsize=9, color=INK)
        ax.legend(fontsize=7, frameon=False, loc='upper right')
    else:
        ax.set_yticklabels([])
axes[2].set_xlabel('位置 i（token 序号）', fontsize=9, color=INK, labelpad=26)
fig.suptitle('不同维度对 j 的正弦 / 余弦曲线：j 越大，周期越长、频率越低',
             fontsize=11, color=INK, y=1.02)
fig.tight_layout()
fig.savefig(f'{FIGS}/fig_pe_waves.png', facecolor='white', bbox_inches='tight')
plt.close(fig)

# ---------- 图 2：位置编码矩阵热图（行 = 位置 i，列 = 维度 j，与矩阵 P 的 (L, d) 一致） ----------
fig, ax = plt.subplots(figsize=(5.6, 5.9), dpi=300)
full = np.zeros((L, d))
full[:, 0::2] = P_sin
full[:, 1::2] = P_cos
im = ax.imshow(full, aspect='auto', cmap='Blues', origin='lower',
               vmin=-1, vmax=1, interpolation='nearest')
ax.set_xlabel('维度（2j 填 sin，2j+1 填 cos）→  j 增大', fontsize=9, color=INK)
ax.set_ylabel('位置 i（token 序号，向上递增）→', fontsize=9, color=INK)
ax.set_title('位置 i 为行、维度 j 为列：左边那几列几步一翻，右边几乎一条色带',
             fontsize=9.6, color=INK, pad=10)
ax.set_xticks([0, 8, 16, 24, 31])
ax.set_xticklabels(['0', '8', '16', '24', '31'], fontsize=8, color=INK)
ax.set_yticks([0, 20, 40, 59])
ax.set_yticklabels(['0', '20', '40', '59'], fontsize=8, color=INK)
for sp in ax.spines.values():
    sp.set_color('#9A9894')
cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cbar.ax.tick_params(labelsize=7.5, colors=INK)
cbar.set_label('编码值（-1 ~ 1）', fontsize=8, color=INK)

ax.annotate('j 小：上下几步就翻一次色\n相邻两个位置立刻分得开',
            xy=(3, 40), xytext=(11.5, 50), fontsize=7.8, color=INK,
            ha='center', va='center',
            arrowprops=dict(arrowstyle='-|>', lw=0.8, color=AMBER),
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=AMBER, lw=0.8, alpha=0.93))
ax.annotate('j 大：整列几乎一个色\n要隔很远才拉得开',
            xy=(29, 12), xytext=(20.5, 22), fontsize=7.8, color=INK,
            ha='center', va='center',
            arrowprops=dict(arrowstyle='-|>', lw=0.8, color=GREEN),
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=GREEN, lw=0.8, alpha=0.93))
fig.tight_layout()
fig.savefig(f'{FIGS}/fig_pe_heatmap.png', facecolor='white', bbox_inches='tight')
plt.close(fig)

# ---------- 图 3：偏移 Δ 上各转速指针的相位（cos(Δθ)） ----------
fig, ax = plt.subplots(figsize=(6.8, 3.4), dpi=300)
D = np.linspace(0, 50, 2000)
for th, c, lab in [(1.0, RED, 'θ = 1（快针，j 小）'),
                   (0.1, AMBER, 'θ = 0.1（中速）'),
                   (0.02, BLUE, 'θ = 0.02（慢针，j 大）')]:
    ax.plot(D, np.cos(D * th), color=c, lw=1.7, label=lab)

ax.axvspan(0, 6, color='#378ADD', alpha=0.07, lw=0)
ax.axvspan(40, 50, color='#639922', alpha=0.08, lw=0)
ax.text(3, 1.12, 'Δi 小：快针已在变', fontsize=8, color=BLUE, ha='center',
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.9))
ax.text(45, 1.12, 'Δi 大：快针转花，慢针才展开', fontsize=8, color='#3B6D11', ha='center',
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.9))

ax.set_xlabel('位置差 Δi = i′ − i（两个 token 隔多少步）', fontsize=9, color=INK)
ax.set_ylabel('cos(Δi·θ)', fontsize=9, color=INK)
ax.set_title('同一批指针，转速不同：同一个 Δi 转出的角度各不相同', fontsize=10.5, color=INK, pad=10)
ax.set_xlim(0, 50)
ax.set_ylim(-1.25, 1.3)
ax.grid(True, lw=0.35, alpha=0.28)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
for s in ('left', 'bottom'):
    ax.spines[s].set_color('#9A9894')
ax.tick_params(labelsize=8, colors=INK)
ax.legend(fontsize=8, frameon=True, framealpha=0.92, edgecolor='none', loc='lower right')
fig.tight_layout()
fig.savefig(f'{FIGS}/fig_pe_delta.png', facecolor='white', bbox_inches='tight')
plt.close(fig)

# ---------- 图 4：唯一 ≠ 能当位置编码（正弦矩阵 vs 任意单射） ----------
# 方向与图 2 一致：行 = 位置 i，列 = 维度
d4, L4 = 24, 60
ii = np.arange(L4)[:, None]                      # (L4, 1)
jj = np.arange(d4 // 2)[None, :]                 # (1, d4/2)
th4 = 10000.0 ** (-2 * jj / d4)

Psin = np.empty((L4, d4))
Psin[:, 0::2] = np.sin(ii * th4)
Psin[:, 1::2] = np.cos(ii * th4)

jjf = np.arange(d4)[None, :]
F = ii * d4 + jjf                                # f(i,j) = i*M + j, M = d4，显然单射
Fn = 2 * (F - F.min()) / (F.max() - F.min()) - 1  # 拉到 [-1,1]，与左图同标尺比较

fig, axes = plt.subplots(1, 2, figsize=(6.8, 4.2), dpi=300)
panels = [
    (axes[0], Psin, '正弦编码：一行是一组读数，一列是一条正弦'),
    (axes[1], Fn,   'f(i,j) = i·M + j：同样唯一，只剩一个斜坡'),
]
for ax, mat, ttl in panels:
    im = ax.imshow(mat, aspect='auto', cmap='Blues', origin='lower',
                   vmin=-1, vmax=1, interpolation='nearest')
    ax.set_title(ttl, fontsize=8.4, color=INK, pad=6)
    ax.set_xlabel('维度 →', fontsize=8.5, color=INK)
    ax.set_ylabel('位置 i →', fontsize=8.5, color=INK)
    ax.set_xticks([0, 7, 15, 23])
    ax.set_xticklabels(['0', '7', '15', '23'], fontsize=7, color=INK)
    ax.set_yticks([0, 15, 30, 45, 59])
    ax.set_yticklabels(['0', '15', '30', '45', '59'], fontsize=7, color=INK)
    for sp in ax.spines.values():
        sp.set_color('#9A9894')

axes[0].annotate('往右的列上下越平缓：\n每右移一档，转得慢一截',
                 xy=(17, 34), xytext=(11, 50), fontsize=7.2, color=INK,
                 ha='center', va='center',
                 arrowprops=dict(arrowstyle='-|>', lw=0.7, color=AMBER),
                 bbox=dict(boxstyle='round,pad=0.26', fc='white', ec=AMBER, lw=0.7, alpha=0.93))
axes[0].annotate('固定一列往下走\n= 一条正弦曲线',
                 xy=(2, 26), xytext=(6.5, 10), fontsize=7.2, color=INK,
                 ha='center', va='center',
                 arrowprops=dict(arrowstyle='-|>', lw=0.7, color=BLUE),
                 bbox=dict(boxstyle='round,pad=0.26', fc='white', ec=BLUE, lw=0.7, alpha=0.93))

axes[1].annotate('一行之内几乎同色：\n维度分不出差别',
                 xy=(11, 40), xytext=(11, 22), fontsize=7.2, color=INK,
                 ha='center', va='center',
                 arrowprops=dict(arrowstyle='-|>', lw=0.7, color=RED),
                 bbox=dict(boxstyle='round,pad=0.26', fc='white', ec=RED, lw=0.7, alpha=0.93))
axes[1].annotate('行与行只差一个常数：\n没有周期、没有尺度',
                 xy=(20, 14), xytext=(12, 52), fontsize=7.2, color=INK,
                 ha='center', va='center',
                 arrowprops=dict(arrowstyle='-|>', lw=0.7, color=RED),
                 bbox=dict(boxstyle='round,pad=0.26', fc='white', ec=RED, lw=0.7, alpha=0.93))

cbar = fig.colorbar(im, ax=axes, fraction=0.022, pad=0.02)
cbar.ax.tick_params(labelsize=7, colors=INK)
cbar.set_label('取值（已归一化到 -1 ~ 1）', fontsize=7.5, color=INK)
fig.savefig(f'{FIGS}/fig_pe_why.png', facecolor='white', bbox_inches='tight')
plt.close(fig)

# ---------- 图 5：单维会混叠——同一对位置，两根针的看法不同 ----------
# 取 d=512 的前两维：theta0=1, theta1=10000^(-2/512)
th0, th1 = 1.0, 10000.0 ** (-2 / 512)
N = 46
xx = np.arange(N)
DI = 44

fig, ax = plt.subplots(figsize=(6.8, 3.5), dpi=300)
ax.plot(xx, np.sin(xx * th0), color=BLUE, lw=1.5, label='j=0（θ=1，最快）')
ax.plot(xx, np.sin(xx * th1), color=AMBER, lw=1.5, label='j=1（θ≈0.96）')
ax.axvline(0, color='#9A9894', lw=0.7, ls=':')
ax.axvline(DI, color='#9A9894', lw=0.7, ls=':')

v0 = np.sin(DI * th0)
v1 = np.sin(DI * th1)
ax.plot([DI], [v0], 'o', ms=6.5, color=BLUE, zorder=5)
ax.plot([DI], [v1], 'o', ms=6.5, color=AMBER, zorder=5)

ax.annotate(f'第 0 维回到 {v0:.3f}，\n几乎就是起点 0 → 撞车',
            xy=(DI, v0), xytext=(31, 0.62), fontsize=8, color=INK, ha='center', va='center',
            arrowprops=dict(arrowstyle='-|>', lw=0.8, color=BLUE),
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=BLUE, lw=0.8, alpha=0.93))
ax.annotate(f'第 1 维停在 {v1:.2f}，\n与起点毫无关系 → 分得开',
            xy=(DI, v1), xytext=(31, -1.02), fontsize=8, color=INK, ha='center', va='center',
            arrowprops=dict(arrowstyle='-|>', lw=0.8, color=AMBER),
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=AMBER, lw=0.8, alpha=0.93))
ax.annotate('i=0：两维\n都从 0 出发', xy=(0, 0), xytext=(6.5, -0.75), fontsize=8, color=INK,
            ha='center', va='center',
            arrowprops=dict(arrowstyle='-|>', lw=0.8, color='#6B6A66'),
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#9A9894', lw=0.7, alpha=0.9))

ax.set_xlabel('位置 i', fontsize=9, color=INK)
ax.set_ylabel('sin(i·θ)', fontsize=9, color=INK)
ax.set_title('同一对位置 i=0 与 i=44：第 0 维几乎转回原点，第 1 维没有', fontsize=10, color=INK, pad=9)
ax.set_xlim(-2, N - 1)
ax.set_ylim(-1.28, 1.28)
ax.grid(True, lw=0.35, alpha=0.28)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    ax.spines[sp].set_color('#9A9894')
ax.tick_params(labelsize=8, colors=INK)
ax.legend(fontsize=7.8, frameon=True, framealpha=0.92, edgecolor='none', loc='lower left', ncol=2)
fig.tight_layout()
fig.savefig(f'{FIGS}/fig_pe_alias.png', facecolor='white', bbox_inches='tight')
plt.close(fig)

print('saved: fig_pe_waves.png, fig_pe_heatmap.png, fig_pe_delta.png, fig_pe_why.png, fig_pe_alias.png')
