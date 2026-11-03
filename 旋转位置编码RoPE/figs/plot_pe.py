# 绘制正弦位置编码的两张直观图（曲线 + 热图），风格对齐笔记仓库
# 运行: python3 plot_pe.py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

FIGS = '/Users/songlulu/Documents/github/Deep Learning Notes/旋转位置编码RoPE/figs'
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
speed = ['1 弧度/步\n约 6.3 步转一圈', '≈0.50 弧度/步\n约 13 步一圈', '≈0.25 弧度/步\n约 25 步一圈',
         '≈0.063 弧度/步\n约 100 步一圈', '≈0.004 弧度/步\n约 1600 步一圈']
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

# ---------- 图 2：位置编码矩阵热图 ----------
fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=300)
full = np.zeros((L, d))
full[:, 0::2] = P_sin
full[:, 1::2] = P_cos
im = ax.imshow(full.T, aspect='auto', cmap='Blues', origin='lower',
               vmin=-1, vmax=1, interpolation='nearest')
ax.set_xlabel('位置 i（token 序号）→', fontsize=9, color=INK)
ax.set_ylabel('编码维度（2j 填 sin，2j+1 填 cos）→', fontsize=9, color=INK)
ax.set_title('位置编码矩阵热图：底部条纹密（高频），顶部渐变平缓（低频）',
             fontsize=10.5, color=INK, pad=10)
ax.set_xticks([0, 20, 40, 59])
ax.set_xticklabels(['0', '20', '40', '59'], fontsize=8, color=INK)
yt = [0, 2, 4, 8, 16, 24, 31]
ax.set_yticks(yt)
ax.set_yticklabels([str(t) for t in yt], fontsize=8, color=INK)
for s in ax.spines.values():
    s.set_color('#9A9894')
cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cbar.ax.tick_params(labelsize=7.5, colors=INK)
cbar.set_label('编码值（-1 ~ 1）', fontsize=8, color=INK)
# 标注两个区域：低维在底部（origin='lower'），高维在顶部
ax.annotate('低维（j 小）：竖条纹密集——\n相邻位置的颜色也明显不同', xy=(22, 3.5), fontsize=8,
            color=INK, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=AMBER, lw=0.8, alpha=0.92))
ax.annotate('高维（j 大）：沿位置几乎不变——\n长距离也不混淆', xy=(22, 27.5), fontsize=8,
            color=INK, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=GREEN, lw=0.8, alpha=0.92))
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

print('saved: fig_pe_waves.png, fig_pe_heatmap.png, fig_pe_delta.png')
