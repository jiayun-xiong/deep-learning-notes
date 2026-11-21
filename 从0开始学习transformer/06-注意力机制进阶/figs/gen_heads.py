#!/usr/bin/env python3
# 生成图 7：8 个头的注意力模式（分工示意图）
# 用法：python3 gen_heads.py > fig_7_heads.svg
# 说明：模式取自真实可视化里常见的专化头类型，属示意图，非某一模型的实测矩阵。

CELL = 8
N = 11
PANEL_W = 196
PANEL_H = 144
GRID_X = (PANEL_W - N * CELL) / 2
GRID_Y = 28

TOKENS = ["科", "软", "放", "实", "习", "吸", "引", "学", "生", "报", "考"]


def w_head1(i, j):  # 看前一个词
    if j == i - 1:
        return 0.85
    if j == i:
        return 0.25
    return 0.0


def w_head2(i, j):  # 看自己
    return 0.9 if j == i else 0.0


def w_head3(i, j):  # 句首 sink
    if j == 1:
        return 0.9
    return 0.2 if j == i and i > 1 else 0.0


def w_head4(i, j):  # 局部窗口
    if j == i:
        return 0.6
    if j == i - 1:
        return 0.45
    if j == i - 2:
        return 0.2
    return 0.0


def w_head5(i, j):  # 盯「实习」(4,5)
    return 0.8 if j in (4, 5) else 0.0


def w_head6(i, j):  # 盯「学生」(9,10)
    return 0.8 if j in (9, 10) else 0.0


HEADS = [
    (w_head1, "头 1：盯前一个词", "拿到的是「上一个 token 是谁」"),
    (w_head2, "头 2：盯自己", "保留原位信息，相当于一条近路"),
    (w_head3, "头 3：盯句首", "attention sink：存放无处可去的权重"),
    (w_head4, "头 4：盯局部窗口", "负责构词、固定搭配这类近距离关系"),
    (w_head5, "头 5：盯「实习」", "把一个具体实体当作信息源"),
    (w_head6, "头 6：盯「学生」", "另一个头盯另一个实体，互不干扰"),
]


def color(w):
    if w == 0:
        return "#F7F6F2"
    if w <= 0.25:
        return "#DCEAF7"
    if w <= 0.5:
        return "#9CC3E8"
    if w <= 0.7:
        return "#4C8FCB"
    return "#185FA5"


out = []
out.append(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 404" width="680" height="404" role="img">'
)
out.append("<title>多头注意力的分工</title>")
out.append(
    "<desc>同一个输入、同一张 11×11 的分数矩阵，8 个头填出 8 种不同模式：盯前一个词、盯自己、盯句首、盯局部窗口、盯某个实体。</desc>"
)
out.append(
    '<text x="24" y="26" font-family="PingFang SC, sans-serif" font-size="13.5" font-weight="500" fill="#2C2C2A">多头：同一张 11×11 矩阵，每个头填出一种模式</text>'
)
out.append(
    '<text x="24" y="46" font-family="PingFang SC, sans-serif" font-size="11.5" fill="#5F5E5A">行 = 发起查询的位置，列 = 被看的位置；灰色格是因果掩码挡掉的（看不到未来）</text>'
)

positions = [(34, 60), (242, 60), (450, 60), (34, 212), (242, 212), (450, 212)]

for idx, ((px, py), (fn, title, note)) in enumerate(zip(positions, HEADS)):
    out.append(
        f'<rect x="{px}" y="{py}" width="{PANEL_W}" height="{PANEL_H}" rx="8" fill="#FFFFFF" stroke="#888780" stroke-width="0.5"/>'
    )
    out.append(
        f'<text x="{px + PANEL_W / 2}" y="{py + 18}" text-anchor="middle" font-family="PingFang SC, sans-serif" font-size="10.5" font-weight="500" fill="#2C2C2A">{title}</text>'
    )
    gx, gy = px + GRID_X, py + GRID_Y
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            x = gx + (j - 1) * CELL
            y = gy + (i - 1) * CELL
            if j > i:
                fill = "#EDEBE5"
            else:
                fill = color(fn(i, j))
            out.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{CELL}" height="{CELL}" fill="{fill}" stroke="#FFFFFF" stroke-width="0.6"/>'
            )
    out.append(
        f'<rect x="{gx:.1f}" y="{gy:.1f}" width="{N * CELL}" height="{N * CELL}" fill="none" stroke="#888780" stroke-width="0.5"/>'
    )
    out.append(
        f'<text x="{px + PANEL_W / 2}" y="{py + 136}" text-anchor="middle" font-family="PingFang SC, sans-serif" font-size="9" fill="#5F5E5A">{note}</text>'
    )

out.append(
    '<text x="24" y="378" font-family="PingFang SC, sans-serif" font-size="10.5" fill="#2C2C2A">六个面板没有一行相同：一侧深色落在次对角线，一侧整列发亮，一侧只亮两格——这些模式没有一个被指定，全是训练长出来的。</text>'
)
out.append(
    '<text x="24" y="396" font-family="PingFang SC, sans-serif" font-size="9.5" fill="#888780">示意图：模式取自真实可视化里常见的几类专化头（位置头、sink 头、实体头），这里取 6 个展示，不是某个模型的实测矩阵。</text>'
)
out.append("</svg>")

print("\n".join(out))
