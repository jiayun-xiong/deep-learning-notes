# 配图规范（SVG）

## 画布与字体

- 固定宽度 `viewBox="0 0 680 H"`，高度按内容给，坐标全部用绝对像素。
- 字体统一 `font-family="PingFang SC, sans-serif"`（图内英文标签也用这套，保证字形一致）。
- 字号：图标题 14 / 副标题 11.5 / 正文块 12 / 注释 10.5。
- 顶部留白：标题 y=26，副标题 y=46，图形从 y≈64 开始。

## 配色（沿用仓库既有图）

| 语义 | 浅底 | 描边 | 深字 |
| --- | --- | --- | --- |
| 编码器 / 蓝色系 | `#E6F1FB` | `#185FA5` | `#0C447C` |
| 解码器 / 紫色系 | `#EEEDFE` | `#534AB7` | `#3C3489` |
| 中性块 / 容器 | `#F1EFE8` | `#888780` | `#2C2C2A` |
| 强调 / 绿色系 | `#9FE1CB` | `#1D9E75` | `#04342C` |
| 正文灰 | — | — | `#5F5E5A` |

方块：`rx="8"`，`stroke-width="0.5"`。箭头：统一 marker，`stroke="#888780" stroke-width="1.2"`。

## 图内公式：必须转 path

用 `<frag>` 占位，再交给工具链展开，禁止用 `<text>` 直接写公式。

```xml
<frag x="170" y="120" size="13" color="#2C2C2A" anchor="middle" tex="\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V"/>
```

属性只允许 `x y size color anchor tex`（anchor 取 `start|middle|right`，默认 start）。

构建命令（在仓库根目录执行）：

```bash
node .tools/latex2svg/figbuild.mjs <笔记目录>/figs/xxx.svg
```

该命令会：把 `<frag>` 替换为 MathJax 的 path 片段并**原地写回 SVG**，同时在同目录生成 3x 白底 PNG（resvg，`PingFang SC`）。

## 检查流程

1. 打开生成的 `.png`（3x，字体正确），逐张看：文字重叠、溢出方框、超出画布、箭头是否指到正确的框、左右栏是否对齐。
2. 不用 cairosvg 做终检：它缺中文字形，会把汉字渲染成方块，容易误判。
3. 若某张图需改动，改 SVG 后重新跑 figbuild；已展开的图没有 `<frag>`，改动后仍可重跑。

## 布局原则

- 严格照参考论文的图的布局与元素命名（如 Vaswani 2017 Figure 1 的左右两栏 + `N×` 堆叠）。
- 图内标签只用英文；中文放图注。
- 矩阵类图：**行 = 位置 $i$，列 = 维度 $j$**，并在图注里重申一次方向。
- 信息流方向统一：自上而下（架构）或自左而右（流程/对比）；同一篇内不混用。
- 图注格式：`**图 N**：<一句说明>。<必要时一句关键提示>。`
