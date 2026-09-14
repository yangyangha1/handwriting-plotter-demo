# Algorithms v0.2

## 1. OCR-guided character segmentation

PaddleOCR 给出文本 `c1...cn` 和文本框。对横排框计算垂直墨迹投影，对竖排框计算水平墨迹投影。第 i 个分割位置以等分位置为先验，只在其邻域内寻找低墨迹谷值：

```text
cut_i = argmin_x [projection(x) + α |x - expected_i|]
```

这样比纯等宽切对字间距变化更稳，同时不会使用 connected components 把一个汉字的多个离散笔画错误拆开。

## 2. Canonical stroke prior

OCR 审核后的字符用于查询 Make Me A Hanzi。标准 median 已按规范笔顺排列，因此静态图不承担“猜笔顺”的任务。

## 3. Skeleton constrained fitting

手写 crop -> Otsu/预处理 -> tight crop -> square normalization -> skeleton。

对标准轨迹点 p：

```text
p <- p
     - λd normalize(∇D(p))
     + λs Laplacian(p)
     + λa (p0 - p)
```

- `D`: 到 skeleton 的距离场。
- attraction: 靠近真实手写中心。
- Laplacian: 抑制锯齿。
- anchor: 防止交叉处串笔。

## 4. Global writer profile

标准笔画 a(t) 与拟合笔画 b(t) 重采样后求最佳 similarity transform：

```text
b(t) ≈ A a(t) + t0 + r(t)
```

统计：

- `r(t)` 均值/标准差；
- slant；
- scale。

得到可用于未见字生成的全局 writer prior。

## 5. Weak component-conditioned profile

`dictionary.txt` decomposition 提供字符中有哪些组件，但不保证静态图中每一笔到组件的精确映射。因此对包含组件 C 的所有样本，聚合“字符级笔画残差摘要”，形成：

```text
P(style | component=C)
```

这是弱标签先验，不是组件逐笔真值。

## 6. Coverage

### geometry coverage
标准 median 用末端方向划分 8 个方向桶，并根据弦长/路径长度比粗分 straight/curve，加上 dot 类，形成近似几何覆盖空间。

### component coverage
对核心部件目标集合 T：

```text
coverage = |seen ∩ T| / |T|
```

### overall
有 decomposition 时：

```text
0.65 * component + 0.35 * geometry
```

没有 decomposition 时保守降级，不输出虚假的 component coverage。

## 7. Active acquisition / 补字建议

令 M 为缺失组件。候选字 c 的收益：

```text
gain(c) = components(c) ∩ M
```

每轮选择 `|gain(c)|` 最大的未写字符，移除已覆盖组件后继续。等价于 set-cover 的经典贪心近似。

## Context-conditioned unseen glyph generation (v0.5)

For a canonical stroke `P0(t)`, the system builds several independently learned targets from facsimile data and blends them in increasing contextual specificity:

`Pglobal -> Pcomponent -> Pposition -> Pstructure -> Padjacency -> Ptiny`.

Each statistical target stores similarity-normalized residual mean/std plus average slant/scale. The tiny model predicts a bounded residual from canonical point features, tangent, curvature, normalized stroke index/count, component hash and IDS operator hash. It does not decide character identity or stroke order.

The neural stage is trained only when `TrainingReadiness.tiny_ready` is true. Otherwise the program returns `NEED_MORE_SAMPLES` and the recommendation engine proposes characters that reduce both component deficits and structure/position deficits.
