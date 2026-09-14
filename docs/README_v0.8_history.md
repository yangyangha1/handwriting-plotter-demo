# Handwriting Plotter Demo v0.2

一个从手写图片到写字机中心轨迹的可运行工程 Demo：

1. 导入手写图片。
2. PaddleOCR 识别文字和文本框。
3. 正常手写文本切块：OCR 四点文本框透视校正 + 墨迹投影全局优化，不要求方格纸或预设字符位置。
4. GUI 人工逐字核对：可改正 OCR 字符，也可屏蔽错误切块。
5. 只对确认样本加载 Make Me A Hanzi 标准笔顺/median。
6. 标准笔画中心线向手写墨迹骨架拟合。
7. 输出单字 SVG / JSON polyline，以及保持原图布局的 `page_vectors.svg`。
8. 学习全局个人风格；如提供 `dictionary.txt`，额外学习基于 MMH matches 的逐笔偏旁风格模型。
9. 统计笔画几何与核心偏旁覆盖率，显示完成度、缺失部件，并用 greedy set-cover 推荐下一批应补写的汉字。
10. 保留 TinyStyleNet 和 G-code 导出接口，供后续写字机部署与神经风格模型升级。

> 重要边界：静态图片不能唯一恢复真实落笔时间顺序。因此系统使用 Make Me A Hanzi 的标准笔顺作为语义先验，再拟合个人几何形态；不会把图像骨架遍历顺序冒充真实笔顺。

## 1. 工程结构

```text
src/hwplotter/
├── gui.py                    # Tkinter GUI / 人工核对工作流
├── workflow.py               # 应用服务层，连接 OCR、拟合、训练与输出
├── pipeline.py               # 单字符/CLI 基础流水线
├── geometry.py               # 几何、重采样、相似变换
├── model/
│   ├── entities.py           # Character / Stroke / OCRToken 等领域对象
│   ├── session.py            # OCR 人工核对会话
│   ├── features.py
│   └── tiny_style.py         # 可选微型神经模型
├── ocr/
│   ├── paddle_adapter.py     # PaddleOCR 3.x 适配层
│   └── cropper.py            # OCR 文本框 -> 单字块
├── stroke_data/
│   ├── mmh.py                # graphics.txt 标准笔顺/median
│   └── components.py         # dictionary.txt decomposition/偏旁解析
├── vision/preprocess.py      # 二值化、tight crop、归一化
├── fitting/skeleton_snap.py  # CPU 中心轨迹拟合 baseline
├── style/
│   ├── profile.py            # 全局个人书写风格
│   ├── component_profile.py  # 偏旁→笔画精确对齐风格
│   └── coverage.py           # 覆盖度、缺失项、补字推荐
└── exporters/
    ├── svg.py
    ├── page_svg.py           # 按原图位置组合整页矢量
    ├── json_polyline.py
    └── gcode.py
```

## 2. 安装

Python 3.10+。

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
pip install -U pip
pip install -e ".[dev,gui]"
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -U pip
pip install -e '.[dev,gui]'
```

OCR 另外安装 PaddleOCR 和 PaddlePaddle。PaddlePaddle 的具体 wheel 请按操作系统/CPU/GPU 选择；装好 PaddlePaddle 后：

```bash
pip install -e '.[ocr]'
```

## 3. 下载 Make Me A Hanzi 数据

```bash
python scripts/download_mmh.py
```

得到：

```text
data/graphics.txt
data/dictionary.txt
```

其中 `graphics.txt` 是核心依赖；`dictionary.txt` 可选，但没有它就不能做可靠的偏旁语义覆盖分析，只能退化为笔画几何覆盖。

## 4. GUI

```bash
hwplotter-gui
```

也可以：

```bash
python -m hwplotter.gui
```

### GUI 标准流程

**第一步：OCR识别并切块**

选择图片、`graphics.txt`、可选 `dictionary.txt` 与输出目录，点击“1. OCR识别并切块”。GUI 使用 PySide6。

**第二步：人工核对**

列表中每个 OCR 字符块包含：

- 是否纳入；
- OCR 字符；
- 置信度；
- bbox；
- 处理状态。

选择一行可查看缩略图。OCR 字符错了就修改“正确字符”；切块错误则取消“纳入后续处理”。双击第一列可以快速屏蔽/恢复。

**第三步：生成矢量**

点击“2. 确认后生成矢量”。只有确认纳入的条目进入后端。输出：

```text
out/gui_session/
├── ocr/
│   ├── crops/
│   └── review_manifest.json
└── vectors/
    ├── 00000_xxxx.svg
    ├── 00000_xxxx.json
    ├── ...
    └── page_vectors.svg
```

`page_vectors.svg` 会把拟合后的中心轨迹放回原图片对应 bbox，是“图片中文字 -> 对应整页矢量文字”的直接结果。

**第四步：训练书写习惯 / 检查完成度**

点击“3. 训练/评估书写习惯”。输出：

```text
style/
├── style.json              # 全局 writer profile
├── component_style.json    # 有 dictionary.txt 时生成
└── coverage.json           # 完成度、缺失部件、建议补字
```

GUI 同时显示：

- 总体完成度；
- 笔画几何覆盖；
- 核心偏旁覆盖；
- 缺失核心部件；
- 建议下一批补写字符。

## 5. 覆盖度算法

### 笔画几何覆盖

把标准 median 按主方向（8 个方向）和直/曲形态粗分类。它只是工程覆盖指标，不等同于书法学的正式笔画分类。

### 偏旁覆盖

`dictionary.txt` 的 `decomposition` 提供部件语义。样本字符包含哪些部件，就认为这些部件至少出现过训练实例。

偏旁风格现在使用 Make Me A Hanzi `dictionary.txt` 的 `matches` 字段做**逐笔精确归属**。`matches[i]` 是第 `i` 个笔画在 IDS decomposition 树中的路径；程序解析 IDS 树后，可得到“组件 → 标准 stroke index → 实际拟合 stroke”的对应关系。`component_style.json` 因而保存的是组件内部第 1/2/… 笔各自的 residual、倾斜和尺度统计，而不是整字弱监督平均。

### 总完成度

有 dictionary 数据时：

```text
completion = 0.70 × component_coverage + 0.30 × geometry_coverage
```

没有 dictionary 时只报告保守的几何完成度，不伪造偏旁覆盖。

### 补字建议

对缺失核心部件执行 greedy set-cover。每轮选择能覆盖最多当前缺失部件的一个未写字符，直到补齐或达到推荐上限。因此建议文本不是随机常用字，而是针对当前样本缺口生成。

## 6. 核心拟合算法

对每个 OCR 确认字符：

```text
standard MMH median
      ↓
按手写墨迹 bbox 初始化
      ↓
手写墨迹 skeleton + distance field
      ↓
gradient attraction
+ Laplacian smoothness
+ canonical anchor
      ↓
个人中心轨迹
```

离散目标：

```text
E = λd E_skeleton + λs E_smooth + λa E_prior
```

`E_prior` 很重要：它限制标准笔画不要在“十/田/木”等交叉位置任意跳到另一条骨架支路。

## 7. CLI

原有 CLI 仍保留：

```bash
hwplotter fit --char 永 --image yong.png --graphics data/graphics.txt --out out/yong
hwplotter train-style --manifest out/ocr_crops/manifest.json --graphics data/graphics.txt --out out/style.json
hwplotter generate --text "中国电力" --graphics data/graphics.txt --style out/style/style.json --dictionary data/dictionary.txt --component-style out/style/component_style.json --out-dir out/generated
```

## 8. 测试

```bash
pytest -q
python -m compileall -q src
```

静态检查：

```bash
ruff check src tests
mypy src
```

CI 配置在 `.github/workflows/ci.yml`。

## 9. 当前明确限制

1. 输入直接支持正常手写文本图片：默认使用 PP-OCRv5 server 检测/识别并开启页面方向、文档去畸变和文本行方向处理；OCR 检出多行文本框后再做四点透视校正，并依据已识别字符序列、实际墨迹谷值和软宽度先验做全局切分。严重连笔/字符完全粘连或 OCR/切块错误时，GUI 只能排除该样本并要求重新书写，不允许人工改字后进入训练；不要求方格纸、模板坐标或逐字手工录入。
2. Skeleton Snap 是 CPU baseline，并不等价于 Handwriting Stroke Transfer 的完整 DeepFit/FlowFit。
3. 已完成 Make Me A Hanzi `matches` 的组件→stroke index 精确映射；已增加“同一组件所处的左/右/上/下位置”位置条件模型。
4. 完成度是“当前目标部件集合的覆盖指标”，不是一个可以证明“已学会某人全部汉字书写”的概率。
5. G-code 的抬笔/落笔、舵机 PWM、Z 轴高度与速度必须按具体写字机控制板标定。

## 10. 输入方式约束

本项目的主输入是普通纸张上的正常手写文本照片或扫描件，不采用方格纸作为算法前提，也不要求用户逐字点击或手工录入字符位置。OCR 与自动切分负责产生逐字候选；GUI 只承担结果核对与屏蔽；错误项要求重新书写，不允许改字。被排除或复刻失败的字符会写入 `rewrite_required.txt`，用于下一轮补写。

后续精度增强应继续围绕自然手写输入：更强的粘连字符分离、跨行/弯曲基线处理、OCR/切块错误项的排除重写和拍照畸变标定，而不是把采集流程退化成模板格输入。

详细设计见 `docs/ARCHITECTURE.md`、`docs/ALGORITHMS.md`、`docs/DEVELOPMENT.md`。


## v0.3：偏旁到具体笔画精确对齐

Make Me A Hanzi 的 `dictionary.txt` 不只有 `decomposition`，还包含 `matches`。每个 `matches[i]` 对应汉字第 `i` 笔，并给出该笔在 IDS decomposition 树中的路径。例如 `⿰亻⿱夂彡` 中路径 `[1,0]` 指向 `夂`。本项目现在完整解析 IDS 二叉/三叉树，并生成：

```text
character
  └─ decomposition tree
       └─ matches[stroke_index]
            └─ component node
                 └─ component_strokes[component] = [stroke indexes]
```

训练时只比较同一组件实际拥有的标准/拟合笔画，并按组件内部相对笔序保存统计。生成时也只把组件模型作用到这些笔画；其余笔画保留全局 writer style。覆盖率同样改为以 `matches` 成功归属且达到最小样本数的组件为“完成”，只出现一次的组件显示为“样本不足”，补字推荐采用 multi-cover 继续补足样本次数。

## v0.4: Facsimile-first pipeline

The pipeline is intentionally ordered as follows:

1. **Observed glyph facsimile**: OCR/reviewed crops are reconstructed directly from their ink, with MMH used only as stroke-count/order/direction prior. Ink is partitioned per stroke, skeletonized, ordered along the stroke prior, and exported with local ink width and per-stroke coverage confidence.
2. **Observed sample library**: every accepted facsimile is stored as a real variant under `facsimile_library/`. A character may keep multiple photographed variants.
3. **Style/component training**: global and component models are trained from the confirmed facsimile trajectories, not from canonical MMH geometry.
4. **Hybrid generation**: when a requested character exists in the facsimile library, the program replays a real observed variant. Only unseen characters are synthesized from global/component style models.

This matters: a seen glyph must not be regenerated from a style model when the real trajectory is already available.

### Facsimile JSON
Each stroke now stores both the ordered centerline and an estimated local ink width:

```json
{
  "source": "facsimile",
  "metadata": {"facsimile_quality": 0.93},
  "strokes": [{
    "points": [[0.1, 0.2], [0.11, 0.21]],
    "widths": [0.018, 0.021],
    "confidence": 0.95
  }]
}
```

For compatible plotter controllers, `GCodeOptions(dynamic_pressure=True, dynamic_speed=True)` maps recovered width to pen-power/Z-like commands and slows at high-curvature turns. Controllers differ, so those commands require hardware calibration.

## v0.5：完整未写字推演链与强制样本门控

v0.5 不再把组件位置、汉字结构、邻接关系和微型轨迹网络留作 TODO。训练目录现在同时产生：

- `style.json`：全局个人书写风格；
- `component_style.json`：偏旁/组件逐笔风格；
- `position_style.json`：同一组件在 left/right/top/bottom/center 等位置的风格；
- `structure_style.json`：`⿰/⿱/⿴/⿵/⿸/⿺...` 结构条件风格；
- `adjacency_style.json`：组件在相邻组件/结构上下文下的局部变化；
- `tiny_trajectory.pt` + `tiny_trajectory_meta.json`：微型轨迹残差网络（只有样本门控通过才训练）；
- `training_readiness.json`：训练是否可用于高置信未写字推演。

严格生成顺序：

```text
真实字样本库 replay
        ↓（没有真实字）
MMH 标准笔顺
        ↓
全局个人风格
        ↓
偏旁逐笔风格
        ↓
组件位置模型
        ↓
汉字结构模型
        ↓
邻接关系模型
        ↓
Tiny Trajectory residual
        ↓
SVG / JSON / G-code
```

### 样本不足不允许“假完成”

训练后状态只有两种：

- `READY`：允许 GUI 正常生成未写字；
- `NEED_MORE_SAMPLES`：GUI 禁用“生成矢量文字”，显示缺口并要求继续导入/书写建议字符。

默认门控（工程初始阈值，可配置）同时检查：

- 不同汉字数 ≥ 50；
- 有效笔画数 ≥ 300；
- 核心偏旁覆盖率 ≥ 65%；
- 笔画几何覆盖率 ≥ 70%；
- 至少 4 类主要汉字结构达到重复样本；
- left/right/top/bottom 四类组件位置均达到样本要求；
- 至少 12 种邻接上下文；
- Tiny 网络训练点 ≥ 2000。

这些是工程启动阈值，不是“书法统计学真值”。实际设备/用户数据积累后，应通过留出验证集误差重新标定。

### GUI v0.5

GUI 四步：

1. OCR识别并切块；
2. 人工确认/屏蔽后高保真复刻；
3. 训练完整个人模型并显示 `READY / NEED_MORE_SAMPLES`；
4. `READY` 后输入任意文字生成。已写字始终优先重放真实样本，未写字才进入层级推演。

如果状态为 `NEED_MORE_SAMPLES`，界面会显示：不同字数、总笔画、核心偏旁、结构类型、位置类型、邻接上下文等具体缺口，以及自动补写文本。


## v0.7：自然手写图片高保真页级复刻、轻微连笔与笔顺推断

本版本重新审计“图片 → 写字机轨迹”主链。目标输入仍是普通手写文本照片/扫描件，不使用方格采集。

### 1. 页面几何不再因单字 tight-crop 丢失

v0.6 在单字拟合前会裁掉空白并归一到方形，这会损失字在原字符区域中的左右留白、重心和局部基线位置。v0.7 改为完整 OCR 字符单元 letterbox：不裁掉字符单元空白，并保存 OCR 分割得到的原始四点多边形。恢复轨迹后再把坐标反变换回原字符单元，页级 SVG/G-code 通过四点透视映射回原图片坐标。

因此页级复刻同时保留：

- 字间距；
- 行间距；
- 字符自身左右/上下偏移；
- 轻微倾斜与透视；
- 行基线位置。

### 2. 轻微跨字连笔作为独立 transition trajectory

普通行书/快速手写中，相邻字之间可能有细小连线。直接按字符切图会把连接线切成两半并污染前后两个字的训练数据。现在分割器会在字符边界附近寻找“细、跨越边界、连续”的墨迹组件：

```text
前字真实笔迹 ── transition connector ── 后字真实笔迹
```

检测到后：

1. connector 保存为原页面坐标的独立轨迹；
2. 从前后两个单字训练 crop 中剥离该窄连接段；
3. 单字/偏旁模型不学习跨字连接线；
4. 页级 SVG 把 connector 按原位置补回；
5. 页级 G-code 在端点距离满足条件时，把“前字最后一个 pen-down path → connector → 后字第一个 pen-down path”作为一次连续落笔执行，不人为抬笔。

### 3. 单字内部连笔：标准 stroke 保留，物理 pen-down path 单独建模

风格训练仍保留 Make Me A Hanzi 的 canonical stroke index，以保证偏旁、组件和笔画模型语义稳定；同时新增：

- `stroke_order`：推断后的实际笔画次序；
- `pen_down_groups`：实际一次不抬笔连续写过的 canonical stroke 组；
- `stroke_order_confidence`；
- `stroke_order_mode`；
- `stroke_order_ambiguous`。

例如标准上是 4 个笔画，但图片显示第 2、3 笔轻微连写，可保持训练语义为 4 笔，同时写字机实际执行为：

```text
[stroke0] 抬笔
[stroke1 + stroke2] 连续落笔
[stroke3]
```

### 4. 草书/行书笔顺不是硬套楷书

静态图片没有时间轴，因此不能在所有交叉、覆盖墨迹中数学上唯一恢复真实书写时间顺序。v0.7 不再把 MMH 楷书笔顺直接声明为“识别结果”，而是把它作为先验之一。

候选来源包括：

1. Make Me A Hanzi 标准笔顺；
2. 可选 `cursive_orders.jsonl` 中的行书/草书变体；
3. 根据当前图片生成的局部异序候选（最多两次相邻换序）；
4. 图片端点距离；
5. 两笔之间实际墨迹连续性；
6. 笔画复刻质量。

候选通过几何/墨迹证据打分，取得分最优顺序并计算与次优方案的 margin。如果时序证据不足，则：

```text
stroke_order_ambiguous = true
→ 样本自动标记失败
→ 不进入真实样本库
→ 不进入个人模型训练
→ rewrite_required.txt 要求重新书写
```

因此系统不会把无法从静态图确定的草书笔顺“猜成正确答案”。

可选草书顺序文件格式：

```json
{"character":"字","variants":[{"name":"running-1","order":[0,1,3,2,4]}]}
```

其中数字是 MMH canonical stroke index。没有该文件时仍会执行图片驱动的局部异序推断与连笔合并。

### 5. 局部笔宽进入矢量复刻

Facsimile 已恢复每个中心轨迹点附近的真实墨迹宽度。v0.7 在 letterbox 反变换时同步恢复宽度尺度：

- 单字 SVG 可按每小段局部宽度绘制；
- 页级 SVG 可按原字符区域尺度恢复粗细变化；
- 支持动态压力的写字机可继续用 `dynamic_pressure` 将宽度映射到压力/Z/PWM。

### 6. 输出

图片复刻阶段现在同时输出：

```text
vectors/
├── 00000_xxxx.svg
├── 00000_xxxx.json
├── ...
├── page_vectors.svg
└── page_vectors.gcode
```

`page_vectors.svg` 用于检查页级形状、间距和连笔；`page_vectors.gcode` 按推断物理笔顺、连笔组和跨字 connector 输出写字机轨迹。

### 7. 质量门控

一个 OCR 正确的字仍可能因静态图笔顺歧义而被拒绝。进入训练至少需要：

```text
OCR/分割被用户接受
+ facsimile 成功
+ stroke_order_confidence 达标
+ stroke_order_ambiguous == false
```

GUI 新增“笔顺/连笔”列显示推断模式和置信度。用户仍然只做两件事：保留或排除；不允许手改 OCR 标签或手工指定笔顺。

### 8. 现实边界

对于轻微连笔、行书式并笔和局部顺序变化，当前实现已建立图片驱动的处理链。对于高度简化、多个标准笔画已经变成完全不同草书字形的重度草书，仅靠 MMH canonical stroke 拟合仍不应声称可完美恢复；此类样本会更容易落入低置信门控。要支持重度草书，需要加入具有真实时间轨迹的草书/在线手写数据训练更强的时序先验，而不能只依赖静态字图。

## v0.8：轻量模型泛化 + 自动持续学习

v0.8 不再把个人神经模型当成一次性训练文件。每一批正常手写文本图片完成 OCR、排除错误样本和 Facsimile 复刻后，GUI 会自动更新当前个人 profile：

1. 从 `facsimile_library/` 汇总该 profile 历史全部有效真实轨迹；
2. 按“汉字身份”划分训练集和留出验证集，避免同一个字的不同版本泄漏到两侧；
3. 训练/更新确定性风格模型、组件/位置/结构/邻接模型；
4. 训练 `writer_structure.pt`（Structure NN），学习个人字形整体压缩、展开、重心和结构比例；
5. 训练 `writer_stroke.pt`（Stroke NN），学习规则层无法解释的逐笔 `Δx/Δy`；
6. 用从未参与训练的留出汉字计算验证误差；
7. 新版本优于当前最佳版本才激活，否则自动恢复上一版本；
8. 根据偏旁、结构位置和验证薄弱区域生成 `active_learning_plan.json`，要求用户继续以正常连续文本方式补写最有信息量的字符。

模型文件示例：

```text
style/
├── writer_structure.pt
├── writer_structure_meta.json
├── writer_stroke.pt
├── writer_stroke_meta.json
├── active_learning_plan.json
├── continual_state.json
└── versions/
    ├── v0001/
    ├── v0002/
    └── ...
```

`continual_state.json` 记录当前激活版本、历史验证误差和每次候选模型是否被接受。若本轮新增图片使验证误差变差，候选版本会被拒绝，`style/` 自动恢复到上一激活版本。

### 为什么能减少首次输入量

固定规则模型需要分别覆盖大量“偏旁 × 位置 × 邻接组件”组合。v0.8 的 Structure NN / Stroke NN 学习连续变形规律，例如“右侧组件越宽，左偏旁通常收窄多少”，因此不需要穷举所有组合。与此同时，主动学习不会机械要求写满固定字数，而是优先推荐当前模型最缺的信息。

不过软件仍保留 `NEED_MORE_SAMPLES` 门控：验证集不足、关键结构缺失或模型泛化误差尚未稳定时，不会把个人模型标记为 READY。用户只需继续正常书写推荐内容并导入图片，软件自动重复上述训练、验证和版本验收流程。
