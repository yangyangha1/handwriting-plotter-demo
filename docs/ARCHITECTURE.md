# Architecture v0.2

## 分层

### Presentation
`gui.py` 只管理用户交互、状态展示和后台线程，不直接实现 OCR/拟合算法。

### Application
`workflow.py` 是应用服务层，负责 OCR 会话、审核 manifest、批量矢量化、训练和覆盖分析。GUI 和未来 Web/API 都应调用这一层。

### Domain
`model/entities.py`、`model/session.py` 定义 Stroke、Character、ReviewItem、ReviewSession 等稳定 IR。所有算法通过这些对象通信，避免绑定 PaddleOCR 或某个上游仓库内部对象。

### Infrastructure / Adapters
- `ocr/paddle_adapter.py`: PaddleOCR 适配。
- `stroke_data/mmh.py`: Make Me A Hanzi graphics 数据。
- `stroke_data/components.py`: dictionary decomposition。
- `exporters/*`: SVG/JSON/G-code。

### Algorithms
- `vision/preprocess.py`: 图像预处理。
- `fitting/skeleton_snap.py`: 标准轨迹 -> 手写骨架。
- `style/profile.py`: 全局风格。
- `style/component_profile.py`: 弱监督偏旁条件风格。
- `style/coverage.py`: 覆盖度和主动补样建议。

## 数据流

```text
Image
  -> OCRToken[]
  -> CharacterCrop[]
  -> ReviewSession
  -> human correction / screening
  -> accepted ReviewItem[]
  -> MMH Character canonical strokes
  -> fitted Character trajectories
  -> per-character SVG/JSON + page SVG
  -> StyleSample[]
  -> global StyleProfile
  -> weak ComponentStyleModel
  -> CoverageReport
  -> suggested acquisition characters
```

## 失败隔离

批量矢量化按字符隔离异常。单个字符 MMH 缺失或拟合失败时，记录到 `ReviewItem.error`，其余字符继续处理。GUI 会把失败条目标为错误状态。

## 性能

`vectorize_reviewed()` 每批只实例化一次 `MakeMeAHanziSource`，避免对 `graphics.txt` 重复全量索引。后续若数据量更大，可换 SQLite/LMDB/预编译索引，而无需改变 Domain API。

## 并发

GUI 的 OCR、矢量化、训练都在 daemon worker thread 执行；Qt 主线程只处理 UI。结果通过 Qt Signal 回到主线程，避免 Tk 跨线程更新。

## v0.5 Hierarchical Generation

The synthesis subsystem is confidence-gated and hierarchical. It intentionally avoids an end-to-end black box.

1. Facsimile replay for observed glyphs.
2. Canonical MMH medians for unseen glyphs.
3. Global writer residual profile.
4. Exact component/stroke-slot conditioning from MMH `matches`.
5. Component-position conditioning (`component@left/right/top/bottom/...`).
6. IDS root-structure conditioning (`⿰`, `⿱`, enclosure families, etc.).
7. Generalized adjacency context (`component@position | neighbor-position-set | IDS-op`).
8. TinyTrajectoryNet residual refinement.
9. SVG/JSON/G-code export.

`training_readiness.json` is a hard quality gate. The production GUI will not enable unseen-glyph generation while the state is `NEED_MORE_SAMPLES`. Real observed glyphs remain replayable because they do not require inference.
