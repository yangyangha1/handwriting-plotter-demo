# Development Process

## Definition of Done

一个功能合入前至少满足：

1. Domain/API 边界明确，不让 GUI 直接依赖算法内部实现。
2. 新逻辑有单元测试或端到端后端测试。
3. `pytest -q` 通过。
4. `python -m compileall -q src` 通过。
5. 本地/CI 安装 dev 依赖后执行 `ruff check src tests`。
6. 数据格式/算法假设写入 docs。
7. 任何无法从静态图恢复的事实必须标明先验或弱监督，不伪造成观测真值。

## 当前验证结果

本版本构建时已执行：

```text
pytest -q
7 passed

python -m compileall -q src
passed
```

当前构建容器没有安装 `ruff`，因此未声称静态检查通过；CI/本地 dev 环境应执行该步骤。

## Branch/commit 建议

```text
main
feature/gui-review
feature/component-coverage
feature/deepfit-backend
feature/plotter-driver
```

Commit 遵循单一职责，例如：

```text
feat(gui): add OCR review and screening workflow
feat(style): add decomposition-aware coverage report
perf(mmh): reuse graphics index for batch fitting
test(workflow): exclude screened OCR items from fitting
```

## 测试层次

- geometry unit tests
- exporter unit tests
- component parser / coverage tests
- screening workflow integration test
- synthetic image -> fitted vectors integration test
- PaddleOCR smoke test（需要可用 Paddle 运行时和模型）
- GUI manual acceptance test

## GUI 验收清单

1. 可选择图片、graphics、dictionary、输出目录。
2. OCR 后显示每个字符块。
3. 选择字符块可预览。
4. OCR 或切块错误时只能排除该样本，并要求用户重新书写；不允许人工改字后进入训练。
5. 错误切块可屏蔽。
6. 被屏蔽条目不进入 vectorize。
7. 单字 SVG/JSON 正常生成。
8. `page_vectors.svg` 与原图 bbox 布局对应。
9. 风格训练输出 `style.json`。
10. dictionary 存在时输出 `component_style.json`、偏旁覆盖、缺失项、补写建议。
11. dictionary 不存在时明确降级，不显示伪偏旁完成度。

## v0.5 acceptance checks

Required before release:

```bash
python -m compileall -q src tests
pytest -q
python -m hwplotter.cli --help
```

Tiny-model smoke test must train a small synthetic dataset, persist a state dict, reload it and return `source=tiny-trajectory-refined`.

GUI runtime verification additionally requires the optional `gui` dependency (`PySide6`). Static compilation alone is not counted as GUI runtime verification.
