# Third-party notes

This demo does not copy source code from the referenced upstream projects.

- Handwriting Stroke Transfer: code reported as MIT by its repository. Architectural reference: canonical Hanzi stroke semantics + target glyph fitting; lightweight skeleton/median snapping; DeepFit/FlowFit stages.
- stroke-order: repository code is MIT; its README lists per-data-source licenses separately.
- Make Me A Hanzi: `graphics.txt` is distributed separately and has its own license terms. Do not assume the demo code license changes the data license.
- PaddleOCR: optional runtime dependency; follow its project/model licenses for deployment.

Before commercial distribution, pin exact upstream versions and reproduce their license/notice files as required.


## v1.2 bundled materials

- MMH graphics.txt and dictionary.txt are included, with upstream MMH_COPYING, LGPL and APL notices in data/. Data source: https://github.com/skishore/makemeahanzi . Generated experimental glyphs derive from the MMH graphics; retain the corresponding Arphic notices when redistributing.
- Historic manuscript thumbnail: Wang Xizhi, Kuaixue Shiqing Tie (detail); photographic reproduction of an old public-domain work, source https://commons.wikimedia.org/wiki/File:快雪時晴帖(局部).jpg . Kept only as a transparent test fixture; not a newly authored handwriting sample.
- FontTools, RapidOCR/ONNXRuntime and OpenCV are runtime dependencies, not vendored code. No DeepCalliFont weights or source are included.
- Chromium and Noto CJK were used only in the test environment and are not shipped in this code archive.
