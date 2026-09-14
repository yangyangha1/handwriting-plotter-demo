from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from hwplotter.model.session import ReviewSession
from hwplotter.workflow import (
    create_ocr_session,
    generate_text_v05,
    train_reviewed_style,
    vectorize_reviewed,
)


class WorkerSignals(QObject):
    finished = Signal(object)
    error = Signal(str)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            self.signals.finished.emit(self.fn(*self.args, **self.kwargs))
        except Exception as exc:  # noqa: BLE001 - report per-job failure to user
            self.signals.error.emit(f"{type(exc).__name__}: {exc}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HWPlotter - 手写字 OCR / 矢量化 / 风格拟合")
        self.resize(1380, 880)
        self.pool = QThreadPool.globalInstance()
        self.image_path: Path | None = None
        self.session: ReviewSession | None = None
        self.samples = []
        self.readiness = None
        self.work_dir = Path("out/gui_session")
        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)

        files = QGroupBox("工程数据")
        fg = QGridLayout(files)
        self.graphics_edit = QLineEdit("data/graphics.txt")
        self.dictionary_edit = QLineEdit("data/dictionary.txt")
        self.image_edit = QLineEdit()
        self.image_edit.setReadOnly(True)
        fg.addWidget(QLabel("图片"), 0, 0)
        fg.addWidget(self.image_edit, 0, 1)
        b = QPushButton("导入图片")
        b.clicked.connect(self.choose_image)
        fg.addWidget(b, 0, 2)
        fg.addWidget(QLabel("graphics.txt"), 1, 0)
        fg.addWidget(self.graphics_edit, 1, 1)
        b = QPushButton("选择")
        b.clicked.connect(lambda: self.choose_data(self.graphics_edit))
        fg.addWidget(b, 1, 2)
        fg.addWidget(QLabel("dictionary.txt"), 2, 0)
        fg.addWidget(self.dictionary_edit, 2, 1)
        b = QPushButton("选择")
        b.clicked.connect(lambda: self.choose_data(self.dictionary_edit))
        fg.addWidget(b, 2, 2)
        self.output_edit = QLineEdit(str(self.work_dir))
        fg.addWidget(QLabel("输出目录"), 3, 0)
        fg.addWidget(self.output_edit, 3, 1)
        b = QPushButton("选择")
        b.clicked.connect(self.choose_output)
        fg.addWidget(b, 3, 2)
        outer.addWidget(files)

        actions = QHBoxLayout()
        self.ocr_btn = QPushButton("1. OCR 识别并切块")
        self.ocr_btn.clicked.connect(self.run_ocr)
        self.vec_btn = QPushButton("2. 确认后生成矢量")
        self.vec_btn.clicked.connect(self.run_vectorize)
        self.vec_btn.setEnabled(False)
        self.train_btn = QPushButton("3. 训练完整个人模型")
        self.train_btn.clicked.connect(self.run_training)
        self.train_btn.setEnabled(False)
        for x in (self.ocr_btn, self.vec_btn, self.train_btn):
            actions.addWidget(x)
        self.status = QLabel("等待导入图片")
        actions.addWidget(self.status, 1)
        outer.addLayout(actions)

        split = QSplitter(Qt.Horizontal)
        outer.addWidget(split, 1)
        self.image_label = QLabel("原图预览")
        self.image_label.setAlignment(Qt.AlignCenter)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(self.image_label)
        split.addWidget(sc)

        right = QWidget()
        rv = QVBoxLayout(right)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["使用", "切块", "OCR结果（只读）", "OCR置信度", "复刻质量", "笔顺/连笔", "状态"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 55)
        self.table.setColumnWidth(1, 90)
        self.table.setColumnWidth(2, 160)
        rv.addWidget(
            QLabel(
                "逐字检查：识别或切块错误时取消“使用”。错误项不允许改字，直接排除并要求重新书写。"
            )
        )
        rv.addWidget(self.table, 3)
        self.svg = QSvgWidget()
        self.svg.setMinimumHeight(230)
        rv.addWidget(self.svg, 2)
        split.addWidget(right)
        split.setSizes([600, 780])

        coverage = QGroupBox("拟合完成度 / 补字建议")
        cv = QVBoxLayout(coverage)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        cv.addWidget(self.progress)
        self.coverage_text = QTextEdit()
        self.coverage_text.setReadOnly(True)
        self.coverage_text.setMaximumHeight(170)
        cv.addWidget(self.coverage_text)
        outer.addWidget(coverage)

        gen = QGroupBox("未写字推演 / 真实字优先重放")
        gg = QHBoxLayout(gen)
        self.generate_edit = QLineEdit()
        self.generate_edit.setPlaceholderText(
            "输入要生成的文字；已写过的字直接复刻，未写字走完整层级模型"
        )
        self.generate_btn = QPushButton("4. 生成矢量文字")
        self.generate_btn.setEnabled(False)
        self.generate_btn.clicked.connect(self.run_generate)
        gg.addWidget(self.generate_edit, 1)
        gg.addWidget(self.generate_btn)
        outer.addWidget(gen)

    def choose_image(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "选择手写图片", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not p:
            return
        self.image_path = Path(p)
        self.image_edit.setText(p)
        pix = QPixmap(p)
        self.image_label.setPixmap(
            pix.scaled(720, 650, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self.status.setText("图片已导入，可以 OCR")

    def choose_data(self, edit):
        p, _ = QFileDialog.getOpenFileName(
            self, "选择 Make Me A Hanzi 数据", "", "Text (*.txt);;All (*)"
        )
        if p:
            edit.setText(p)

    def choose_output(self):
        p = QFileDialog.getExistingDirectory(self, "选择输出目录", str(self.work_dir))
        if p:
            self.work_dir = Path(p)
            self.output_edit.setText(p)

    def _busy(self, text):
        self.status.setText(text)
        self.ocr_btn.setEnabled(False)
        self.vec_btn.setEnabled(False)
        self.train_btn.setEnabled(False)

    def _error(self, msg):
        QMessageBox.critical(self, "执行失败", msg)
        self.status.setText(msg)
        self.ocr_btn.setEnabled(True)
        self.vec_btn.setEnabled(self.session is not None)
        self.train_btn.setEnabled(bool(self.samples))

    def run_ocr(self):
        if not self.image_path:
            return QMessageBox.warning(self, "缺少图片", "请先导入图片")
        self.work_dir = Path(self.output_edit.text().strip() or "out/gui_session")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._busy("OCR 模型识别中…首次运行可能下载模型")
        w = Worker(create_ocr_session, self.image_path, self.work_dir / "ocr")
        w.signals.finished.connect(self.ocr_done)
        w.signals.error.connect(self._error)
        self.pool.start(w)

    def ocr_done(self, session):
        self.session = session
        self.samples = []
        self.table.setRowCount(len(session.items))
        for r, item in enumerate(session.items):
            cb = QCheckBox()
            cb.setChecked(True)
            self.table.setCellWidget(r, 0, cb)
            pix = QPixmap(str(item.image_path)).scaled(
                64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            lab = QLabel()
            lab.setPixmap(pix)
            lab.setAlignment(Qt.AlignCenter)
            self.table.setCellWidget(r, 1, lab)
            ch_item = QTableWidgetItem(item.char)
            ch_item.setFlags(ch_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(r, 2, ch_item)
            conf_item = QTableWidgetItem(f"{item.confidence:.3f}")
            conf_item.setFlags(conf_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(r, 3, conf_item)
            self.table.setItem(r, 4, QTableWidgetItem("—"))
            self.table.setItem(r, 5, QTableWidgetItem("—"))
            self.table.setItem(r, 6, QTableWidgetItem("待确认"))
        self.status.setText(
            f"OCR 完成：{len(session.items)} 个单字；错误项直接取消使用并重新书写，不做人工改字"
        )
        self.ocr_btn.setEnabled(True)
        self.vec_btn.setEnabled(bool(session.items))

    def _sync_review(self):
        assert self.session
        for r, item in enumerate(self.session.items):
            cb = self.table.cellWidget(r, 0)
            item.included = bool(cb and cb.isChecked())
            item.rejection_reason = None if item.included else "manual_exclude_rewrite"
            # OCR label is intentionally immutable: bad recognition/cropping is rejected,
            # never corrected in-place, to keep training labels image-derived.
            item.char = item.original_char or item.char

    def run_vectorize(self):
        if not self.session:
            return
        self._sync_review()
        graphics = Path(self.graphics_edit.text())
        if not graphics.exists():
            return self._error(f"找不到 {graphics}。请下载 Make Me A Hanzi graphics.txt。")
        self._busy("正在高保真复刻已写字：笔顺约束、逐笔墨迹分配、中心线与局部笔宽恢复…")
        w = Worker(vectorize_reviewed, self.session, graphics, self.work_dir / "vectors")
        w.signals.finished.connect(self.vector_done)
        w.signals.error.connect(self._error)
        self.pool.start(w)

    def vector_done(self, result):
        self.samples, page_svg = result
        for r, item in enumerate(self.session.items):
            state = (
                "已排除·需重写"
                if not item.included
                else (f"失败·需重写: {item.error}" if item.error else "已矢量化")
            )
            q = "—" if item.facsimile_quality is None else f"{item.facsimile_quality * 100:.1f}%"
            order_text = (
                "—"
                if item.stroke_order_confidence is None
                else f"{item.stroke_order_mode} {item.stroke_order_confidence * 100:.1f}%"
            )
            self.table.setItem(r, 4, QTableWidgetItem(q))
            self.table.setItem(r, 5, QTableWidgetItem(order_text))
            self.table.setItem(r, 6, QTableWidgetItem(state))
        if page_svg.exists():
            self.svg.load(str(page_svg))
        rewrite = self.session.rewrite_text() if self.session else ""
        suffix = f"；需重新书写：{rewrite}" if rewrite else ""
        self.status.setText(
            f"原字复刻完成：有效样本 {len(self.samples)} 个，已写字进入真实样本库{suffix}；正在自动更新个人模型…"
        )
        self.ocr_btn.setEnabled(True)
        self.vec_btn.setEnabled(True)
        self.train_btn.setEnabled(bool(self.samples))
        # v0.8 continual learning: every accepted acquisition batch automatically
        # updates the personal profile.  The trainer itself performs held-out
        # validation and rolls back if the candidate model regresses.
        d = Path(self.dictionary_edit.text())
        if self.samples and d.exists():
            self.run_training()

    def run_training(self):
        if not self.session or not self.samples:
            return
        graphics = Path(self.graphics_edit.text())
        d = Path(self.dictionary_edit.text())
        dictionary = d if d.exists() else None
        self._busy("正在统计笔画形变、偏旁风格与覆盖缺口…")
        w = Worker(
            train_reviewed_style,
            self.session,
            self.samples,
            graphics,
            self.work_dir / "style",
            dictionary,
        )
        w.signals.finished.connect(self.training_done)
        w.signals.error.connect(self._error)
        self.pool.start(w)

    def training_done(self, result):
        _, report, readiness = result
        self.readiness = readiness
        pct = round(report.overall_completion * 100)
        self.progress.setValue(pct)
        missing = "、".join(report.missing_components[:40]) or "无"
        insufficient = "、".join(report.insufficient_components[:40]) or "无"
        rewrite = self.session.rewrite_text() if self.session else ""
        model_suggestion = report.suggested_text or ""
        suggestion = rewrite + model_suggestion
        # Preserve order while removing duplicates so rejected OCR/crops are requested first.
        suggestion = (
            "".join(dict.fromkeys(suggestion)) or "当前核心组件已达到目标，或暂无可用补字建议。"
        )
        notes = "\n".join(f"• {x}" for x in report.notes)
        readiness_text = (
            "READY：可以进行未写字高置信推演"
            if readiness.ready
            else "NEED_MORE_SAMPLES：样本不足，请继续输入"
        )
        version_text = (
            f"v{readiness.active_model_version:04d}"
            if readiness.active_model_version
            else "未建立稳定版本"
        )
        val_text = (
            "—" if readiness.validation_error is None else f"{readiness.validation_error:.6f}"
        )
        gaps = "\n".join(f"• {x}" for x in readiness.reasons) or "• 无"
        self.coverage_text.setPlainText(
            f"训练状态：{readiness_text}\n"
            f"当前个人模型：{version_text}；留出验证误差：{val_text}\n"
            f"综合完成度：{pct}%\n偏旁笔画级覆盖：{report.component_coverage * 100:.1f}%\n"
            f"笔画几何覆盖：{report.geometry_coverage * 100:.1f}%\n"
            f"有效字：{report.unique_characters}；总笔画：{report.stroke_count_seen}；"
            f"已归属偏旁笔画：{report.aligned_component_strokes}\n\n"
            f"关键缺口：\n{gaps}\n\n"
            f"完全缺失偏旁：{missing}\n"
            f"已有样本但不足：{insufficient}\n\n"
            f"建议继续补写：{suggestion}\n\n{notes}"
        )
        self.status.setText(
            f"个人模型 {version_text} 已通过验证并启用"
            if readiness.ready
            else f"个人模型已自动更新/验收；样本不足，请按建议继续输入（当前 {version_text}）"
        )
        self.generate_btn.setEnabled(readiness.ready)
        self.ocr_btn.setEnabled(True)
        self.vec_btn.setEnabled(True)
        self.train_btn.setEnabled(True)

    def run_generate(self):
        text = self.generate_edit.text().strip()
        if not text:
            return QMessageBox.warning(self, "缺少文字", "请输入需要生成的文字")
        if not self.readiness or not self.readiness.ready:
            return QMessageBox.warning(
                self, "样本不足", "当前训练状态为 NEED_MORE_SAMPLES，请按补字建议继续输入样本。"
            )
        graphics = Path(self.graphics_edit.text())
        dictionary = Path(self.dictionary_edit.text())
        self._busy("正在生成：真实字优先复刻，未写字使用位置/结构/邻接/Tiny层级推演…")
        self.generate_btn.setEnabled(False)
        w = Worker(
            generate_text_v05,
            text,
            graphics,
            dictionary,
            self.work_dir / "style",
            self.work_dir / "facsimile_library",
            self.work_dir / "generated",
        )
        w.signals.finished.connect(self.generate_done)
        w.signals.error.connect(self._error)
        self.pool.start(w)

    def generate_done(self, outputs):
        self.status.setText(
            f"生成完成：{len(outputs)} 个字符，输出目录 {self.work_dir / 'generated'}"
        )
        if outputs:
            self.svg.load(str(outputs[0]))
        self.ocr_btn.setEnabled(True)
        self.vec_btn.setEnabled(True)
        self.train_btn.setEnabled(True)
        self.generate_btn.setEnabled(bool(self.readiness and self.readiness.ready))


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
