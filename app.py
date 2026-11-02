"""Предварительная обработка пар панорам петрографических шлифов:
совмещение с усреднением геометрии, исправление локальных расхождений и общая обрезка."""
import copy
import math
import os
import sys
import time
import traceback
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QPoint, QPointF, QRectF, QRunnable, Qt, QThread, QThreadPool, QTimer, Signal
from PySide6.QtGui import (QBrush, QColor, QFont, QIcon, QImage, QKeySequence, QPainter, QPainterPath, QPalette,
                           QPen, QPixmap, QShortcut, QTransform)
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QFileDialog, QFormLayout, QFrame, QGraphicsItem, QGraphicsScene,
                               QGraphicsView, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
                               QProgressDialog, QPushButton, QRadioButton, QScrollArea, QSlider, QSpinBox, QSplitter, QTextEdit,
                               QVBoxLayout, QWidget)

import core
import geometry

PREVIEW_MAX = 4096   # максимальная сторона предпросмотра, px
CACHE_ITEMS = 8      # сколько предпросмотров держать в памяти
UNDO_LIMIT = 100
FONT_FAMILY = "Arial"
VIEW_BG = "#1f2329"
CHECKER = (np.array([0x9a, 0x9f, 0xa8], np.uint8), np.array([0x6b, 0x71, 0x7b], np.uint8))

STYLE = """
QMainWindow, QDialog { background: #f3f4f6; }
QToolBar { background: #ffffff; border: none; border-bottom: 1px solid #e5e7eb; padding: 6px 10px; spacing: 4px; }
QToolBar::separator { width: 1px; background: #e5e7eb; margin: 6px 8px; }
QToolButton { background: transparent; border: 1px solid transparent; border-radius: 6px; padding: 5px 10px; }
QToolButton:hover { background: #f1f5f9; border-color: #e2e8f0; }
QToolButton:checked { background: #dbeafe; border-color: #93c5fd; color: #1d4ed8; }
QToolButton:disabled { color: #9ca3af; }
QToolButton#primary { background: #2563eb; color: #ffffff; font-weight: 600; }
QToolButton#primary:hover { background: #1d4ed8; }
QToolButton#primary:disabled { background: #bfdbfe; }
QFrame#card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px; }
QLabel#header { color: #6b7280; font-size: 8pt; font-weight: 600; }
QLabel#hint { color: #9ca3af; font-size: 8pt; }
QLabel#next { color: #1d4ed8; font-weight: 600; }
QLabel#viewTitle { font-weight: 600; color: #111827; padding: 2px 4px 6px 4px; }
QLabel#viewTitle[target="true"] { color: #1d4ed8; }
QPushButton { background: #ffffff; border: 1px solid #d1d5db; border-radius: 6px; padding: 6px 12px; }
QPushButton:hover { background: #f8fafc; border-color: #9ca3af; }
QPushButton:pressed { background: #eef2f7; }
QPushButton:checked { background: #dbeafe; border-color: #3b82f6; color: #1d4ed8; }
QPushButton:disabled { color: #9ca3af; border-color: #e5e7eb; }
QPushButton#primary { background: #2563eb; border-color: #2563eb; color: #ffffff; font-weight: 600; }
QPushButton#primary:hover { background: #1d4ed8; }
QPushButton#primary:disabled { background: #bfdbfe; border-color: #bfdbfe; }
QPushButton#danger { color: #b91c1c; }
QPushButton#danger:disabled { color: #9ca3af; }
QListWidget { background: #ffffff; border: none; outline: 0; }
QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }
QListWidget::item { padding: 5px 6px; border-radius: 6px; }
QListWidget::item:hover { background: #f1f5f9; }
QListWidget::item:selected { background: #dbeafe; color: #1e3a8a; }
QStatusBar { background: #ffffff; border-top: 1px solid #e5e7eb; color: #4b5563; }
QSplitter::handle { background: transparent; }
QGraphicsView { border: none; border-radius: 6px; }
QToolTip { background: #111827; color: #ffffff; border: none; padding: 4px 8px; }
QProgressBar { border: 1px solid #e5e7eb; border-radius: 4px; background: #f8fafc; max-height: 10px; }
QProgressBar::chunk { background: #2563eb; border-radius: 3px; }
QSlider::groove:horizontal { height: 4px; background: #e5e7eb; border-radius: 2px; }
QSlider::handle:horizontal { background: #2563eb; width: 14px; margin: -6px 0; border-radius: 7px; }
"""


def make_font(px, bold=False):
    f = QFont(FONT_FAMILY)
    f.setPixelSize(max(1, round(px)))
    f.setBold(bold)
    return f


def to_qimage(arr):
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def checker_fill(tile, mask, x, y):
    """Пустые области результата в предпросмотре — серая «шахматка»."""
    yy, xx = np.nonzero(mask)
    odd = (((xx + x) // 12 + (yy + y) // 12) % 2).astype(bool)
    tile[yy[odd], xx[odd]] = CHECKER[0]
    tile[yy[~odd], xx[~odd]] = CHECKER[1]


def is_text_input(w):
    if isinstance(w, (QLineEdit, QAbstractSpinBox, QTextEdit, QPlainTextEdit)):
        return True
    return isinstance(w, QComboBox) and w.isEditable()


# ---------------------------------------------------------------- фоновые задачи

class _Signals(QObject):
    done = Signal(str, object, str)


class _Task(QRunnable):
    def __init__(self, key, fn, signals):
        super().__init__()
        self.key, self.fn, self.signals = key, fn, signals

    def run(self):
        try:
            self.signals.done.emit(self.key, self.fn(), "")
        except geometry.AlignError as e:
            self.signals.done.emit(self.key, None, str(e))
        except Exception as e:  # noqa: BLE001 — показать пользователю любую ошибку
            self.signals.done.emit(self.key, None, f"{type(e).__name__}: {e}")


class Tasks(QObject):
    """Очередь фоновых задач (загрузка предпросмотров, расчёт совмещения)."""
    finished = Signal(str, object, str)
    busy = Signal(int)

    def __init__(self):
        super().__init__()
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(2)
        self.running = set()
        self.signals = _Signals()
        self.signals.done.connect(self._done)

    def submit(self, key, fn):
        if key in self.running:
            return
        self.running.add(key)
        self.pool.start(_Task(key, fn, self.signals))
        self.busy.emit(len(self.running))

    def _done(self, key, result, err):
        self.running.discard(key)
        self.busy.emit(len(self.running))
        self.finished.emit(key, result, err)


def task_load_source(path):
    def fn():
        arr, size = core.load_preview(path, PREVIEW_MAX)
        return {"arr": arr, "size": size}
    return fn


def preview_scale(canvas):
    return min(1.0, PREVIEW_MAX / max(canvas[2], canvas[3]))


def task_warp(path, side, params, src):
    """Исправленный предпросмотр: уменьшенный исходник → общая система координат (холст целиком)."""
    def fn():
        s = src or task_load_source(path)()
        field = geometry.get_field(params)
        canvas = params["canvas"]
        arr = field.render(side, s["arr"], s["size"], canvas, preview_scale(canvas), fill=checker_fill,
                           interp=cv2.INTER_LINEAR)
        return {"arr": arr, "rect": list(canvas)}
    return fn


# ---------------------------------------------------------------- графические элементы

class MarkerItem(QGraphicsItem):
    """Точка совмещения с номером; размер на экране не зависит от увеличения."""
    COLORS = {"normal": "#ff2bd6", "selected": "#facc15", "ghost": "#22d3ee", "common": "#ff2bd6"}

    def __init__(self, text, kind, index):
        super().__init__()
        self.text, self.kind, self.index = text, kind, index
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setZValue(2e6 + (1 if kind == "selected" else 0))

    def boundingRect(self):
        return QRectF(-16, -34, 60, 52)

    def paint(self, p, option, widget=None):
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = 11 if self.kind == "selected" else 8
        path = QPainterPath()
        path.addEllipse(QPointF(0, 0), r, r)
        for a, b in (((-r - 5, 0), (r + 5, 0)), ((0, -r - 5), (0, r + 5))):
            path.moveTo(*a)
            path.lineTo(*b)
        if self.text:
            path.addText(QPointF(r + 4, -r - 2), make_font(18, bold=True), self.text)
        color = QColor(self.COLORS[self.kind])
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(0, 0, 0, 170) if self.kind == "ghost" else QColor("#ffffff"), 4))
        p.drawPath(path)
        pen = QPen(color, 3 if self.kind == "selected" else 2)
        if self.kind == "ghost":
            pen.setStyle(Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.drawPath(path)


class CropItem(QGraphicsItem):
    """Рамка обрезки: затемнение снаружи, контур и маркеры краёв/углов."""

    def __init__(self, canvas):
        super().__init__()
        self.canvas = canvas
        self.rect = None
        self.applied = False
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setZValue(1e6)

    def set_state(self, rect, applied):
        self.prepareGeometryChange()
        self.rect, self.applied = rect, applied
        self.update()

    def boundingRect(self):
        m = max(self.canvas.width(), self.canvas.height()) * 0.05
        return self.canvas.adjusted(-m, -m, m, m)

    def paint(self, p, option, widget=None):
        if self.rect is None:
            return
        path = QPainterPath()
        path.setFillRule(Qt.FillRule.OddEvenFill)
        path.addRect(self.canvas)
        path.addRect(self.rect.intersected(self.canvas))
        p.fillPath(path, QColor(0, 0, 0, 150))
        k = 1 / max(1e-9, p.worldTransform().m11())
        pen = QPen(QColor("#22c55e" if self.applied else "#3b82f6"), 2)
        pen.setCosmetic(True)
        if not self.applied:
            pen.setStyle(Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(self.rect)
        s = 4 * k
        p.setBrush(QColor("#ffffff"))
        hp = QPen(QColor("#2563eb"), 1.5)
        hp.setCosmetic(True)
        p.setPen(hp)
        r = self.rect
        for x in (r.left(), r.center().x(), r.right()):
            for y in (r.top(), r.center().y(), r.bottom()):
                if x == r.center().x() and y == r.center().y():
                    continue
                p.drawRect(QRectF(x - s, y - s, 2 * s, 2 * s))


CURSORS = {"l": Qt.CursorShape.SizeHorCursor, "r": Qt.CursorShape.SizeHorCursor,
           "t": Qt.CursorShape.SizeVerCursor, "b": Qt.CursorShape.SizeVerCursor,
           "lt": Qt.CursorShape.SizeFDiagCursor, "rb": Qt.CursorShape.SizeFDiagCursor,
           "rt": Qt.CursorShape.SizeBDiagCursor, "lb": Qt.CursorShape.SizeBDiagCursor,
           "move": Qt.CursorShape.SizeAllCursor, "new": Qt.CursorShape.CrossCursor}


class PanoramaView(QGraphicsView):
    def __init__(self, index, main):
        super().__init__()
        self.index, self.main = index, main
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)  # привязка к курсору — вручную
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QColor(VIEW_BG))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.viewport().setMouseTracking(True)
        self.horizontalScrollBar().valueChanged.connect(lambda *_: self.main.view_changed(self))
        self.verticalScrollBar().valueChanged.connect(lambda *_: self.main.view_changed(self))
        self.side = None              # исходник в отдельном окне (режим «Рядом» на исходниках) или None
        self.frames = {}              # сторона → (A, B): координаты сцены = A·(пиксели исходника) + B
        self.common = (1 + 0j, 0j)    # (A, B): координаты сцены = A·(общая система C) + B
        self.content = QRectF()
        self._wheel_anchor = None
        self.layers, self.layer_keys = {}, {}
        self.markers = []
        self.crop_item = None
        self.placeholder = []
        self._pan = None
        self._press = None
        self._drag = None
        self._space = False

    # --- содержимое
    def reset(self, rect, message, side=None, frames=None, common=(1 + 0j, 0j)):
        self.main.syncing += 1
        self.scene().clear()
        self.layers, self.layer_keys, self.markers, self.crop_item = {}, {}, [], None
        self.side = side
        self.frames = frames or {}
        self.common = common
        self._wheel_anchor = None
        self.content = QRectF(rect)
        bg = self.scene().addRect(self.content, QPen(Qt.PenStyle.NoPen), QBrush(QColor("#2d323a")))
        bg.setZValue(-1e6)
        text = self.scene().addSimpleText(message, make_font(max(8.0, self.content.width() / 30)))
        text.setBrush(QColor("#9ca3af"))
        br = text.boundingRect()
        text.setPos(self.content.center().x() - br.width() / 2, self.content.center().y() - br.height() / 2)
        text.setZValue(-1e6)
        self.placeholder = [bg, text]
        # большой запас вокруг изображения: точка под курсором остаётся на месте при любом допустимом увеличении
        m = 10 * max(self.content.width(), self.content.height())
        self.setSceneRect(self.content.adjusted(-m, -m, m, m))
        self.main.syncing -= 1

    def set_message(self, message):
        if len(self.placeholder) == 2:
            self.placeholder[1].setText(message)

    def set_layer(self, side, arr, tf, key):
        """Слой side: пиксели arr → сцена преобразованием tf (QTransform)."""
        if self.layer_keys.get(side) == key:
            return
        old = self.layers.pop(side, None)
        if old is not None:
            self.scene().removeItem(old)
        if len(self.placeholder) == 2:
            self.scene().removeItem(self.placeholder[1])
            self.placeholder = self.placeholder[:1]
        item = self.scene().addPixmap(QPixmap.fromImage(to_qimage(arr)))
        item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        item.setTransform(tf)
        item.setZValue(-1e5 + (1 if side == "crossed" else 0))
        item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.layers[side] = item
        self.layer_keys[side] = key

    def set_layer_state(self, visible, opacity):
        for side, item in self.layers.items():
            item.setVisible(visible.get(side, False))
            item.setOpacity(opacity.get(side, 1.0))

    def set_markers(self, specs):
        for it in self.markers:
            self.scene().removeItem(it)
        self.markers = []
        for text, x, y, kind, index in specs:
            it = MarkerItem(text, kind, index)
            it.setPos(x, y)
            self.scene().addItem(it)
            self.markers.append(it)

    def set_crop(self, canvas, rect, applied):
        if self.crop_item is None:
            self.crop_item = CropItem(QRectF(*canvas))
            self.scene().addItem(self.crop_item)
        self.crop_item.set_state(QRectF(*rect) if rect else None, applied)

    def fit(self, rect=None):
        self.fitInView(rect or self.content, Qt.AspectRatioMode.KeepAspectRatio)
        self.main.view_changed(self)

    def center(self):
        return self.mapToScene(self.viewport().rect().center())

    def scale_factor(self):
        """Пикселей экрана на единицу сцены."""
        t = self.transform()
        return math.hypot(t.m11(), t.m12())

    def fit_scale(self):
        if self.content.isEmpty():
            return 1.0
        vp = self.viewport().rect()
        return min(vp.width() / self.content.width(), vp.height() / self.content.height())

    def zoom_at(self, p, factor):
        """Увеличение в factor раз; точка сцены под экранной точкой p остаётся на месте."""
        anchor = self.mapToScene(p)
        if self._wheel_anchor and self._wheel_anchor[0] == p and \
                (self._wheel_anchor[1] - anchor).manhattanLength() * self.scale_factor() <= 2:
            anchor = self._wheel_anchor[1]   # серия шагов колеса: без накопления ошибок округления
        self._wheel_anchor = (QPoint(p), QPointF(anchor))
        self.main.syncing += 1
        try:
            self.scale(factor, factor)
            q = self.mapFromScene(anchor)
            hb, vb = self.horizontalScrollBar(), self.verticalScrollBar()
            hb.setValue(hb.value() + q.x() - p.x())
            vb.setValue(vb.value() + q.y() - p.y())
        finally:
            self.main.syncing -= 1
        self.main.view_changed(self)

    # --- курсор и попадания
    def update_cursor(self, hit=None):
        if self._pan is not None:
            shape = Qt.CursorShape.ClosedHandCursor
        elif self._space:
            shape = Qt.CursorShape.OpenHandCursor
        elif hit:
            shape = CURSORS.get(hit, Qt.CursorShape.ArrowCursor)
        elif self.main.mode() == "points" and self.main.adding:
            shape = Qt.CursorShape.CrossCursor
        else:
            shape = Qt.CursorShape.ArrowCursor
        self.viewport().setCursor(shape)

    def marker_at(self, vpos):
        best, bd = None, 14.0
        for it in self.markers:
            if it.kind in ("normal", "selected"):
                d = math.dist(self.mapFromScene(it.pos()).toPointF().toTuple(), vpos.toTuple())
                if d < bd:
                    best, bd = it, d
        return best

    def crop_hit(self, p):
        r = self.main.crop_rect()
        if not r:
            return "new"
        tol = 8 / max(1e-9, self.scale_factor())
        x0, y0, x1, y1 = r[0], r[1], r[0] + r[2], r[1] + r[3]
        inside_y = y0 - tol <= p.y() <= y1 + tol
        inside_x = x0 - tol <= p.x() <= x1 + tol
        h = ""
        if inside_y and abs(p.x() - x0) <= tol:
            h += "l"
        elif inside_y and abs(p.x() - x1) <= tol:
            h += "r"
        if inside_x and abs(p.y() - y0) <= tol:
            h += "t"
        elif inside_x and abs(p.y() - y1) <= tol:
            h += "b"
        if h:
            return h
        if x0 < p.x() < x1 and y0 < p.y() < y1:
            return "move"
        return "new"

    # --- события
    def focusNextPrevChild(self, nxt):
        return False  # Tab — переключение параллельных / скрещённых николей

    def wheelEvent(self, e):
        d = e.angleDelta().y() or e.pixelDelta().y()
        if not d:
            return
        cur = self.scale_factor()
        new = min(max(cur * 1.0015 ** d, 0.1 * self.fit_scale()), 40.0)
        if abs(new / cur - 1) > 1e-9:
            self.zoom_at(e.position().toPoint(), new / cur)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.MiddleButton or (e.button() == Qt.MouseButton.LeftButton and self._space):
            self._pan = e.position()
            self.update_cursor()
            return
        if e.button() != Qt.MouseButton.LeftButton:
            return
        p = self.mapToScene(e.position().toPoint())
        mode = self.main.mode()
        side = self.main.interaction_side(self)
        if mode == "points" and side:
            m = self.marker_at(e.position())
            if m is not None:
                self._drag = ("point", m, m.pos() - p, side)
                self.main.select_point(m.index)
                return
            self._press = (e.position(), side)
            return
        if mode == "points" and self.main.single():
            self.main.statusBar().showMessage("В режиме наложения точки не ставятся — выберите «Переключение» "
                                              "или «Рядом»", 4000)
            return
        if mode == "result":
            hit = self.crop_hit(p)
            self._drag = ("crop", hit, p, self.main.crop_rect(), e.position())
            self.main.crop_begin()

    def mouseMoveEvent(self, e):
        if self._pan is not None:
            d = e.position() - self._pan
            self._pan = e.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - round(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - round(d.y()))
            return
        p = self.mapToScene(e.position().toPoint())
        if self._drag and self._drag[0] == "point":
            self._drag[1].setPos(p + self._drag[2])
            return
        if self._drag and self._drag[0] == "crop":
            _, hit, p0, start, _ = self._drag
            self.main.crop_drag(hit, p0, p, start)
            return
        if self.main.mode() == "result" and not self._space:
            self.update_cursor(self.crop_hit(p))
        elif self.main.interaction_side(self) and self.marker_at(e.position()) is not None:
            self.viewport().setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self.update_cursor()

    def mouseReleaseEvent(self, e):
        if self._pan is not None:
            self._pan = None
            self.update_cursor()
            return
        if e.button() != Qt.MouseButton.LeftButton:
            return
        p = self.mapToScene(e.position().toPoint())
        if self._drag and self._drag[0] == "point":
            item, side = self._drag[1], self._drag[3]
            self._drag = None
            self.main.move_point(side, item.index, *self.main.scene_to_source(self, side, item.pos()))
            return
        if self._drag and self._drag[0] == "crop":
            _, hit, p0, start, v0 = self._drag
            self._drag = None
            click = (e.position() - v0).manhattanLength() < 4
            self.main.crop_end(start if click and hit == "new" else None)
            return
        if self._press is not None:
            (v0, side), self._press = self._press, None
            if (e.position() - v0).manhattanLength() < 6:
                self.main.add_point(side, *self.main.scene_to_source(self, side, p))

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
            self._space = True
            self.update_cursor()
            return
        if e.key() == Qt.Key.Key_Tab:
            self.main.toggle_switch()
            return
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
            self._space = False
            self._pan = None
            self.update_cursor()
            return
        super().keyReleaseEvent(e)

    def enterEvent(self, e):
        if not is_text_input(QApplication.focusWidget()):
            self.setFocus()
        super().enterEvent(e)


# ---------------------------------------------------------------- диалоги и оформление

class AssignDialog(QDialog):
    def __init__(self, pair, files, parent):
        super().__init__(parent)
        self.setWindowTitle(f"Назначение изображений — {pair['name']}")
        form = QFormLayout(self)
        self.combos = {}
        for side in core.SIDES:
            cb = QComboBox()
            cb.addItem("— не назначено —", None)
            for f in files:
                cb.addItem(f, f)
            cb.setCurrentIndex(max(0, cb.findData(pair[side])))
            cb.setMinimumWidth(350)
            form.addRow(core.SIDE_TITLES[side], cb)
            self.combos[side] = cb
        note = QLabel("Точки на заменённом изображении будут удалены, совмещение нужно будет повторить.")
        note.setObjectName("hint")
        note.setWordWrap(True)
        form.addRow(note)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def values(self):
        return {side: cb.currentData() for side, cb in self.combos.items()}


def status_icon(kind):
    """Кружок-отметка в списке шлифов: обработана / совмещена / нет."""
    pm = QPixmap(16, 16)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    if kind == "done":
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#16a34a"))
        p.drawEllipse(QRectF(1, 1, 14, 14))
        pen = QPen(QColor("#ffffff"), 2)
        p.setPen(pen)
        p.drawPolyline([QPointF(4.5, 8.5), QPointF(7, 11), QPointF(11.5, 5.5)])
    elif kind == "aligned":
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#2563eb"))
        p.drawEllipse(QRectF(3, 3, 10, 10))
    else:
        p.setPen(QPen(QColor("#9ca3af"), 1.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QRectF(3, 3, 10, 10))
    p.end()
    return QIcon(pm)


def card(*widgets, title=None):
    """Белая карточка с необязательным заголовком."""
    frame = QFrame()
    frame.setObjectName("card")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(10, 10, 10, 10)
    lay.setSpacing(8)
    if title:
        h = QLabel(title.upper())
        h.setObjectName("header")
        lay.addWidget(h)
    for w in widgets:
        lay.addWidget(w)
    return frame, lay


def hint(text):
    lb = QLabel(text)
    lb.setObjectName("hint")
    lb.setWordWrap(True)
    return lb


def rich_label():
    lb = QLabel()
    lb.setWordWrap(True)
    lb.setTextFormat(Qt.TextFormat.RichText)
    return lb


def row(*widgets):
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(6)
    for x in widgets:
        lay.addWidget(x, 1)
    return w


# ---------------------------------------------------------------- экспорт в фоне

class ExportWorker(QThread):
    progress = Signal(float, int, str)
    report = Signal(list, int, bool, str)

    def __init__(self, pairs, image_dir, out_dir, quality, fmt):
        super().__init__()
        self.pairs, self.image_dir, self.out_dir = pairs, image_dir, out_dir
        self.quality, self.fmt = quality, fmt
        self.cancelled = False

    def run(self):
        errors, details, done = [], [], 0
        for i, pair in enumerate(self.pairs):
            if self.cancelled:
                break
            self.progress.emit(float(i), i, pair["name"])
            try:
                res = core.export_pair(pair, self.image_dir, self.out_dir, self.quality, self.fmt,
                                       progress=lambda f, i=i: self.progress.emit(i + f, i, pair["name"]),
                                       cancelled=lambda: self.cancelled)
                if res:
                    done += 1
            except Exception as e:  # noqa: BLE001 — ошибка одной пары не прерывает пакет
                errors.append(f"{pair['name']}: {e}")
                details.append(f"=== {pair['name']}\nПараллельные: {pair['parallel']}\nСкрещённые: {pair['crossed']}\n"
                               + "".join(traceback.format_exception(e)))
        self.report.emit(errors, done, self.cancelled, "\n".join(details))


# ---------------------------------------------------------------- главное окно

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.project = None
        self.cur = -1
        self.sel_point = None
        self.adding = False          # режим «Добавить точки»
        self.show_sources = False    # «Показать исходники» для совмещённой пары
        self.view_mode = "pair"      # pair / overlay / switch
        self.switch_side = "parallel"
        self.frame = None            # (α, β) общей системы для показа исходников в одном окне
        self.syncing = 0
        self.undo, self.redo = {}, {}
        self.crop_before = None
        self.worker = None
        self.last_export_dir = ""
        self.cache = OrderedDict()
        self.errors = {}
        self.pending_align = {}
        self.tasks = Tasks()
        self.tasks.finished.connect(self._task_done)
        self.tasks.busy.connect(self._busy)
        self.save_timer = QTimer(self, singleShot=True, interval=2000, timeout=self.save_project)
        self._build_ui()
        self._update_title()
        self.update_panel()

    # --- интерфейс
    def _build_ui(self):
        tb = self.addToolBar("Главная")
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.addAction("Новый проект…", self.new_project)
        tb.addAction("Открыть…", self.open_project)
        tb.addAction("Сохранить", self.save_project).setToolTip("Сохранить проект (Ctrl+S)")
        tb.addSeparator()
        tb.addAction("Показать целиком", self.fit_views)
        spacer = QWidget()
        spacer.setSizePolicy(spacer.sizePolicy().Policy.Expanding, spacer.sizePolicy().Policy.Preferred)
        tb.addWidget(spacer)
        tb.addWidget(QLabel("Формат "))
        self.format_combo = QComboBox()
        for key, text in core.EXPORT_FORMATS.items():
            self.format_combo.addItem(text, key)
        self.format_combo.setToolTip("Исходный: PNG сохраняется в PNG (без потерь), JPEG — в JPEG с исходным "
                                     "расширением. «Всё в JPEG» — оба файла .jpg")
        self.format_combo.activated.connect(self._format_changed)
        tb.addWidget(self.format_combo)
        tb.addWidget(QLabel("  Качество JPEG "))
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(50, 100)
        self.quality_spin.setValue(core.DEFAULT_SETTINGS["jpeg_quality"])
        self.quality_spin.setToolTip("Качество JPEG при экспорте (при 90 и выше — без субдискретизации цвета)")
        self.quality_spin.valueChanged.connect(self._quality_changed)
        tb.addWidget(self.quality_spin)
        tb.addSeparator()
        self.act_exp_cur = tb.addAction("Экспорт текущей", lambda: self.export(False))
        self.act_exp_all = tb.addAction("Экспорт обработанных", lambda: self.export(True))
        self.act_exp_all.setToolTip("Экспортировать все пары с отметкой «Пара обработана»")
        tb.widgetForAction(self.act_exp_all).setObjectName("primary")

        # левая область — список шлифов
        self.pair_list = QListWidget()
        self.pair_list.currentRowChanged.connect(self.select_pair)
        b_swap = QPushButton("Поменять местами")
        b_swap.setToolTip("Поменять местами изображения параллельных и скрещённых николей")
        b_swap.clicked.connect(self.swap_pair)
        b_assign = QPushButton("Назначить файлы…")
        b_assign.clicked.connect(self.assign_files)
        b_del = QPushButton("Удалить из списка…")
        b_del.setObjectName("danger")
        b_del.setToolTip("Убрать пару из проекта; исходные фотографии на диске не удаляются")
        b_del.clicked.connect(self.delete_pair)
        self.pair_buttons = [b_swap, b_assign, b_del]
        left, _ = card(self.pair_list, b_swap, b_assign, b_del, title="Шлифы")

        # центр — две панорамы
        center = QSplitter(Qt.Orientation.Horizontal)
        center.setHandleWidth(8)
        self.views, self.view_titles, self.view_cards = [], [], []
        for i in range(2):
            title = QLabel()
            title.setObjectName("viewTitle")
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            view = PanoramaView(i, self)
            frame, _ = card(title, view)
            center.addWidget(frame)
            self.views.append(view)
            self.view_titles.append(title)
            self.view_cards.append(frame)

        # справа — инструменты
        self.next_label = QLabel()
        self.next_label.setObjectName("next")
        self.next_label.setWordWrap(True)
        self.point_list = QListWidget()
        self.point_list.setMinimumHeight(120)
        self.point_list.currentRowChanged.connect(lambda r: self.select_point(r if r >= 0 else None, False))
        self.btn_add = QPushButton("Добавить точки")
        self.btn_add.setCheckable(True)
        self.btn_add.setToolTip("Щелчок по исходному изображению ставит точку; ставьте её поочерёдно на обеих "
                                "панорамах на одной и той же детали. Esc — выход из режима.")
        self.btn_add.clicked.connect(self.toggle_adding)
        self.btn_del_point = QPushButton("Удалить точку")
        self.btn_del_point.setObjectName("danger")
        self.btn_del_point.setToolTip("Удалить выбранную пару точек (Delete)")
        self.btn_del_point.clicked.connect(self.delete_point)
        self.btn_average = QPushButton("Усреднить геометрию")
        self.btn_average.setObjectName("primary")
        self.btn_average.setToolTip("Привести обе панорамы к общей промежуточной геометрии (вклад 50/50) по точкам. "
                                    "Расчёт всегда выполняется из исходных изображений.")
        self.btn_average.clicked.connect(self.average_geometry)
        self.btn_sources = QPushButton("Показать исходники")
        self.btn_sources.setCheckable(True)
        self.btn_sources.setToolTip("Показать исходные изображения с точками; совмещение и обрезка сохраняются")
        self.btn_sources.clicked.connect(self.toggle_sources)
        self.btn_reset = QPushButton("Сбросить совмещение")
        self.btn_reset.setToolTip("Вернуться к исходным изображениям: совмещение и обрезка удаляются, точки остаются")
        self.btn_reset.clicked.connect(self.reset_alignment)
        self.align_label = rich_label()
        align_card, _ = card(self.next_label, self.point_list, row(self.btn_add, self.btn_del_point),
                             self.btn_average, row(self.btn_sources, self.btn_reset), self.align_label,
                             title="Совмещение")

        self.mode_group = QButtonGroup(self)
        self.mode_radios = {}
        for key, text, tip in (("pair", "Рядом", "Две панорамы рядом"),
                               ("overlay", "Наложение", "Скрещённые поверх параллельных с прозрачностью"),
                               ("switch", "Переключение", "Одно изображение; Tab — переключить на другое")):
            rb = QRadioButton(text)
            rb.setToolTip(tip)
            rb.toggled.connect(lambda on, k=key: on and self.set_view_mode(k))
            self.mode_group.addButton(rb)
            self.mode_radios[key] = rb
        self.mode_radios["pair"].blockSignals(True)
        self.mode_radios["pair"].setChecked(True)
        self.mode_radios["pair"].blockSignals(False)
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(50)
        self.opacity.setToolTip("Прозрачность скрещённых николей при наложении")
        self.opacity.valueChanged.connect(lambda *_: self.apply_layers())
        self.opacity_label = QLabel()
        self.btn_switch = QPushButton("Переключить (Tab)")
        self.btn_switch.clicked.connect(self.toggle_switch)
        self.sync_cb = QCheckBox("Синхронная навигация")
        self.sync_cb.setToolTip("Одинаковое увеличение и положение на обеих панорамах")
        self.sync_cb.setChecked(True)
        self.sync_cb.toggled.connect(self._sync_toggled)
        self.common_cb = QCheckBox("Показывать точки на результате")
        self.common_cb.setChecked(True)
        self.common_cb.toggled.connect(lambda *_: self.update_markers())
        view_card, _ = card(row(*self.mode_radios.values()), row(self.opacity_label, self.opacity), self.btn_switch,
                            self.sync_cb, self.common_cb, title="Просмотр")

        self.btn_apply_crop = QPushButton("Применить обрезку")
        self.btn_apply_crop.setObjectName("primary")
        self.btn_apply_crop.clicked.connect(self.apply_crop)
        self.btn_reset_crop = QPushButton("Сбросить обрезку")
        self.btn_reset_crop.setToolTip("Убрать рамку; совмещение сохраняется")
        self.btn_reset_crop.clicked.connect(self.reset_crop)
        self.crop_label = rich_label()
        crop_card, _ = card(hint("После совмещения нарисуйте рамку мышью на любой панораме; её можно двигать и "
                                 "менять за края и углы. Рамка общая для обеих панорам. Серая «шахматка» — "
                                 "пустые области после деформации."),
                            row(self.btn_apply_crop, self.btn_reset_crop), self.crop_label, title="Обрезка")

        self.done_cb = QCheckBox("Пара обработана")
        self.done_cb.setToolTip("Отметка для «Экспорт обработанных»; ставится автоматически при применении обрезки")
        self.done_cb.toggled.connect(self.set_done)
        status_card, _ = card(self.done_cb, title="Статус")

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(8)
        for c in (align_card, view_card, crop_card, status_card):
            rv.addWidget(c)
        rv.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(right)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(320)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(8)
        split.addWidget(left)
        split.addWidget(center)
        split.addWidget(scroll)
        split.setStretchFactor(1, 1)
        split.setSizes([240, 1200, 370])
        outer = QWidget()
        ol = QVBoxLayout(outer)
        ol.setContentsMargins(8, 8, 8, 8)
        ol.addWidget(split)
        self.setCentralWidget(outer)

        self.info_label = QLabel()
        self.statusBar().addPermanentWidget(self.info_label)
        self.busy_label = QLabel()
        self.busy_bar = QProgressBar()
        self.busy_bar.setRange(0, 0)
        self.busy_bar.setFixedWidth(140)
        self.busy_bar.setTextVisible(False)
        self.statusBar().addPermanentWidget(self.busy_label)
        self.statusBar().addPermanentWidget(self.busy_bar)
        self.busy_label.hide()
        self.busy_bar.hide()

        for key, fn in (("Esc", self.escape), ("Delete", self.delete_point), ("Ctrl+S", self.save_project),
                        ("Ctrl+Z", self.do_undo), ("Ctrl+Y", self.do_redo), ("Ctrl+Shift+Z", self.do_redo)):
            QShortcut(QKeySequence(key), self, activated=lambda f=fn: self._hotkey(f))

    def _hotkey(self, fn):
        if not is_text_input(QApplication.focusWidget()):
            fn()

    def _update_title(self):
        name = os.path.basename(self.project.path) if self.project else "нет проекта"
        self.setWindowTitle(f"Предварительная обработка панорам шлифов — {name}")

    def _busy(self, n):
        self.busy_bar.setVisible(n > 0)
        self.busy_label.setVisible(n > 0)
        computing = any(k.startswith("align|") for k in self.tasks.running)
        self.busy_label.setText("Расчёт совмещения…" if computing else "Загрузка и подготовка предпросмотра…")

    # --- проект
    def pair(self):
        if self.project and 0 <= self.cur < len(self.project.pairs):
            return self.project.pairs[self.cur]
        return None

    def new_project(self):
        folder = QFileDialog.getExistingDirectory(self, "Папка с панорамами")
        if not folder:
            return
        try:
            pairs = core.scan_folder(folder)
        except OSError as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось прочитать папку:\n{e}")
            return
        if not pairs:
            QMessageBox.warning(self, "Нет изображений", "В выбранной папке нет файлов PNG/JPG/JPEG.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить проект как",
                                              os.path.join(folder, "предобработка.json"), "Проект (*.json)")
        if not path:
            return
        self.save_project()
        project = core.Project(path, folder, pairs)
        try:
            project.save()
        except OSError as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось сохранить проект:\n{e}")
            return
        self.set_project(project)
        bad = [p for p in pairs if p["problem"]]
        msg = f"Найдено шлифов: {len(pairs)}."
        if bad:
            lines = "\n".join(f"• {p['name']}: {p['problem']}" for p in bad[:30])
            more = f"\n… и ещё {len(bad) - 30}" if len(bad) > 30 else ""
            QMessageBox.warning(self, "Требуется проверка пар",
                                f"{msg}\nНеполные или неоднозначные пары ({len(bad)}):\n{lines}{more}\n\n"
                                "Отмечены знаком ⚠. Исправьте их кнопкой «Назначить файлы…».")
        else:
            self.statusBar().showMessage(msg, 5000)

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Открыть проект", "", "Проект (*.json)")
        if not path:
            return
        try:
            project = core.Project.load(path)
        except core.ProjectError as e:
            QMessageBox.warning(self, "Ошибка", str(e))
            return
        self.save_project()
        missing = project.missing_files()
        if missing:
            lines = "\n".join(missing[:15]) + (f"\n… и ещё {len(missing) - 15}" if len(missing) > 15 else "")
            ans = QMessageBox.question(
                self, "Исходные файлы не найдены",
                f"В папке «{project.image_dir}» не найдено файлов: {len(missing)}\n{lines}\n\n"
                "Указать другую папку с изображениями?")
            if ans == QMessageBox.StandardButton.Yes:
                folder = QFileDialog.getExistingDirectory(self, "Папка с панорамами", project.image_dir)
                if folder:
                    project.image_dir = folder
                    still = len(project.missing_files())
                    if still:
                        QMessageBox.information(self, "Файлы не найдены",
                                                f"В новой папке по-прежнему не найдено файлов: {still}.")
        core.clear_size_cache()
        reset = project.check_sizes()
        if reset:
            QMessageBox.information(self, "Совмещение сброшено",
                                    "Размер исходных файлов изменился, совмещение нужно повторить:\n" +
                                    "\n".join(reset[:20]))
        try:
            project.save()
        except OSError:
            pass
        self.set_project(project)

    def set_project(self, project):
        self.project = project
        self.undo, self.redo = {}, {}
        self.cur, self.sel_point = -1, None
        core.clear_size_cache()
        self.errors.clear()
        self.quality_spin.blockSignals(True)
        self.quality_spin.setValue(int(project.settings["jpeg_quality"]))
        self.quality_spin.blockSignals(False)
        idx = self.format_combo.findData(project.settings.get("export_format", "source"))
        self.format_combo.setCurrentIndex(max(0, idx))
        self.pair_list.blockSignals(True)
        self.pair_list.clear()
        for _ in project.pairs:
            self.pair_list.addItem(QListWidgetItem())
        self.pair_list.blockSignals(False)
        for i in range(len(project.pairs)):
            self._refresh_pair_item(i)
        self._update_title()
        if project.pairs:
            self.pair_list.setCurrentRow(0)
        else:
            self.update_panel()

    def _refresh_pair_item(self, i):
        p = self.project.pairs[i]
        missing = [p[s] for s in core.SIDES if p[s] and not os.path.isfile(self.project.image_path(p, s))]
        problem = p["problem"] or ("файл не найден: " + ", ".join(missing) if missing else "")
        item = self.pair_list.item(i)
        item.setText(("⚠ " if problem else "") + p["name"])
        item.setIcon(status_icon("done" if p["done"] else "aligned" if p["align"] else "none"))
        state = "обработана" if p["done"] else "совмещена" if p["align"] else "не совмещена"
        tip = f"Параллельные: {p['parallel'] or '—'}\nСкрещённые: {p['crossed'] or '—'}\nСостояние: {state}"
        item.setToolTip(tip + (f"\n\n{problem}" if problem else ""))

    def save_project(self):
        self.save_timer.stop()
        if not self.project:
            return
        try:
            self.project.save()
            self.statusBar().showMessage("Проект сохранён", 2000)
        except OSError as e:
            self.statusBar().showMessage(f"Ошибка сохранения проекта: {e}", 10000)

    def changed(self):
        self.save_timer.start()

    def _format_changed(self, *_):
        if self.project:
            self.project.settings["export_format"] = self.format_combo.currentData()
            self.changed()

    def _quality_changed(self, v):
        if self.project:
            self.project.settings["jpeg_quality"] = int(v)
            self.changed()

    # --- режимы показа
    def mode(self):
        pair = self.pair()
        if not pair:
            return "none"
        if core.is_aligned(pair) and not self.show_sources:
            return "result"
        return "points"

    def select_pair(self, i):
        if not self.project or i < 0:
            return
        self.save_project()
        self.cur, self.sel_point = i, None
        pair = self.pair()
        self.show_sources = False
        self.adding = not core.is_aligned(pair)
        self.rebuild(fit=True)

    def rebuild(self, fit=False):
        """Полная перестройка сцен (смена пары, режима или совмещения). Без fit положение и увеличение
        просмотра сохраняются (пересчитываются через общую систему координат)."""
        states = None if fit else self.capture_views()
        pair = self.pair()
        mode = self.mode()
        if mode == "none":
            self.frame = None
            for v in self.views:
                v.reset(QRectF(0, 0, 4000, 3000), "Нет шлифов" if self.project else "Создайте или откройте проект")
            self.apply_layout()
            self.update_panel()
            return
        sizes = self.sizes(pair)
        self.frame = geometry.display_frame(pair["points"], sizes, pair["align"]) if all(sizes.values()) else None
        if mode == "result":
            canvas = pair["align"]["canvas"]
            for v in self.views:
                v.reset(QRectF(*canvas), "Подготовка исправленного изображения…")
        elif self.single():
            frames = self.source_frames()
            pts = [A * complex(x, y) + B for side, (A, B) in frames.items()
                   for x, y in ((0, 0), (sizes[side][0], 0), (0, sizes[side][1]), sizes[side])]
            x0, x1 = min(p.real for p in pts), max(p.real for p in pts)
            y0, y1 = min(p.imag for p in pts), max(p.imag for p in pts)
            rect = QRectF(x0, y0, x1 - x0, y1 - y0)
            self.views[0].reset(rect, "Загрузка…", frames=frames)
            self.views[1].reset(rect, "")
        else:
            for v, side in zip(self.views, core.SIDES):
                path = self.project.image_path(pair, side)
                size = sizes[side]
                if not pair[side]:
                    msg = "Изображение не назначено"
                elif not os.path.isfile(path):
                    msg = f"Файл не найден:\n{pair[side]}"
                elif size is None:
                    msg = "Не удалось прочитать файл:\n" + pair[side]
                else:
                    msg = "Загрузка…"
                v.reset(QRectF(0, 0, *(size or (4000, 3000))), msg, side if size else None,
                        frames={side: (1 + 0j, 0j)} if size else None, common=self.common_for(side))
        self.request_layers()
        self.apply_layout()
        self.update_markers()
        self.update_crop()
        self.update_panel()
        if fit or not any(states):
            QTimer.singleShot(0, self.fit_views)
        else:
            QTimer.singleShot(0, lambda: self.restore_views(states))

    def single(self):
        """Одно окно (наложение или переключение) — для результата и для исходников."""
        if self.view_mode == "pair":
            return False
        mode = self.mode()
        return mode == "result" or (mode == "points" and self.frame is not None)

    def source_frames(self):
        """Исходники в общей системе (только общее подобие, без локальной деформации)."""
        a, b = self.frame
        return {"parallel": (1 / a, -b / a), "crossed": (a, b)}

    def common_for(self, side):
        """(A, B): пиксели исходника side = A·C + B."""
        if self.frame is None:
            return (1 + 0j, 0j)
        a, b = self.frame
        return (a, b) if side == "parallel" else (1 / a, -b / a)

    def scene_to_source(self, view, side, p):
        A, B = view.frames.get(side, (1 + 0j, 0j))
        z = (complex(p.x(), p.y()) - B) / A
        return z.real, z.imag

    def interaction_side(self, view):
        """На каком исходнике щелчки в этом окне ставят и двигают точки (None — нигде)."""
        if self.mode() != "points":
            return None
        if self.single():
            return self.switch_side if self.view_mode == "switch" and view.index == 0 else None
        return view.side

    def capture_views(self):
        """Положение и увеличение видимых окон в общей системе координат."""
        out = []
        for v in self.views:
            if not v.isVisible() or v.content.isEmpty():
                out.append(None)
                continue
            A, B = v.common
            c = v.center()
            out.append(((complex(c.x(), c.y()) - B) / A, v.scale_factor() * abs(A)))
        return out

    def restore_views(self, states):
        first = next((st for st in states if st), None)
        if first is None:
            self.fit_views()
            return
        self.syncing += 1
        for v, st in zip(self.views, states):
            c, k = st or first
            A, B = v.common
            z = A * c + B
            v.setTransform(QTransform.fromScale(k / abs(A), k / abs(A)))
            v.centerOn(QPointF(z.real, z.imag))
        self.syncing -= 1
        self.view_changed(self.views[0])

    def layer_keys(self):
        """Нужные изображения: {сторона: (ключ кэша, задача, ключ исходника)}. Исправленный предпросмотр
        готовится из уменьшенного исходника, поэтому сначала загружается исходник."""
        pair = self.pair()
        out = {}
        if not pair:
            return out
        result = self.mode() == "result"
        pk = geometry.params_key(pair["align"]) if result else None
        for side in core.SIDES:
            path = self.project.image_path(pair, side)
            if not (path and os.path.isfile(path) and core.image_size(path)):
                continue
            src_key = f"src|{path}"
            if result:
                out[side] = (f"warp|{side}|{pk}|{path}",
                             task_warp(path, side, pair["align"], self.cache.get(src_key)), src_key)
            else:
                out[side] = (src_key, task_load_source(path), None)
        return out

    def request_layers(self):
        for side, (key, fn, dep) in self.layer_keys().items():
            if key in self.cache:
                self.cache.move_to_end(key)
                self._show_layer(side, key)
            elif key in self.errors or (dep and dep in self.errors):
                err = self.errors.get(key) or self.errors.get(dep)
                for v in self.views:
                    if side in v.frames or not v.frames:
                        v.set_message(f"Не удалось подготовить изображение:\n{err}")
            elif dep and dep not in self.cache:
                self.tasks.submit(dep, task_load_source(dep[4:]))
            else:
                if dep:
                    self.cache.move_to_end(dep)
                self.tasks.submit(key, fn)
        self.apply_layers()

    def _show_layer(self, side, key):
        item = self.cache[key]
        arr = item["arr"]
        h, w = arr.shape[:2]
        if self.mode() == "result":
            x, y, cw, ch = item["rect"]
            for v in self.views:
                v.set_layer(side, arr, QTransform(cw / w, 0, 0, ch / h, x, y), key)
            return
        base = QTransform.fromScale(item["size"][0] / w, item["size"][1] / h)  # предпросмотр → пиксели исходника
        for v in self.views:
            if side in v.frames:
                A, B = v.frames[side]
                v.set_layer(side, arr, base * QTransform(A.real, A.imag, -A.imag, A.real, B.real, B.imag), key)

    def _task_done(self, key, result, err):
        if key.startswith("align|"):
            self._align_done(key, result, err)
            return
        if result is None:
            self.errors[key] = err
        else:
            self.cache[key] = result
            while len(self.cache) > CACHE_ITEMS:
                self.cache.popitem(last=False)
        if self.project:
            self.request_layers()

    def apply_layout(self):
        result = self.mode() == "result"
        single = self.single()
        self.view_cards[1].setVisible(not single)
        pair = self.pair()
        nxt = self.next_point() if pair and not result and self.adding else None
        state = "исправлено" if result else "исходное"
        for i, title in enumerate(self.view_titles):
            target = False
            if not pair or (single and i == 1):
                text = ""
            elif single and self.view_mode == "overlay":
                text = "Наложение: скрещённые поверх параллельных" + \
                    ("" if result else " — исходные (общий сдвиг, поворот и масштаб по точкам)")
            elif single:
                side = self.switch_side
                text = f"{core.SIDE_TITLES[side]} — {state} (Tab — переключить)"
                if nxt and nxt[1] == side:
                    text = f"▶ {core.SIDE_TITLES[side]} — поставьте точку {nxt[0] + 1} (Tab — переключить)"
                    target = True
                elif nxt and nxt[1]:
                    text += f" · точка {nxt[0] + 1} ставится на {geometry.SIDE_GEN[nxt[1]]}"
            else:
                side = core.SIDES[i]
                text = f"{core.SIDE_TITLES[side]} — {state}"
                if nxt and nxt[1] == side:
                    text = f"▶ {core.SIDE_TITLES[side]} — поставьте точку {nxt[0] + 1}"
                    target = True
            title.setText(text)
            if title.property("target") != target:
                title.setProperty("target", target)
                title.style().unpolish(title)
                title.style().polish(title)
        self.apply_layers()

    def apply_layers(self):
        k = self.opacity.value() / 100
        self.opacity_label.setText(f"Прозрачность {self.opacity.value()} %")
        if not self.single():
            if self.mode() == "result":
                self.views[0].set_layer_state({"parallel": True}, {})
                self.views[1].set_layer_state({"crossed": True}, {})
            else:
                for v in self.views:
                    v.set_layer_state({s: True for s in v.frames}, {})
        elif self.view_mode == "overlay":
            self.views[0].set_layer_state({"parallel": True, "crossed": True}, {"crossed": 1 - k})
        else:
            self.views[0].set_layer_state({self.switch_side: True}, {})

    def set_view_mode(self, mode):
        if mode == self.view_mode:
            return
        self.view_mode = mode
        if self.mode() == "points":
            self.rebuild()  # исходники: одно окно в общей системе или два окна в пикселях исходников
            return
        self.apply_layout()
        self.update_panel()
        if mode == "pair":
            QTimer.singleShot(0, lambda: self.view_changed(self.views[0]))

    def toggle_switch(self):
        """Tab: чередование параллельных и скрещённых в одном окне (результат или исходники);
        увеличение и положение сохраняются, точки и совмещение не меняются."""
        if self.mode() == "none":
            return
        if self.view_mode != "switch":
            self.mode_radios["switch"].setChecked(True)
            return
        self.switch_side = core.other_side(self.switch_side)
        self.apply_layout()
        self.update_markers()

    def fit_views(self):
        pair = self.pair()
        rect = None
        if self.mode() == "result" and pair.get("crop") and pair.get("crop") == pair.get("crop_draft"):
            rect = QRectF(*core.output_rect(pair))
        self.syncing += 1
        for v in self.views:
            v.fit(rect)
        self.syncing -= 1

    def sync_active(self):
        return self.mode() == "result" and self.view_mode == "pair" and self.sync_cb.isChecked()

    def _sync_toggled(self, on):
        if on:
            self.view_changed(self.views[0])

    def view_changed(self, view):
        if self.syncing or not self.sync_active():
            return
        other = self.views[1 - view.index]
        self.syncing += 1
        other.setTransform(view.transform())
        other.centerOn(view.center())
        self.syncing -= 1

    # --- точки
    def points(self):
        p = self.pair()
        return p["points"] if p else []

    def next_point(self):
        """(индекс, сторона) — куда ставить следующую точку; сторона None — на любое изображение."""
        pts = self.points()
        if self.sel_point is not None and self.sel_point < len(pts):
            pt = pts[self.sel_point]
            missing = [s for s in core.SIDES if not pt[s]]
            if len(missing) == 1:
                return self.sel_point, missing[0]
        for i, pt in enumerate(pts):
            missing = [s for s in core.SIDES if not pt[s]]
            if len(missing) == 1:
                return i, missing[0]
        return len(pts), None

    def toggle_adding(self):
        pair = self.pair()
        if not pair:
            self.btn_add.setChecked(False)
            return
        self.adding = self.btn_add.isChecked()
        if self.adding and self.mode() == "result":
            self.show_sources = True
            self.rebuild()
            return
        self.apply_layout()
        self.update_markers()
        self.update_panel()

    def add_point(self, side, x, y):
        pair = self.pair()
        if not pair:
            return
        size = core.image_size(self.project.image_path(pair, side))
        if not size or not (0 <= x <= size[0] and 0 <= y <= size[1]):
            self.statusBar().showMessage("Щёлкните внутри изображения", 3000)
            return
        if not self.adding:
            self.statusBar().showMessage("Чтобы поставить новую точку, нажмите «Добавить точки». "
                                         "Существующие точки можно перетаскивать.", 4000)
            return
        self.push_undo()
        idx, need = self.next_point()
        pts = pair["points"]
        if need == side:
            pts[idx][side] = [x, y]
        else:
            pts.append({s: None for s in core.SIDES})
            idx = len(pts) - 1
            pts[idx][side] = [x, y]
        self.sel_point = idx
        self.points_edited()

    def move_point(self, side, idx, x, y):
        pts = self.points()
        if side is None or idx >= len(pts) or not pts[idx][side]:
            return
        old = pts[idx][side]
        if abs(old[0] - x) + abs(old[1] - y) < 1e-6:
            return
        size = core.image_size(self.project.image_path(self.pair(), side))
        x = min(max(x, 0.0), float(size[0]))
        y = min(max(y, 0.0), float(size[1]))
        self.push_undo()
        pts[idx][side] = [x, y]
        self.points_edited()

    def delete_point(self):
        pts = self.points()
        if self.sel_point is None or self.sel_point >= len(pts) or self.mode() != "points":
            return
        self.push_undo()
        del pts[self.sel_point]
        self.sel_point = min(self.sel_point, len(pts) - 1) if pts else None
        self.points_edited()

    def select_point(self, idx, from_view=True):
        if idx is not None and idx >= len(self.points()):
            idx = None
        if idx == self.sel_point:
            return
        self.sel_point = idx
        self.update_markers()
        self.update_panel()
        self.apply_layout()

    def points_edited(self):
        self.update_markers()
        self.apply_layout()
        self.update_panel()
        self.changed()

    def update_markers(self):
        pair = self.pair()
        if not pair:
            return
        if self.mode() == "result":
            specs = []
            if self.common_cb.isChecked():
                specs = [(str(n), x, y, "common", n - 1) for n, x, y in geometry.common_points(pair["align"])]
            for v in self.views:
                v.set_markers(specs)
            return
        pts = pair["points"]
        single = self.single()
        for v in self.views:
            if single:
                sides = [] if v.index else (core.SIDES if self.view_mode == "overlay" else [self.switch_side])
            else:
                sides = list(v.frames)
            specs = []
            for side in sides:
                A, B = v.frames[side]

                def at(xy):
                    z = A * complex(*xy) + B
                    return z.real, z.imag
                for i, pt in enumerate(pts):
                    if pt[side]:
                        specs.append((str(i + 1), *at(pt[side]), "selected" if i == self.sel_point else "normal", i))
                if self.sel_point is not None and self.sel_point < len(pts):
                    pt = pts[self.sel_point]
                    other = core.other_side(side)
                    if not pt[side] and pt[other]:
                        guess = geometry.predict_point(pts, other, pt[other])
                        if guess:
                            specs.append(("?", *at(guess), "ghost", self.sel_point))
            v.set_markers(specs)

    # --- совмещение
    def sizes(self, pair):
        return {s: core.image_size(self.project.image_path(pair, s)) for s in core.SIDES}

    def average_geometry(self):
        pair = self.pair()
        if not pair:
            return
        sizes = self.sizes(pair)
        if not all(sizes.values()):
            QMessageBox.warning(self, "Усреднение геометрии", "Не удалось прочитать одно из изображений пары.")
            return
        errors, _ = geometry.check_points(pair["points"], sizes)
        if errors:
            QMessageBox.warning(self, "Проверьте точки", "\n\n".join(errors))
            return
        snapshot = copy.deepcopy(pair["points"])
        key = f"align|{id(pair)}"
        self.pending_align[key] = (pair, snapshot)
        self.tasks.submit(key, lambda: geometry.compute_alignment(snapshot, sizes))
        self.update_panel()

    def _align_done(self, key, result, err):
        pair, snapshot = self.pending_align.pop(key, (None, None))
        self.update_panel()
        if pair is None or not self.project or pair not in self.project.pairs:
            return
        if pair["points"] != snapshot:
            self.statusBar().showMessage("Точки изменились во время расчёта — нажмите «Усреднить геометрию» ещё раз",
                                         6000)
            return
        if result is None:
            QMessageBox.warning(self, "Усреднение геометрии", err)
            return
        params, field = result
        geometry.put_field(params, field)
        self.push_undo(pair=pair)
        first = not pair["align"]
        pair["align"] = params
        canvas = params["canvas"]
        pair["crop"] = core.clamp_rect(pair.get("crop"), canvas)
        pair["crop_draft"] = core.clamp_rect(pair.get("crop_draft"), canvas)
        self._refresh_pair_item(self.project.pairs.index(pair))
        self.changed()
        if pair is self.pair():
            self.show_sources = False
            self.adding = False
            if first:
                self.sync_cb.setChecked(True)
            self.rebuild()
        if params["warnings"]:
            QMessageBox.information(self, "Усреднение геометрии", "\n\n".join(params["warnings"]))

    def toggle_sources(self):
        self.show_sources = self.btn_sources.isChecked()
        if not self.show_sources:
            self.adding = False
        self.rebuild()

    def reset_alignment(self):
        pair = self.pair()
        if not pair or not pair["align"]:
            return
        self.push_undo()
        pair["align"] = None
        pair["crop"] = pair["crop_draft"] = None
        pair["done"] = False
        self._refresh_pair_item(self.cur)
        self.show_sources = False
        self.changed()
        self.rebuild()

    # --- обрезка
    def crop_rect(self):
        pair = self.pair()
        return pair.get("crop_draft") if pair else None

    def crop_begin(self):
        pair = self.pair()
        self.crop_before = self._snapshot(pair) if pair else None

    def crop_drag(self, hit, p0, p, start):
        pair = self.pair()
        if not pair or not pair["align"]:
            return
        canvas = pair["align"]["canvas"]
        cx0, cy0, cx1, cy1 = canvas[0], canvas[1], canvas[0] + canvas[2], canvas[1] + canvas[3]
        dx, dy = p.x() - p0.x(), p.y() - p0.y()
        if hit == "new" or not start:
            x0, x1 = sorted((p0.x(), p.x()))
            y0, y1 = sorted((p0.y(), p.y()))
        elif hit == "move":
            x0 = min(max(start[0] + dx, cx0), cx1 - start[2])
            y0 = min(max(start[1] + dy, cy0), cy1 - start[3])
            x1, y1 = x0 + start[2], y0 + start[3]
        else:
            x0, y0, x1, y1 = start[0], start[1], start[0] + start[2], start[1] + start[3]
            if "l" in hit:
                x0 += dx
            if "r" in hit:
                x1 += dx
            if "t" in hit:
                y0 += dy
            if "b" in hit:
                y1 += dy
            x0, x1 = sorted((x0, x1))
            y0, y1 = sorted((y0, y1))
        rect = core.clamp_rect([x0, y0, x1 - x0, y1 - y0], canvas)
        if rect:
            pair["crop_draft"] = rect
            self.update_crop()

    def crop_end(self, restore):
        pair = self.pair()
        before, self.crop_before = self.crop_before, None
        if not pair:
            return
        if restore is not None or (before and before["crop_draft"] == pair["crop_draft"]):
            if restore is not None:
                pair["crop_draft"] = restore
            self.update_crop()
            return
        if before is not None:
            self.push_undo(snapshot=before)
        self.changed()
        self.update_crop()
        self.update_panel()

    def update_crop(self):
        pair = self.pair()
        if not pair or self.mode() != "result":
            return
        canvas = pair["align"]["canvas"]
        draft = pair.get("crop_draft")
        applied = bool(draft) and draft == pair.get("crop")
        for v in self.views:
            v.set_crop(canvas, draft, applied)
        self.update_crop_label()

    def update_crop_label(self):
        pair = self.pair()
        if not pair or not pair["align"]:
            self.crop_label.setText("<span style='color:#6b7280'>Доступна после совмещения.</span>")
            return
        draft, crop = pair.get("crop_draft"), pair.get("crop")
        canvas = pair["align"]["canvas"]
        if crop and draft == crop:
            text = f"<b style='color:#16a34a'>Обрезка применена:</b> {core.rect_text(crop)}"
        elif draft:
            text = f"<b style='color:#1d4ed8'>Рамка не применена:</b> {core.rect_text(draft)}"
            if crop:
                text += f"<br>Применённая: {core.rect_text(crop)}"
        else:
            text = (f"Рамки нет — экспортируется весь холст {core.rect_text(canvas)} "
                    "(пустые области — белые).")
        self.crop_label.setText(text)

    def apply_crop(self):
        pair = self.pair()
        if not pair or not pair["align"] or not pair.get("crop_draft"):
            return
        self.push_undo()
        pair["crop"] = list(pair["crop_draft"])
        pair["done"] = True
        self._refresh_pair_item(self.cur)
        self.changed()
        self.update_crop()
        self.update_panel()

    def reset_crop(self):
        pair = self.pair()
        if not pair or not (pair.get("crop") or pair.get("crop_draft")):
            return
        self.push_undo()
        pair["crop"] = pair["crop_draft"] = None
        self.changed()
        self.update_crop()
        self.update_panel()

    def set_done(self, on):
        pair = self.pair()
        if pair and pair["done"] != on:
            pair["done"] = on
            self._refresh_pair_item(self.cur)
            self.changed()

    # --- панель справа
    def update_panel(self):
        pair = self.pair()
        mode = self.mode()
        aligned = core.is_aligned(pair)
        for b in self.pair_buttons:
            b.setEnabled(pair is not None)
        busy_align = pair is not None and f"align|{id(pair)}" in self.pending_align

        pts = self.points()
        self.point_list.blockSignals(True)
        self.point_list.clear()
        for i, pt in enumerate(pts):
            marks = "   ".join(f"{core.SIDE_SHORT[s][:3]}. {'✓' if pt[s] else '—'}" for s in core.SIDES)
            self.point_list.addItem(f"{i + 1}.   {marks}")
        if self.sel_point is not None and self.sel_point < len(pts):
            self.point_list.setCurrentRow(self.sel_point)
        self.point_list.blockSignals(False)

        nxt = self.next_point() if pair else None
        if not pair:
            self.next_label.setText("")
        elif mode == "result":
            self.next_label.setText("Показан результат. Чтобы поправить точки — «Показать исходники».")
        elif not self.adding:
            self.next_label.setText("Нажмите «Добавить точки», чтобы ставить новые точки. "
                                    "Точки можно перетаскивать.")
        elif nxt[1]:
            self.next_label.setText(f"Следующая: точка {nxt[0] + 1} — на {geometry.SIDE_GEN[nxt[1]]} "
                                    "(пунктир — ожидаемое место)")
        else:
            self.next_label.setText(f"Следующая: точка {nxt[0] + 1} — на любом изображении, "
                                    "затем та же деталь на другом")
        self.btn_add.setEnabled(pair is not None)
        self.btn_add.setChecked(self.adding and mode == "points")
        self.btn_del_point.setEnabled(mode == "points" and self.sel_point is not None)
        self.point_list.setEnabled(mode == "points")
        full = len(geometry.complete_points(pts))
        self.btn_average.setEnabled(pair is not None and full >= 2 and not busy_align)
        self.btn_average.setText("Расчёт…" if busy_align else
                                 "Повторить усреднение" if aligned and core.points_changed(pair) else
                                 "Усреднить геометрию")
        self.btn_sources.setEnabled(aligned)
        self.btn_sources.setChecked(aligned and self.show_sources)
        self.btn_reset.setEnabled(aligned)

        lines = []
        if pair:
            lines.append(f"Пар точек: {full}" + (f" (и неполных: {len(pts) - full})" if len(pts) > full else ""))
            if aligned:
                a = pair["align"]
                model = {"similarity": "общее подобие (смещение, поворот, масштаб)",
                         "affine": "аффинное (3 точки)", "tps": "плавная деформация (тонкая пластина)"}[a["model"]]
                lines.append(f"<b style='color:#16a34a'>Совмещено:</b> {model}, точек: {len(a['numbers'])}")
                lines.append(f"Масштаб скрещённые → параллельные: {a['scale']:.4f}, поворот "
                             f"{a['rotation_deg']:+.2f}°")
                lines.append(f"Локальные расхождения до исправления: в среднем {core.fmt_px(a['resid_rms'])} px, "
                             f"наибольшее {core.fmt_px(a['resid_max'])} px (точка {a['resid_point']})")
                lines.append(f"Холст: {a['canvas'][2]} × {a['canvas'][3]} px")
                for w in a["warnings"]:
                    lines.append(f"<span style='color:#ca8a04'>{w}</span>")
                if core.points_changed(pair):
                    lines.append("<b style='color:#ca8a04'>Точки изменены после усреднения — нажмите "
                                 "«Повторить усреднение».</b>")
            else:
                lines.append("<span style='color:#6b7280'>Не совмещено. Поставьте не менее 2 пар точек "
                             "(лучше 6–12 по всей панораме) и нажмите «Усреднить геометрию».</span>")
        self.align_label.setText("<br>".join(lines))

        result = mode == "result"
        for rb in self.mode_radios.values():
            rb.setEnabled(pair is not None)
        self.opacity.setEnabled(pair is not None and self.view_mode == "overlay")
        self.btn_switch.setEnabled(pair is not None)
        self.sync_cb.setEnabled(result and self.view_mode == "pair")
        self.common_cb.setEnabled(result)
        self.btn_apply_crop.setEnabled(result and bool(pair.get("crop_draft")) and
                                       pair.get("crop_draft") != pair.get("crop"))
        self.btn_reset_crop.setEnabled(result and bool(pair.get("crop") or pair.get("crop_draft")))
        self.update_crop_label()
        self.done_cb.blockSignals(True)
        self.done_cb.setChecked(bool(pair) and pair["done"])
        self.done_cb.setEnabled(pair is not None)
        self.done_cb.blockSignals(False)
        busy = self.worker is not None
        self.act_exp_cur.setEnabled(aligned and not busy)
        self.act_exp_all.setEnabled(self.project is not None and not busy)
        if pair:
            info = []
            for s in core.SIDES:
                size = core.image_size(self.project.image_path(pair, s))
                info.append(f"{core.SIDE_SHORT[s]}: {size[0]}×{size[1]} px" if size else
                            f"{core.SIDE_SHORT[s]}: —")
            self.info_label.setText(" · ".join(info))
        else:
            self.info_label.setText("")

    def escape(self):
        if self.adding:
            self.adding = False
            self.apply_layout()
            self.update_panel()

    # --- пары
    def swap_pair(self):
        pair = self.pair()
        if not pair:
            return
        had = bool(pair["align"])
        core.swap_sides(pair)
        self._pair_files_changed()
        if had:
            self.statusBar().showMessage("Изображения поменяны местами — нажмите «Усреднить геометрию» "
                                         "(рамка обрезки сохранена)", 6000)

    def assign_files(self):
        pair = self.pair()
        if not pair:
            return
        try:
            files = core.list_images(self.project.image_dir)
        except OSError as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось прочитать папку изображений:\n{e}")
            return
        dlg = AssignDialog(pair, files, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            if core.reassign(pair, dlg.values()):
                self._pair_files_changed()

    def _pair_files_changed(self):
        self.undo.pop(id(self.pair()), None)
        self.redo.pop(id(self.pair()), None)
        self.sel_point = None
        self.show_sources = False
        self.adding = not core.is_aligned(self.pair())
        self._refresh_pair_item(self.cur)
        self.rebuild(fit=True)
        self.changed()

    def delete_pair(self):
        pair = self.pair()
        if not pair:
            return
        ans = QMessageBox.question(
            self, "Удаление пары",
            f"Удалить «{pair['name']}» из списка проекта?\n\n"
            "Исходные фотографии на диске не удаляются, но точки и обрезка этой пары будут удалены из проекта.")
        if ans != QMessageBox.StandardButton.Yes:
            return
        i = self.cur
        self.undo.pop(id(pair), None)
        self.redo.pop(id(pair), None)
        del self.project.pairs[i]
        self.pair_list.blockSignals(True)
        self.pair_list.takeItem(i)
        self.pair_list.blockSignals(False)
        self.cur, self.sel_point = -1, None
        self.save_project()
        if self.project.pairs:
            n = min(i, len(self.project.pairs) - 1)
            self.pair_list.blockSignals(True)
            self.pair_list.setCurrentRow(n)
            self.pair_list.blockSignals(False)
            self.select_pair(n)
        else:
            self.rebuild()

    # --- отмена / повтор (точки, совмещение, обрезка — отдельно для каждой пары)
    @staticmethod
    def _snapshot(pair):
        return copy.deepcopy({k: pair[k] for k in ("points", "align", "crop", "crop_draft", "done")})

    def push_undo(self, pair=None, snapshot=None):
        pair = pair or self.pair()
        if not pair:
            return
        st = self.undo.setdefault(id(pair), [])
        st.append(snapshot or self._snapshot(pair))
        del st[:-UNDO_LIMIT]
        self.redo[id(pair)] = []

    def _restore(self, src, dst):
        pair = self.pair()
        if not pair:
            return
        stack = src.get(id(pair))
        if not stack:
            return
        dst.setdefault(id(pair), []).append(self._snapshot(pair))
        snap = stack.pop()
        align_changed = snap["align"] != pair["align"]
        pair.update(snap)
        if self.sel_point is not None and self.sel_point >= len(pair["points"]):
            self.sel_point = None
        self._refresh_pair_item(self.cur)
        self.changed()
        if align_changed:
            self.rebuild(fit=False)
        else:
            self.update_markers()
            self.update_crop()
            self.apply_layout()
            self.update_panel()

    def do_undo(self):
        self._restore(self.undo, self.redo)

    def do_redo(self):
        self._restore(self.redo, self.undo)

    # --- экспорт
    def export(self, all_pairs):
        if not self.project or self.worker:
            return
        if all_pairs:
            pairs = [p for p in self.project.pairs if p["done"]]
            if not pairs:
                QMessageBox.information(self, "Экспорт", "Нет пар с отметкой «Пара обработана».")
                return
            not_aligned = [p["name"] for p in pairs if not p["align"]]
            if not_aligned:
                ans = QMessageBox.question(
                    self, "Экспорт", "Отмечены как обработанные, но не совмещены (будут пропущены):\n" +
                    "\n".join(not_aligned[:20]) + "\n\nПродолжить?")
                if ans != QMessageBox.StandardButton.Yes:
                    return
                pairs = [p for p in pairs if p["align"]]
                if not pairs:
                    return
        else:
            pair = self.pair()
            if not pair or not pair["align"]:
                return
            if pair.get("crop_draft") and pair["crop_draft"] != pair.get("crop"):
                box = QMessageBox(QMessageBox.Icon.Question, "Экспорт",
                                  "Рамка обрезки изменена, но не применена.", parent=self)
                apply = box.addButton("Применить и экспортировать", QMessageBox.ButtonRole.AcceptRole)
                box.addButton("Экспортировать без неё", QMessageBox.ButtonRole.ActionRole)
                cancel = box.addButton(QMessageBox.StandardButton.Cancel)
                box.setDefaultButton(apply)
                box.exec()
                if box.clickedButton() == cancel:
                    return
                if box.clickedButton() == apply:
                    self.apply_crop()
            pairs = [pair]
        out = QFileDialog.getExistingDirectory(self, "Папка для экспорта", self.last_export_dir)
        if not out:
            return
        if os.path.normcase(os.path.abspath(out)) == os.path.normcase(os.path.abspath(self.project.image_dir)):
            QMessageBox.warning(self, "Экспорт", "Выберите папку, отличную от папки с исходными фотографиями.")
            return
        fmt = self.project.settings.get("export_format", "source")
        # все файлы — сразу в выбранную папку: проверить совпадения имён и уже существующие файлы
        owner, clash = {}, []
        for p in pairs:
            if not (p["parallel"] and p["crossed"]):
                continue
            for path in core.export_targets(p, out, fmt).values():
                key = os.path.normcase(path)
                if key in owner and owner[key] is not p:
                    clash.append(f"{os.path.basename(path)}: «{owner[key]['name']}» и «{p['name']}»")
                owner.setdefault(key, p)
        if clash:
            QMessageBox.warning(self, "Экспорт", "Разные пары дают файлы с одинаковыми именами — исправьте назначение "
                                "файлов (кнопка «Назначить файлы…»):\n" + "\n".join(clash[:20]))
            return
        existing = [os.path.basename(k) for k in owner if os.path.exists(k)]
        if existing:
            lines = "\n".join(existing[:15]) + (f"\n… и ещё {len(existing) - 15}" if len(existing) > 15 else "")
            ans = QMessageBox.question(self, "Экспорт", f"В папке уже есть файлы с такими именами ({len(existing)}):\n"
                                       f"{lines}\n\nЗаменить их?")
            if ans != QMessageBox.StandardButton.Yes:
                return
        self.last_export_dir = out
        self.save_project()
        self.worker = ExportWorker(copy.deepcopy(pairs), self.project.image_dir, out,
                                   int(self.project.settings["jpeg_quality"]), fmt)
        self.progress = QProgressDialog("Экспорт…", "Отмена", 0, 1000 * len(pairs), self)
        self.progress.setWindowTitle("Экспорт")
        self.progress.setWindowModality(Qt.WindowModality.NonModal)
        self.progress.setMinimumDuration(0)
        self.progress.setAutoClose(False)
        self.progress.setAutoReset(False)
        self.progress.canceled.connect(self._cancel_export)
        self.worker.progress.connect(self._export_progress)
        self.worker.report.connect(lambda e, d, c, det: self._export_done(e, d, c, det, len(pairs), out))
        self.update_panel()
        self.progress.show()
        self.worker.start()

    def _cancel_export(self):
        if self.worker:
            self.worker.cancelled = True
            self.progress.setLabelText("Отмена…")

    def _export_progress(self, value, i, name):
        if self.worker and not self.worker.cancelled:
            self.progress.setValue(int(value * 1000))
            self.progress.setLabelText(f"Экспорт {i + 1} из {self.progress.maximum() // 1000}: {name}")

    def _export_done(self, errors, done, cancelled, details, total, out):
        self.worker.wait()
        self.worker = None
        self.progress.close()
        self.update_panel()
        text = f"{'Экспорт прерван' if cancelled else 'Экспорт завершён'}.\n" \
               f"Экспортировано пар: {done} из {total}.\nПапка: {out}"
        if not errors:
            QMessageBox.information(self, "Экспорт", text)
            return
        log = self.write_error_log(errors, details, out)
        lines = "\n".join(errors[:20]) + (f"\n… и ещё {len(errors) - 20}" if len(errors) > 20 else "")
        box = QMessageBox(QMessageBox.Icon.Warning, "Экспорт",
                          f"{text}\n\nОшибки ({len(errors)}):\n{lines}" +
                          (f"\n\nПодробности сохранены в файл:\n{log}" if log else ""), parent=self)
        box.setDetailedText(details)
        box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        box.exec()

    def write_error_log(self, errors, details, out):
        """Журнал ошибок экспорта рядом с файлом проекта (дописывается)."""
        if not self.project:
            return None
        path = os.path.join(os.path.dirname(self.project.path), "ошибки экспорта.txt")
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(f"\n##### {time.strftime('%Y-%m-%d %H:%M:%S')} — экспорт в {out}\n")
                fh.write(f"Python {sys.version.split()[0]}, Pillow {Image.__version__}, numpy {np.__version__}, "
                         f"OpenCV {cv2.__version__}\n")
                fh.write("\n".join(errors) + "\n\n" + details + "\n")
            return path
        except OSError:
            return None

    def closeEvent(self, e):
        if self.worker:
            ans = QMessageBox.question(self, "Экспорт", "Идёт экспорт. Прервать его и закрыть программу?")
            if ans != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            self.worker.cancelled = True
            self.worker.wait()
        self.save_project()
        self.tasks.pool.clear()
        self.tasks.pool.waitForDone()
        e.accept()


def apply_style(app):
    app.setStyle("Fusion")
    pal = QPalette()
    for role, color in ((QPalette.ColorRole.Window, "#f3f4f6"), (QPalette.ColorRole.Base, "#ffffff"),
                        (QPalette.ColorRole.AlternateBase, "#f9fafb"), (QPalette.ColorRole.Button, "#ffffff"),
                        (QPalette.ColorRole.Text, "#1f2937"), (QPalette.ColorRole.WindowText, "#1f2937"),
                        (QPalette.ColorRole.ButtonText, "#1f2937"), (QPalette.ColorRole.Highlight, "#2563eb"),
                        (QPalette.ColorRole.HighlightedText, "#ffffff"), (QPalette.ColorRole.ToolTipBase, "#111827"),
                        (QPalette.ColorRole.ToolTipText, "#ffffff")):
        pal.setColor(role, QColor(color))
    app.setPalette(pal)
    font = QFont("Segoe UI")
    font.setPointSize(10)
    app.setFont(font)
    app.setStyleSheet(STYLE)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Предварительная обработка панорам шлифов")
    apply_style(app)
    w = MainWindow()
    w.resize(1600, 900)
    w.showMaximized()
    if len(sys.argv) > 1 and sys.argv[1].lower().endswith(".json"):
        try:
            w.set_project(core.Project.load(sys.argv[1]))
        except core.ProjectError as e:
            QMessageBox.warning(w, "Ошибка", str(e))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
