"""Проверка интерфейса без экрана (QT_QPA_PLATFORM=offscreen): увеличение колесом к курсору,
синхронная навигация, переключение параллельных/скрещённых на исходниках и на результате.

Запуск:  python tests/test_gui.py   (или pytest tests)
"""
import math
import os
import shutil
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from PIL import Image  # noqa: E402
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent, QMouseEvent, QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

import app as A  # noqa: E402
import core  # noqa: E402
import test_synthetic as T  # noqa: E402


def ok(cond, text):
    print(("OK  " if cond else "FAIL") + "  " + text)
    assert cond, text


class Harness:
    def __init__(self, workdir):
        self.qa = QApplication.instance() or QApplication(sys.argv)
        A.apply_style(self.qa)
        self.messages = []
        for name in ("warning", "information", "question"):
            def fake(*a, _n=name, **k):
                self.messages.append((_n, a[2] if len(a) > 2 else ""))
                return QMessageBox.StandardButton.Yes
            setattr(QMessageBox, name, staticmethod(fake))
        QMessageBox.exec = lambda box: self.messages.append(("box", box.text(), box.detailedText())) or 0
        self.workdir = workdir
        img_dir = os.path.join(workdir, "фото")
        os.makedirs(img_dir)
        st = T.make_structure()
        Image.fromarray(T.render_source("parallel", st)).save(os.path.join(img_dir, "5. Ш-1.jpg"), quality=95)
        Image.fromarray(T.render_source("crossed", st)).save(os.path.join(img_dir, "5. Ш-1 с.jpg"), quality=95)
        self.project = core.Project(os.path.join(workdir, "проект.json"), img_dir, core.scan_folder(img_dir))
        self.project.pairs[0]["points"] = T.landmarks()
        self.w = A.MainWindow()
        self.w.resize(1700, 950)
        self.w.show()
        self.w.set_project(self.project)
        self.wait()

    def pump(self, sec=0.15):
        end = time.time() + sec
        while time.time() < end:
            self.qa.processEvents()
            time.sleep(0.01)

    def wait(self, timeout=60):
        t = time.time()
        self.pump(0.1)
        while self.w.tasks.running and time.time() - t < timeout:
            self.pump(0.05)
        self.pump(0.2)

    def wheel(self, view, pos, dy):
        ev = QWheelEvent(QPointF(pos), QPointF(view.viewport().mapToGlobal(pos)), QPoint(0, 0), QPoint(0, dy),
                         Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase,
                         False)
        self.qa.sendEvent(view.viewport(), ev)

    def click(self, view, scene_pt):
        vp = view.mapFromScene(scene_pt)
        for typ, btns in ((QEvent.Type.MouseButtonPress, Qt.MouseButton.LeftButton),
                          (QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton)):
            ev = QMouseEvent(typ, QPointF(vp), QPointF(view.viewport().mapToGlobal(vp)), Qt.MouseButton.LeftButton,
                             btns, Qt.KeyboardModifier.NoModifier)
            self.qa.sendEvent(view.viewport(), ev)

    def tab(self, view):
        self.qa.sendEvent(view, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Tab, Qt.KeyboardModifier.NoModifier))


def view_state(v):
    c = v.center()
    return round(v.scale_factor(), 9), round(c.x(), 3), round(c.y(), 3)


def check_wheel(h, view, label):
    """Серия шагов колеса: исходная точка сцены всё время остаётся под курсором."""
    pos = QPoint(140, 230)
    anchor = view.mapToScene(pos)
    worst = 0.0
    for dy in [120] * 12 + [-120] * 18 + [240] * 3:
        h.wheel(view, pos, dy)
        q = view.mapFromScene(anchor)
        worst = max(worst, math.hypot(q.x() - pos.x(), q.y() - pos.y()))
    ok(worst <= 1.0, f"{label}: точка под курсором на месте при увеличении и уменьшении "
                     f"(33 шага, отклонение ≤ {worst:.2f} px экрана)")


def run(workdir):
    h = Harness(workdir)
    w = h.w
    pair = w.pair()
    v0, v1 = w.views

    # --- исходники, два окна: независимая навигация
    ok(w.mode() == "points" and v1.isVisible(), "исходники показаны рядом")
    before1 = view_state(v1)
    check_wheel(h, v0, "исходники")
    ok(view_state(v1) == before1, "при расстановке точек навигация независимая (второе окно не изменилось)")

    # --- результат, синхронная навигация
    w.average_geometry()
    h.wait()
    ok(w.mode() == "result" and w.sync_cb.isChecked(), "после усреднения включена синхронная навигация")
    calls = []
    orig = w.view_changed

    def counting(view):
        calls.append((view.index, w.syncing == 0))  # False — вызов из самой синхронизации, сразу отбрасывается
        return orig(view)
    w.view_changed = counting
    check_wheel(h, v0, "результат")
    applied = [i for i, live in calls if live]
    ok(len(applied) == 33 and set(applied) == {0},
       f"нет взаимного зацикливания: на шаг колеса ровно одна синхронизация ({len(applied)} на 33 шага), "
       "ответных от второго окна нет")
    w.view_changed = orig
    ok(abs(v0.scale_factor() - v1.scale_factor()) < 1e-9 and
       (v0.center() - v1.center()).manhattanLength() * v0.scale_factor() <= 2,
       "второе окно показывает тот же участок с тем же увеличением")
    check_wheel(h, v1, "результат, колесо во втором окне")
    ok((v0.center() - v1.center()).manhattanLength() * v0.scale_factor() <= 2, "синхронизация в обратную сторону")

    # --- переключение на результате
    w.mode_radios["switch"].setChecked(True)
    h.pump()
    st = view_state(v0)
    side = w.switch_side
    h.tab(v0)
    h.pump()
    ok(w.switch_side != side and view_state(v0) == st and v0.layers[w.switch_side].isVisible() and
       not v0.layers[side].isVisible(), "Tab на результате: другое изображение, увеличение и положение те же")

    # --- исходники в режиме «Переключение»: тот же кадр, что у результата
    align = core.clone(pair["align"])
    points = core.clone(pair["points"])
    w.btn_sources.setChecked(True)
    w.toggle_sources()
    h.wait()
    ok(w.mode() == "points" and w.single() and not v1.isVisible(), "исходники в одном окне")
    ok(abs(view_state(v0)[0] - st[0]) < 1e-9 and math.hypot(view_state(v0)[1] - st[1], view_state(v0)[2] - st[2])
       * st[0] <= 1.5, "переход к исходникам сохраняет увеличение и положение")
    st = view_state(v0)
    for _ in range(3):
        side = w.switch_side
        h.tab(v0)
        h.pump()
        ok(w.switch_side != side and view_state(v0) == st and v0.layers[w.switch_side].isVisible() and
           not v0.layers[side].isVisible(), f"Tab на исходниках: {core.SIDE_SHORT[w.switch_side].lower()}, "
                                            "увеличение и положение сохранены")
    ok(pair["points"] == points and pair["align"] == align, "переключение не меняет точки и совмещение")

    # соответствие: деталь i на обоих исходниках оказывается рядом (общее подобие), слой построен верно
    for side in core.SIDES:
        item = v0.layers[side]
        img = w.cache[v0.layer_keys[side]]
        sx = img["arr"].shape[1] / img["size"][0]
        sy = img["arr"].shape[0] / img["size"][1]
        p = pair["points"][0][side]
        mapped = item.sceneTransform().map(QPointF(p[0] * sx, p[1] * sy))
        A_, B_ = v0.frames[side]
        z = A_ * complex(*p) + B_
        ok(abs(mapped.x() - z.real) + abs(mapped.y() - z.imag) < 1e-6,
           f"слой исходника ({core.SIDE_SHORT[side].lower()}) размещён в общей системе")
    gaps = []
    for pt in pair["points"]:
        zp = v0.frames["parallel"][0] * complex(*pt["parallel"]) + v0.frames["parallel"][1]
        zc = v0.frames["crossed"][0] * complex(*pt["crossed"]) + v0.frames["crossed"][1]
        gaps.append(abs(zp - zc))
    ok(max(gaps) < 25, f"на исходниках детали совпадают с точностью до локальных расхождений "
                       f"(до обработки: до {max(gaps):.1f} px)")

    # точка ставится и двигается на видимом исходнике
    w.adding = True
    w.switch_side = "crossed"
    w.apply_layout()
    w.update_markers()
    target = pair["points"][2]["crossed"]
    A_, B_ = v0.frames["crossed"]
    z = A_ * complex(*target) + B_
    n = len(pair["points"])
    h.click(v0, QPointF(z.real, z.imag))  # щелчок рядом с точкой 3 захватывает её (не создаёт новую)
    h.pump()
    ok(len(pair["points"]) == n and w.sel_point == 2, "щелчок по точке выбирает её на видимом исходнике")
    free = A_ * complex(300.0, 900.0) + B_
    h.click(v0, QPointF(free.real, free.imag))
    h.pump()
    new = pair["points"][-1]
    ok(len(pair["points"]) == n + 1 and new["crossed"] and new["parallel"] is None and
       math.dist(new["crossed"], (300.0, 900.0)) < 3 / v0.scale_factor() / abs(A_) + 1,
       f"новая точка поставлена на скрещённых в координатах исходника: "
       f"({new['crossed'][0]:.1f}, {new['crossed'][1]:.1f})")
    w.do_undo()
    h.pump()
    ok(pair["points"] == points, "отмена возвращает точки")

    # обратно к результату — то же положение
    st = view_state(v0)
    w.btn_sources.setChecked(False)
    w.toggle_sources()
    h.wait()
    ok(w.mode() == "result" and math.hypot(view_state(v0)[1] - st[1], view_state(v0)[2] - st[2]) * st[0] <= 1.5
       and abs(view_state(v0)[0] - st[0]) < 1e-9, "возврат к результату: увеличение и положение сохранены")
    ok(pair["align"] == align, "результат не сброшен")

    # «Рядом» на исходниках снова два окна в пикселях исходников
    w.btn_sources.setChecked(True)
    w.toggle_sources()
    w.mode_radios["pair"].setChecked(True)
    h.wait()
    ok(v1.isVisible() and v0.side == "parallel" and v1.side == "crossed", "«Рядом» на исходниках: два окна")

    # --- экспорт: все файлы сразу в выбранную папку, в исходном формате
    out = os.path.join(h.workdir, "результат")
    os.makedirs(out)
    A.QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: out)

    def export_now():
        w.export(False)
        t = time.time()
        while w.worker is not None and time.time() - t < 120:
            h.pump(0.05)
        h.pump()
    ok(w.format_combo.currentData() == "source", "по умолчанию — исходный формат")
    export_now()
    ok(sorted(os.listdir(out)) == ["5. Ш-1 с.jpg", "5. Ш-1.jpg"], f"экспорт без подпапок: {sorted(os.listdir(out))}")
    n = len(h.messages)
    export_now()
    ok(any(m[0] == "question" and "Заменить" in m[1] for m in h.messages[n:]), "повторный экспорт спрашивает о замене")
    # ошибка: имя файла занято папкой → понятное сообщение, подробности и журнал рядом с проектом
    shutil.rmtree(out)
    os.makedirs(os.path.join(out, "5. Ш-1.jpg"))
    n = len(h.messages)
    export_now()
    box = [m for m in h.messages[n:] if m[0] == "box"]
    log = os.path.join(h.workdir, "ошибки экспорта.txt")
    ok(box and "запись 5. Ш-1.jpg" in box[0][1] and "Traceback" in box[0][2] and os.path.isfile(log),
       "ошибка экспорта: шаг и файл в сообщении, подробности и журнал «ошибки экспорта.txt»")
    w.close()
    h.pump()


def test_gui():
    d = tempfile.mkdtemp(prefix="pano_gui_")
    try:
        run(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    test_gui()
    print("\nВсе проверки пройдены.")
