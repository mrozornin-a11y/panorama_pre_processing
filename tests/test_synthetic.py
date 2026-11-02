"""Проверка на синтетической паре с известными локальными искажениями.

Запуск:  python tests/test_synthetic.py   (или pytest tests)

Синтетика: общая «истинная» структура шлифа T; параллельные и скрещённые николи — разные раскраски этой
структуры, снятые с разным разрешением, поворотом, смещением и с разными локальными искажениями
(гауссовы «вздутия» поля смещений). Проверяется:
  1) изменяются обе панорамы, а соответствующие детали сближаются (по точкам и по изображению);
  2) цвета каждой панорамы сохраняются (раскраски не смешиваются);
  3) экспорт совпадает с предпросмотром;
  4) общая обрезка даёт одинаковые размеры;
  5) сохранение и повторное открытие проекта восстанавливают обработку;
  6) результат распознаётся как обычная пара по правилам panorama_marking;
  7) вырожденные и ошибочные точки дают понятные сообщения.
"""
import math
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import core  # noqa: E402
import geometry  # noqa: E402

PREVIEW_MAX = 1024
RNG = np.random.default_rng(7)

# --- искажения: исходник (px) → истинная система T
SIM = {
    "parallel": dict(scale=1.0, rot=0.0, off=(100.0, 90.0)),
    "crossed": dict(scale=1.25, rot=3.0, off=(70.0, 150.0)),   # скрещённые сняты с меньшим разрешением
}
BUMP = {
    "parallel": [((800.0, 600.0), 380.0, (22.0, 6.0)), ((1900.0, 1300.0), 300.0, (-8.0, 14.0))],
    "crossed": [((1250.0, 750.0), 320.0, (5.0, -20.0)), ((450.0, 1150.0), 280.0, (16.0, 9.0))],
}
SIZES = {"parallel": (2400, 1800), "crossed": (1900, 1430)}
T_SIZE = (2700, 2100)


def to_truth(side, x, y):
    s = SIM[side]
    a = s["scale"] * np.exp(1j * math.radians(s["rot"]))
    z = a * (np.asarray(x) + 1j * np.asarray(y)) + complex(*s["off"])
    for (cx, cy), sig, (dx, dy) in BUMP[side]:
        g = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sig * sig))
        z = z + g * complex(dx, dy)
    return z.real, z.imag


def from_truth(side, X, Y):
    """Обратное отображение итерациями."""
    s = SIM[side]
    a = s["scale"] * np.exp(1j * math.radians(s["rot"]))
    T = np.asarray(X) + 1j * np.asarray(Y)
    z = (T - complex(*s["off"])) / a
    for _ in range(60):
        tx, ty = to_truth(side, z.real, z.imag)
        z = z - ((tx + 1j * ty) - T) / a
    return z.real, z.imag


def make_structure():
    h, w = T_SIZE[1], T_SIZE[0]
    s = np.zeros((h, w), np.float32)
    for sigma, amp in ((60, 1.0), (15, 0.6), (4, 0.35)):
        n = RNG.standard_normal((h, w)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), sigma)
        s += amp * n / (n.std() + 1e-6)
    # «зёрна»: случайные эллипсы с резкими границами
    img = np.zeros((h, w), np.uint8)
    for _ in range(400):
        c = (int(RNG.integers(0, w)), int(RNG.integers(0, h)))
        ax = (int(RNG.integers(8, 60)), int(RNG.integers(8, 60)))
        cv2.ellipse(img, c, ax, float(RNG.uniform(0, 180)), 0, 360, int(RNG.integers(60, 255)), -1)
    s = (s - s.min()) / (s.max() - s.min()) * 120 + img * 0.5
    return np.clip(s, 0, 255).astype(np.float32)


def palette(side, s):
    """Разные раскраски одной структуры: у параллельных синий канал ≈ 30, у скрещённых красный ≈ 30."""
    if side == "parallel":
        return np.dstack([s, 0.55 * s + 60, np.full_like(s, 30)])
    return np.dstack([np.full_like(s, 30), 255 - s, 0.8 * s + 20])


def render_source(side, structure):
    w, h = SIZES[side]
    xs, ys = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    tx, ty = to_truth(side, xs, ys)
    s = cv2.remap(structure, (tx - 0.5).astype(np.float32), (ty - 0.5).astype(np.float32),
                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return np.clip(palette(side, s), 0, 255).astype(np.uint8)


def landmarks():
    pts = []
    for gx in np.linspace(0.08, 0.92, 5):
        for gy in np.linspace(0.1, 0.9, 4):
            pts.append((gx * SIZES["parallel"][0], gy * SIZES["parallel"][1]))
    pts = np.array(pts)
    tx, ty = to_truth("parallel", pts[:, 0], pts[:, 1])
    qx, qy = from_truth("crossed", tx, ty)
    out = []
    for i in range(len(pts)):
        if 10 < qx[i] < SIZES["crossed"][0] - 10 and 10 < qy[i] < SIZES["crossed"][1] - 10:
            out.append({"parallel": [float(pts[i, 0]), float(pts[i, 1])],
                        "crossed": [float(qx[i]), float(qy[i])]})
    return out


def grad_mag(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.5)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def ncc(a, b, mask):
    a, b = a[mask], b[mask]
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))


def similarity_only(params):
    p = dict(params)
    p["model"] = "similarity"
    return geometry.Field(p)


def run(workdir):
    report = []

    def ok(cond, text):
        report.append(("OK  " if cond else "FAIL") + "  " + text)
        print(report[-1])
        assert cond, text

    structure = make_structure()
    img_dir = os.path.join(workdir, "фото")
    os.makedirs(img_dir)
    src = {}
    for side, fname in (("parallel", "16. Г-170.jpg"), ("crossed", "16. Г-170 с.png")):
        src[side] = render_source(side, structure)
        im = Image.fromarray(src[side])
        if fname.endswith(".png"):
            im.save(os.path.join(img_dir, fname))
        else:
            im.save(os.path.join(img_dir, fname), quality=97, subsampling=0)
    # латинская «c» тоже распознаётся
    pairs = core.scan_folder(img_dir)
    ok(len(pairs) == 1 and pairs[0]["parallel"] == "16. Г-170.jpg" and pairs[0]["crossed"] == "16. Г-170 с.png"
       and pairs[0]["name"] == "16. Г-170", "пара определена по имени (номер сохранён)")
    tmp_dir2 = os.path.join(workdir, "латиница")
    os.makedirs(tmp_dir2)
    for f in ("3. A-1.jpg", "3. A-1 c.jpg"):
        Image.new("RGB", (8, 8)).save(os.path.join(tmp_dir2, f))
    p2 = core.scan_folder(tmp_dir2)
    ok(len(p2) == 1 and p2[0]["crossed"] == "3. A-1 c.jpg", "латинская «c» — скрещённые николи")

    pair = pairs[0]
    pair["points"] = landmarks()
    sizes = {s: core.image_size(os.path.join(img_dir, pair[s])) for s in core.SIDES}
    params, field = geometry.compute_alignment(pair["points"], sizes)
    pair["align"] = params
    ok(params["model"] == "tps", f"модель — плавная деформация (TPS), пар точек: {len(pair['points'])}")
    ok(abs(params["scale"] - 1.25) < 0.03, f"масштаб между исходниками оценён: {params['scale']:.4f} (≈1.25)")
    ok(abs(params["rotation_deg"] - 3.0) < 0.5, f"поворот оценён: {params['rotation_deg']:.2f}° (≈3°)")

    # 1. обе панорамы изменяются, детали сближаются
    alpha = complex(*params["alpha"])
    ok(abs(abs(alpha) - math.sqrt(1.25)) < 0.02, "масштаб общей системы — среднее геометрическое (√s)")
    shift = np.hypot(field.ux, field.uy).max()
    ok(shift > 5, f"локальная поправка ненулевая: до {shift:.1f} px; параллельным +u, скрещённым −u (50/50)")
    sim = similarity_only(params)
    test = RNG.uniform([150, 150], [2250, 1650], size=(300, 2))
    tx, ty = to_truth("parallel", test[:, 0], test[:, 1])
    qx, qy = from_truth("crossed", tx, ty)
    inside = (qx > 50) & (qx < SIZES["crossed"][0] - 50) & (qy > 50) & (qy < SIZES["crossed"][1] - 50)
    before, after, mp, mc = [], [], [], []
    for i in np.nonzero(inside)[0]:
        a = np.array(sim.to_common("parallel", *test[i]))
        b = np.array(sim.to_common("crossed", qx[i], qy[i]))
        c = np.array(field.to_common("parallel", *test[i]))
        d = np.array(field.to_common("crossed", qx[i], qy[i]))
        before.append(np.linalg.norm(a - b))
        after.append(np.linalg.norm(c - d))
        mp.append(np.linalg.norm(c - a))
        mc.append(np.linalg.norm(d - b))
    before, after = np.array(before), np.array(after)
    ok(np.median(mp) > 1 and np.median(mc) > 1,
       f"изменяются обе панорамы: медианный сдвиг деталей параллельных {np.median(mp):.1f} px, "
       f"скрещённых {np.median(mc):.1f} px")
    ok(np.median(after) < 0.35 * np.median(before) and after.max() < before.max(),
       f"расхождение деталей (300 контрольных точек, не опорных): медиана {np.median(before):.1f} → "
       f"{np.median(after):.2f} px, максимум {before.max():.1f} → {after.max():.1f} px")
    lm = [np.linalg.norm(np.array(field.to_common("parallel", *p["parallel"])) -
                         np.array(field.to_common("crossed", *p["crossed"]))) for p in pair["points"]]
    ok(max(lm) < 0.5, f"опорные точки совпадают: максимум {max(lm):.3f} px")

    # изображения: корреляция границ после совмещения выше, чем после одного общего подобия
    canvas = params["canvas"]
    k = 0.5
    imgs = {s: field.render(s, src[s], SIZES[s], canvas, k) for s in core.SIDES}
    imgs_sim = {s: sim.render(s, src[s], SIZES[s], canvas, k) for s in core.SIDES}
    mask = np.ones(imgs["parallel"].shape[:2], bool)
    for s in core.SIDES:
        mask &= ~np.all(imgs[s] == 255, axis=2) & ~np.all(imgs_sim[s] == 255, axis=2)
    mask = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    n_before = ncc(grad_mag(imgs_sim["parallel"]), grad_mag(imgs_sim["crossed"]), mask)
    n_after = ncc(grad_mag(imgs["parallel"]), grad_mag(imgs["crossed"]), mask)
    ok(n_after > n_before + 0.05 and n_after > 0.8,
       f"совпадение границ изображений (корреляция): {n_before:.3f} → {n_after:.3f}")

    # 2. цвета сохраняются
    for s in core.SIDES:
        valid = ~np.all(imgs[s] == 255, axis=2)
        mo, ms = imgs[s][valid].mean(axis=0), src[s].reshape(-1, 3).mean(axis=0)
        ok(np.abs(mo - ms).max() < 6, f"цвета сохранены ({core.SIDE_SHORT[s].lower()}): средний RGB "
           f"{np.round(ms).astype(int).tolist()} → {np.round(mo).astype(int).tolist()}")
    ok(abs(imgs["parallel"][..., 2][mask].mean() - 30) < 3 and abs(imgs["crossed"][..., 0][mask].mean() - 30) < 3,
       "раскраски не смешаны (синий канал параллельных и красный скрещённых остались ≈30)")

    # 4. общая обрезка
    cx, cy, cw, ch = canvas
    crop = [cx + cw // 6, cy + ch // 7, cw * 2 // 3, ch * 5 // 7]
    pair["crop"] = core.clamp_rect(crop, canvas)
    pair["crop_draft"] = list(pair["crop"])
    pair["done"] = True
    hashes = {s: file_hash(os.path.join(img_dir, pair[s])) for s in core.SIDES}
    out_dir = os.path.join(workdir, "экспорт")
    folder = out_dir
    written = core.export_pair(pair, img_dir, out_dir, 95)   # по умолчанию — исходный формат
    names = core.export_names(pair)
    ok(names == {"parallel": "16. Г-170.jpg", "crossed": "16. Г-170 с.png"},
       f"исходный формат и базовые имена: {names['parallel']}, {names['crossed']}")
    ok(sorted(os.listdir(out_dir)) == sorted(names.values()) and
       sorted(written) == sorted(os.path.join(out_dir, n) for n in names.values()),
       "файлы записаны сразу в выбранную папку, без подпапок")
    ok(Image.open(os.path.join(out_dir, names["crossed"])).format == "PNG" and
       Image.open(os.path.join(out_dir, names["parallel"])).format == "JPEG", "PNG сохранён в PNG, JPEG — в JPEG")
    outs = {s: np.asarray(Image.open(os.path.join(folder, names[s])).convert("RGB")) for s in core.SIDES}
    ok(outs["parallel"].shape == outs["crossed"].shape == (pair["crop"][3], pair["crop"][2], 3),
       f"общая обрезка: оба файла {outs['parallel'].shape[1]}×{outs['parallel'].shape[0]} px")
    direct = field.render("crossed", np.asarray(Image.open(os.path.join(img_dir, pair["crossed"])).convert("RGB")),
                          SIZES["crossed"], pair["crop"], block=2048)
    ok(np.array_equal(direct, outs["crossed"]), "PNG-результат без потерь: совпадает с расчётом побитово")
    out_jpg = os.path.join(workdir, "экспорт_jpeg")
    core.export_pair(pair, img_dir, out_jpg, 95, fmt="jpeg")
    ok(sorted(os.listdir(out_jpg)) == ["16. Г-170 с.jpg", "16. Г-170.jpg"], "режим «Всё в JPEG»: оба файла .jpg")
    bad_dir = os.path.join(workdir, "экспорт_ошибка")
    os.makedirs(os.path.join(bad_dir, names["parallel"]))  # имя занято папкой — запись невозможна
    try:
        core.export_pair(pair, img_dir, bad_dir, 95)
        ok(False, "ошибка записи должна сообщаться")
    except core.ExportError as e:
        ok(str(e).startswith("запись 16. Г-170.jpg"), f"ошибка экспорта указывает шаг и файл: {str(e)[:60]}…")
    ok(not [f for f in os.listdir(bad_dir) if f.endswith(".part")], "после ошибки временные файлы удалены")
    ok(all(file_hash(os.path.join(img_dir, pair[s])) == hashes[s] for s in core.SIDES), "исходные файлы не изменены")

    # 3. экспорт = предпросмотр (предпросмотр строится из уменьшенных копий, как в программе)
    for s in core.SIDES:
        prev_src, full_size = core.load_preview(os.path.join(img_dir, pair[s]), PREVIEW_MAX)
        kp = PREVIEW_MAX / max(canvas[2], canvas[3])
        prev = field.render(s, prev_src, full_size, canvas, kp)
        # фрагмент предпросмотра, соответствующий обрезке
        x0 = (pair["crop"][0] - cx) * prev.shape[1] / cw
        y0 = (pair["crop"][1] - cy) * prev.shape[0] / ch
        x1 = x0 + pair["crop"][2] * prev.shape[1] / cw
        y1 = y0 + pair["crop"][3] * prev.shape[0] / ch
        w, h = round(x1 - x0), round(y1 - y0)
        gx = (x0 + (np.arange(w) + 0.5) * (x1 - x0) / w - 0.5).astype(np.float32)
        gy = (y0 + (np.arange(h) + 0.5) * (y1 - y0) / h - 0.5).astype(np.float32)
        mx, my = np.meshgrid(gx, gy)
        prev_crop = cv2.remap(prev, mx, my, cv2.INTER_LINEAR)
        exp_small = cv2.resize(outs[s], (w, h), interpolation=cv2.INTER_AREA)
        inner = (slice(4, h - 4), slice(4, w - 4))
        diff = np.abs(prev_crop[inner].astype(int) - exp_small[inner].astype(int))
        sh = phase_shift(prev_crop[inner], exp_small[inner])
        ok(diff.mean() < 3 and np.hypot(*sh) < 0.15,
           f"экспорт совпадает с предпросмотром ({core.SIDE_SHORT[s].lower()}): средняя разница "
           f"{diff.mean():.2f} уровня, сдвиг {np.hypot(*sh):.3f} px предпросмотра")

    # 5. сохранение и повторное открытие проекта
    proj_path = os.path.join(workdir, "проект.json")
    project = core.Project(proj_path, img_dir, [pair])
    project.save()
    loaded = core.Project.load(proj_path)
    lp = loaded.pairs[0]
    ok(lp["points"] == pair["points"] and lp["crop"] == pair["crop"] and lp["done"] and
       lp["align"]["canvas"] == canvas, "проект: точки, рамка, отметка и параметры восстановлены")
    ok(not core.points_changed(lp), "после открытия точки соответствуют совмещению")
    f2 = geometry.Field(lp["align"])
    ok(np.array_equal(f2.ux, field.ux) and np.array_equal(f2.uy, field.uy),
       "поле деформации после открытия совпадает побитово")
    out2 = os.path.join(workdir, "экспорт2")
    core.export_pair(lp, loaded.image_dir, out2, 95)
    folder2 = out2
    same = all(np.array_equal(np.asarray(Image.open(os.path.join(folder2, names[s]))), outs[s]) for s in core.SIDES)
    ok(same, "экспорт после повторного открытия проекта идентичен")

    # 6. panorama_marking распознаёт результат как обычную пару
    # путь к копии panorama_marking (необязательно): переменная окружения или соседняя папка
    pm = os.environ.get("PANORAMA_MARKING") or next(
        (p for p in (os.path.join(os.path.dirname(os.path.dirname(HERE)), "panorama_marking"),
                     os.path.join(os.path.dirname(os.path.dirname(HERE)), "mrozornin-a11y", "panorama_marking"))
         if os.path.isfile(os.path.join(p, "core.py"))), "")
    exp_pairs = core.scan_folder(folder)   # те же правила, что в panorama_marking
    ok(len(exp_pairs) == 1 and not exp_pairs[0]["problem"] and exp_pairs[0]["name"] == "16. Г-170",
       "результат распознаётся как пара по правилам panorama_marking")
    if os.path.isfile(os.path.join(pm, "core.py")):
        import importlib.util
        spec = importlib.util.spec_from_file_location("pm_core", os.path.join(pm, "core.py"))
        pmc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pmc)
        pp = pmc.scan_folder(folder)
        ok(len(pp) == 1 and not pp[0]["problem"], "core.py из panorama_marking загружает результат как пару")

    # 7. проблемы с точками
    bad = [dict(p) for p in pair["points"]]
    bad[3] = {"parallel": bad[3]["parallel"], "crossed": bad[12]["crossed"]}
    bad[12] = {"parallel": bad[12]["parallel"], "crossed": pair["points"][3]["crossed"]}
    try:
        geometry.compute_alignment(bad, sizes)
        ok(False, "перепутанные точки должны давать сообщение")
    except geometry.AlignError as e:
        ok("складк" in str(e) or "зеркальн" in str(e), "перепутанные точки → сообщение: " + str(e).split(".")[0])
    line = [{"parallel": [200.0 + 400 * i, 300.0 + 200 * i], "crossed": None} for i in range(4)]
    for i, p in enumerate(line):
        tx, ty = to_truth("parallel", *p["parallel"])
        qx, qy = from_truth("crossed", tx, ty)
        p["crossed"] = [float(qx), float(qy)]
    pl, _ = geometry.compute_alignment(line, sizes)
    ok(pl["model"] == "similarity" and pl["warnings"], "точки на одной прямой → только общее подобие + предупреждение")
    mirror = [{"parallel": p["parallel"], "crossed": [SIZES["crossed"][0] - p["crossed"][0], p["crossed"][1]]}
              for p in pair["points"]]
    errs, _ = geometry.check_points(mirror, sizes)
    ok(any("зеркальн" in e for e in errs), "зеркальный порядок точек → сообщение")
    errs, _ = geometry.check_points(pair["points"][:1], sizes)
    ok(bool(errs), "одна пара точек → сообщение")
    half = pair["points"][:4] + [{"parallel": [10.0, 10.0], "crossed": None}]
    errs, _ = geometry.check_points(half, sizes)
    ok(any("только на" in e for e in errs), "точка только на одном изображении → сообщение")
    return report


def file_hash(path):
    import hashlib
    with open(path, "rb") as fh:
        return hashlib.sha1(fh.read()).hexdigest()


def phase_shift(a, b):
    """Субпиксельный сдвиг b относительно a (один шаг Лукаса–Канаде по градиентам)."""
    ga = cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64), (0, 0), 1.0)
    gb = cv2.GaussianBlur(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64), (0, 0), 1.0)
    gx = (np.roll(ga, -1, 1) - np.roll(ga, 1, 1)) / 2
    gy = (np.roll(ga, -1, 0) - np.roll(ga, 1, 0)) / 2
    sl = (slice(2, -2), slice(2, -2))
    gx, gy, dt = gx[sl], gy[sl], (ga - gb)[sl]
    A = np.array([[(gx * gx).sum(), (gx * gy).sum()], [(gx * gy).sum(), (gy * gy).sum()]])
    dx, dy = np.linalg.solve(A, [(gx * dt).sum(), (gy * dt).sum()])
    return dx, dy


def test_synthetic():
    d = tempfile.mkdtemp(prefix="pano_test_")
    try:
        run(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    test_synthetic()
    print("\nВсе проверки пройдены.")
