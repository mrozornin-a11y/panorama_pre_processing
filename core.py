"""Логика без GUI: распознавание пар, проект (JSON), чтение изображений и экспорт."""
import copy
import json
import os
import re
import unicodedata

import numpy as np
from PIL import Image

import geometry

Image.MAX_IMAGE_PIXELS = None  # панорамы бывают очень большими

SIDES = geometry.SIDES
SIDE_TITLES = {"parallel": "Параллельные николи", "crossed": "Скрещённые николи"}
SIDE_SHORT = {"parallel": "Параллельные", "crossed": "Скрещённые"}

IMAGE_EXT = {".png", ".jpg", ".jpeg"}
JPEG_MAX_SIDE = 65500
DEFAULT_SETTINGS = {"jpeg_quality": 95, "export_format": "source"}
EXPORT_FORMATS = {"source": "Исходный", "jpeg": "JPEG"}


# ---------------------------------------------------------------- пары файлов (правила panorama_marking)

_CROSSED_RE = re.compile(r"^(.*\S)\s+[сСcC]$")  # кириллическая или латинская «с»


def natural_key(s):
    return [int(t) if t.isdigit() else t.casefold() for t in re.split(r"(\d+)", s)]


def list_images(folder):
    files = []
    for f in os.listdir(folder):
        if os.path.splitext(f)[1].lower() in IMAGE_EXT and os.path.isfile(os.path.join(folder, f)):
            files.append(f)
    return sorted(files, key=natural_key)


def split_name(filename):
    """Возвращает (базовое имя, скрещённые ли николи)."""
    stem = unicodedata.normalize("NFC", os.path.splitext(filename)[0]).strip()
    m = _CROSSED_RE.match(stem)
    if m:
        return m.group(1).strip(), True
    return stem, False


def other_side(side):
    return SIDES[1] if side == SIDES[0] else SIDES[0]


def new_pair(name, parallel, crossed, problem=""):
    return {"name": name, "parallel": parallel, "crossed": crossed, "problem": problem,
            "points": [],          # [{"parallel": [x, y] | None, "crossed": [x, y] | None}]
            "align": None,         # параметры совмещения (geometry.compute_alignment)
            "crop": None,          # применённая обрезка [x, y, w, h] в общей системе координат
            "crop_draft": None,    # рамка на экране (ещё не применённая)
            "done": False}


def scan_folder(folder):
    groups = {}
    for f in list_images(folder):
        base, crossed = split_name(f)
        g = groups.setdefault(base, {"parallel": [], "crossed": []})
        g["crossed" if crossed else "parallel"].append(f)

    pairs, used = [], set()
    for base in sorted(groups, key=natural_key):
        g = groups[base]
        name = base  # порядковый номер из имени файла сохраняется: «16. Г-170»
        unique, n = name, 2
        while unique.casefold() in used:
            unique = f"{name} ({n})"
            n += 1
        used.add(unique.casefold())
        problems = []
        for side in SIDES:
            if not g[side]:
                problems.append(f"нет изображения: {SIDE_TITLES[side].lower()}")
            elif len(g[side]) > 1:
                problems.append(f"несколько файлов ({SIDE_TITLES[side].lower()}): " + ", ".join(g[side]))
        pairs.append(new_pair(unique, (g["parallel"] or [None])[0], (g["crossed"] or [None])[0],
                              "; ".join(problems)))
    return pairs


def swap_sides(pair):
    """Поменять местами изображения. Точки остаются привязаны к своим изображениям; совмещение нужно
    пересчитать (рамка обрезки сохраняется — общая система координат симметрична)."""
    pair["parallel"], pair["crossed"] = pair["crossed"], pair["parallel"]
    for pt in pair["points"]:
        pt["parallel"], pt["crossed"] = pt["crossed"], pt["parallel"]
    pair["align"] = None


def reassign(pair, values):
    """Назначение файлов вручную; точки сменённого изображения удаляются, совмещение сбрасывается."""
    changed = [s for s in SIDES if pair[s] != values[s]]
    if not changed:
        return False
    for s in changed:
        pair[s] = values[s]
        for pt in pair["points"]:
            pt[s] = None
    pair["points"] = [pt for pt in pair["points"] if pt["parallel"] or pt["crossed"]]
    pair["align"] = None
    pair["problem"] = "" if all(pair[s] for s in SIDES) else "не назначены оба изображения"
    return True


def is_aligned(pair):
    return bool(pair and pair.get("align"))


def points_changed(pair):
    """Точки изменены после последнего усреднения."""
    a = pair.get("align") if pair else None
    if not a:
        return False
    cur = [(n, p, q) for n, p, q in geometry.complete_points(pair["points"])]
    if [n for n, _, _ in cur] != a["numbers"]:
        return True
    for (_, p, q), s in zip(cur, a["points"]):
        if max(abs(p[0] - s[0]), abs(p[1] - s[1]), abs(q[0] - s[2]), abs(q[1] - s[3])) > 1e-6:
            return True
    return False


# ---------------------------------------------------------------- обрезка

def clamp_rect(rect, canvas, min_size=2):
    """Прямоугольник [x, y, w, h] (целые), ограниченный холстом; None, если пересечения нет."""
    if not rect:
        return None
    cx, cy, cw, ch = canvas
    x0, y0 = max(rect[0], cx), max(rect[1], cy)
    x1, y1 = min(rect[0] + rect[2], cx + cw), min(rect[1] + rect[3], cy + ch)
    x0, y0, x1, y1 = round(x0), round(y0), round(x1), round(y1)
    if x1 - x0 < min_size or y1 - y0 < min_size:
        return None
    return [x0, y0, x1 - x0, y1 - y0]


def output_rect(pair):
    """Область экспорта в общей системе: применённая обрезка или весь холст."""
    canvas = pair["align"]["canvas"]
    return clamp_rect(pair.get("crop"), canvas) or list(canvas)


# ---------------------------------------------------------------- проект

class ProjectError(Exception):
    pass


def _pt(v):
    return [float(v[0]), float(v[1])] if v else None


def _rect(v):
    return [int(round(float(t))) for t in v[:4]] if v and len(v) >= 4 else None


class Project:
    def __init__(self, path, image_dir, pairs, settings=None):
        self.path = path
        self.image_dir = image_dir
        self.pairs = pairs
        self.settings = dict(DEFAULT_SETTINGS)
        if settings:
            self.settings.update({k: v for k, v in settings.items() if k in DEFAULT_SETTINGS})

    def image_path(self, pair, side):
        return os.path.join(self.image_dir, pair[side]) if pair and pair[side] else None

    def missing_files(self):
        missing = []
        for p in self.pairs:
            for s in SIDES:
                path = self.image_path(p, s)
                if path and not os.path.isfile(path):
                    missing.append(p[s])
        return missing

    def to_dict(self):
        return {"format": "panorama_pre_processing", "version": 1, "image_dir": self.image_dir,
                "settings": self.settings, "pairs": self.pairs}

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)

    def check_sizes(self):
        """Сбросить совмещение у пар, исходные файлы которых изменили размер. Возвращает их имена."""
        reset = []
        for p in self.pairs:
            a = p.get("align")
            if not a:
                continue
            for s in SIDES:
                size = image_size(self.image_path(p, s))
                if size and list(size) != list(a["sizes"][s]):
                    p["align"] = None
                    reset.append(p["name"])
                    break
        return reset

    @classmethod
    def load(cls, path):
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("format") != "panorama_pre_processing":
                raise ValueError("это не проект программы предварительной обработки панорам")
            pairs = []
            for p in data["pairs"]:
                pair = new_pair(str(p["name"]), p.get("parallel"), p.get("crossed"), p.get("problem", ""))
                pair["points"] = [{s: _pt(pt.get(s)) for s in SIDES} for pt in p.get("points", [])]
                a = p.get("align")
                if a and a.get("version") == 1 and len(a.get("canvas") or []) == 4 and a.get("points"):
                    pair["align"] = a
                pair["crop"] = _rect(p.get("crop"))
                pair["crop_draft"] = _rect(p.get("crop_draft"))
                pair["done"] = bool(p.get("done", False))
                pairs.append(pair)
            return cls(path, data["image_dir"], pairs, data.get("settings"))
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as e:
            raise ProjectError(f"Не удалось прочитать проект «{path}»: {e}") from e


_size_cache = {}


def image_size(path):
    """Размер изображения по заголовку файла (без полной загрузки) или None."""
    if not path:
        return None
    if path not in _size_cache:
        try:
            with Image.open(path) as im:
                _size_cache[path] = im.size
        except (OSError, ValueError):
            return None
    return _size_cache[path]


def clear_size_cache():
    _size_cache.clear()


# ---------------------------------------------------------------- изображения

def to_rgb(img):
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        img = img.point(lambda v: v / 256).convert("L")
    if img.mode == "P":
        img = img.convert("RGBA")
    if img.mode in ("RGBA", "LA", "La", "RGBa"):
        bg = Image.new("RGB", img.size, "white")
        bg.paste(img.convert("RGBA"), mask=img.convert("RGBA").getchannel("A"))
        return bg
    return img.convert("RGB") if img.mode != "RGB" else img


def load_preview(path, max_side):
    """Уменьшенная копия (numpy H×W×3) и размер исходного файла."""
    with Image.open(path) as im:
        size = im.size
        im.draft("RGB", (max_side, max_side))
        img = to_rgb(im)
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        arr = np.asarray(img).copy()
    return arr, size


def load_full(path):
    """Исходник в полном разрешении (Pillow, RGB). Фрагменты для пересэмплирования берутся из него по блокам,
    без полной копии в numpy."""
    im = Image.open(path)
    im.load()
    img = to_rgb(im)
    if img is not im:
        im.close()
    return img


# ---------------------------------------------------------------- экспорт

def safe_filename(name):
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip().rstrip(".")
    return name or "шлиф"


def export_names(pair, fmt="source"):
    """Имена файлов результата — исходные базовые имена. fmt="source": расширение исходного файла
    (PNG остаётся PNG, JPEG — JPEG), "jpeg": «.jpg». Если файлы назначены вручную и их имена не образуют пару
    по правилам panorama_marking — «<название>» и «<название> с»."""
    def ext(side):
        e = os.path.splitext(pair[side])[1]
        return e if fmt == "source" and e.lower() in IMAGE_EXT else ".jpg"

    names = {s: split_name(pair[s]) for s in SIDES}
    if not names["parallel"][1] and names["crossed"][1] and names["parallel"][0] == names["crossed"][0]:
        stems = {s: safe_filename(unicodedata.normalize("NFC", os.path.splitext(pair[s])[0]).strip()) for s in SIDES}
        if split_name(stems["crossed"])[1] and not split_name(stems["parallel"])[1]:
            return {s: stems[s] + ext(s) for s in SIDES}
    base = safe_filename(pair["name"])
    return {"parallel": base + ext("parallel"), "crossed": base + " с" + ext("crossed")}


class ExportError(Exception):
    """Ошибка экспорта с указанием шага (чтение / расчёт / запись)."""


def save_image(img, path, quality, icc=None):
    """Запись по расширению path: PNG — без потерь, JPEG — с заданным качеством. Сначала во временный файл."""
    tmp = path + ".part"
    extra = {"icc_profile": icc} if icc else {}
    if os.path.splitext(path)[1].lower() == ".png":
        img.save(tmp, "PNG", compress_level=4, **extra)
    else:
        if max(img.size) > JPEG_MAX_SIDE:
            raise ValueError(f"размер {img.width}×{img.height} превышает предел формата JPEG ({JPEG_MAX_SIDE} px) — "
                             "сохраните в PNG или уменьшите рамку")
        q = int(quality)
        img.save(tmp, "JPEG", quality=q, subsampling=0 if q >= 90 else 2,
                 optimize=img.width * img.height <= 60_000_000, **extra)
    try:
        os.replace(tmp, path)
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def icc_profile(path):
    try:
        with Image.open(path) as im:
            return im.info.get("icc_profile")
    except (OSError, ValueError):
        return None


def export_targets(pair, out_dir, fmt="source"):
    """Полные пути файлов результата пары: {сторона: путь}."""
    return {s: os.path.join(out_dir, n) for s, n in export_names(pair, fmt).items()}


def export_pair(pair, image_dir, out_dir, quality, fmt="source", progress=None, cancelled=None):
    """Экспорт пары: два файла сразу в out_dir (исходные базовые имена). Преобразование и обрезка — одно
    пересэмплирование из оригиналов, блоками. Возвращает список записанных файлов или None при отмене."""
    if not pair.get("align"):
        raise ValueError("пара не совмещена — нажмите «Усреднить геометрию»")
    for side in SIDES:
        if not pair[side]:
            raise ValueError(f"не назначено изображение: {SIDE_TITLES[side].lower()}")
        if not os.path.isfile(os.path.join(image_dir, pair[side])):
            raise ValueError(f"файл не найден: {pair[side]}")
    params = pair["align"]
    for side in SIDES:
        size = image_size(os.path.join(image_dir, pair[side]))
        if size and list(size) != list(params["sizes"][side]):
            raise ValueError(f"размер файла {pair[side]} изменился после совмещения — повторите усреднение")
    field = geometry.get_field(params)
    rect = output_rect(pair)
    os.makedirs(out_dir, exist_ok=True)
    targets = export_targets(pair, out_dir, fmt)
    written = []
    for n, side in enumerate(SIDES):
        if cancelled and cancelled():
            return None
        path = os.path.join(image_dir, pair[side])
        stage = f"чтение {pair[side]}"
        try:
            icc = icc_profile(path)
            src = load_full(path)
            size = src.size
            stage = f"расчёт {os.path.basename(targets[side])}"
            out = Image.new("RGB", (rect[2], rect[3]))

            def sink(tile, x, y, out=out):
                out.paste(Image.fromarray(tile), (x, y))

            def prog(i, total, n=n):
                if progress:
                    progress((n + i / total * 0.9) / 2)

            done = field.render(side, src, size, rect, sink=sink, block=2048, progress=prog, cancelled=cancelled)
            del src
            if done is None and cancelled and cancelled():
                return None
            stage = f"запись {os.path.basename(targets[side])}"
            save_image(out, targets[side], quality, icc)
            del out
        except ExportError:
            raise
        except MemoryError:
            raise ExportError(f"недостаточно памяти ({stage}, результат {rect[2]}×{rect[3]} px) — закройте другие "
                              "программы или уменьшите рамку") from None
        except PermissionError as e:
            raise ExportError(f"нет доступа к файлу ({stage}) — возможно, он открыт в другой программе "
                              f"или папка защищена: {e}") from e
        except Exception as e:  # noqa: BLE001 — добавить к любой ошибке, на каком шаге она случилась
            raise ExportError(f"{stage}: {type(e).__name__}: {e}") from e
        written.append(targets[side])
        if progress:
            progress((n + 1) / 2)
    return written


def clone(obj):
    return copy.deepcopy(obj)


def rect_text(rect):
    return f"{rect[2]} × {rect[3]} px" if rect else "—"


def fmt_px(v):
    return f"{v:.1f}" if v < 100 else f"{v:.0f}"

