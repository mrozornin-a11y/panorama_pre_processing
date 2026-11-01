"""Усреднение геометрии пары панорам: общая промежуточная система координат и плавная деформация.

Обозначения
-----------
P_i — точка i на параллельных николях (пиксели исходного файла), Q_i — та же деталь на скрещённых.
Координаты непрерывные: пиксель (u, v) занимает квадрат [u, u+1) × [v, v+1), его центр — (u+0.5, v+0.5).
Точки на плоскости записываются комплексными числами z = x + iy.

1. Подобие S (сдвиг + поворот + масштаб) «скрещённые → параллельные» по методу наименьших
   квадратов: P ≈ a·Q + b.
2. «Половина» подобия H(z) = α·z + β, α = √a, β = b / (1 + α), так что H(H(z)) = S(z).
   Общая система C: для скрещённых C = H(Q), для параллельных C = H⁻¹(P).
   Масштаб C — среднее геометрическое разрешений исходников, поворот — пополам.
3. В системе C точки обоих изображений сравнимы: p'_i = H⁻¹(P_i), c'_i = H(Q_i),
   промежуточные положения m_i = (p'_i + c'_i) / 2.
4. Поле смещений u(x) (тонкая пластина, TPS, с аффинной частью) с u(m_i) = (p'_i − c'_i) / 2.
   Обратные отображения результата в исходники:
       параллельные:  P = H(x + u(x)),   скрещённые:  Q = H⁻¹(x − u(x)).
   Поправка делится поровну и с противоположным знаком: обе панорамы деформируются навстречу.
5. Поле u вычисляется точно в узлах регулярной сетки (шаг G пикселей результата) и
   билинейно интерполируется между ними — одинаково для предпросмотра и экспорта.
"""
import hashlib
import json
import math
import threading
from collections import OrderedDict

import numpy as np

SIDES = ("parallel", "crossed")

GRID_NODES = 1024          # узлов поля по большей стороне холста
MIN_DET = 0.2              # меньше — складка/переворот или чрезмерное сжатие
MAX_DET = 5.0              # больше — чрезмерное растяжение
COLLINEAR_RATIO = 0.04     # отношение сингулярных чисел облака точек: «почти на одной прямой»
MIN_SCALE, MAX_SCALE = 0.25, 4.0


class AlignError(Exception):
    """Проблема с точками — текст для пользователя."""


SIDE_GEN = {"parallel": "параллельных николях", "crossed": "скрещённых николях"}


# ---------------------------------------------------------------- точки

def complete_points(points):
    """[(номер, P, Q)] для пар точек, поставленных на обоих изображениях."""
    return [(n + 1, pt["parallel"], pt["crossed"]) for n, pt in enumerate(points)
            if pt.get("parallel") and pt.get("crossed")]


def fit_similarity(P, Q):
    """Подобие Q → P (комплексные массивы): P ≈ a·Q + b, без отражения."""
    qm, pm = Q.mean(), P.mean()
    qc, pc = Q - qm, P - pm
    den = float(np.sum(np.abs(qc) ** 2))
    if den <= 0:
        raise AlignError("Точки на скрещённых николях совпадают — расставьте их в разных местах.")
    a = complex(np.sum(np.conj(qc) * pc) / den)
    return a, pm - a * qm


def half_similarity(a, b):
    alpha = complex(np.sqrt(a))  # главный корень: поворот пополам, масштаб — квадратный корень
    return alpha, b / (1 + alpha)


def _affine_det(src, dst):
    """Определитель линейной части аффинного преобразования src → dst (МНК)."""
    A = np.column_stack([src.real, src.imag, np.ones(len(src))])
    coef, *_ = np.linalg.lstsq(A, np.column_stack([dst.real, dst.imag]), rcond=None)
    return coef[0, 0] * coef[1, 1] - coef[1, 0] * coef[0, 1]


def _spread_ratio(Z):
    c = np.column_stack([Z.real - Z.real.mean(), Z.imag - Z.imag.mean()])
    sv = np.linalg.svd(c, compute_uv=False)
    return sv[-1] / sv[0] if sv[0] > 0 else 0.0


def check_points(points, sizes):
    """Проверка расстановки точек. Возвращает (список ошибок, список предупреждений)."""
    errors, warnings = [], []
    for n, pt in enumerate(points):
        have = [s for s in SIDES if pt.get(s)]
        if len(have) == 1:
            missing = SIDES[1] if have[0] == SIDES[0] else SIDES[0]
            errors.append(f"Точка {n + 1} поставлена только на {SIDE_GEN[have[0]]} — "
                          f"поставьте её и на {SIDE_GEN[missing]} или удалите.")
    full = complete_points(points)
    if len(full) < 2:
        errors.append("Нужно не менее двух пар точек (для исправления локальных расхождений — "
                      "четыре и больше, равномерно по всей панораме).")
        return errors, warnings
    for s, idx in (("parallel", 1), ("crossed", 2)):
        w, h = sizes[s]
        tol = max(3.0, 0.001 * math.hypot(w, h))
        for i in range(len(full)):
            for j in range(i + 1, len(full)):
                if math.dist(full[i][idx], full[j][idx]) < tol:
                    errors.append(f"Точки {full[i][0]} и {full[j][0]} на {SIDE_GEN[s]} почти совпадают — "
                                  "раздвиньте их или удалите одну.")
    if errors:
        return errors, warnings
    P = np.array([complex(*p) for _, p, _ in full])
    Q = np.array([complex(*q) for _, _, q in full])
    a, b = fit_similarity(P, Q)
    s = abs(a)
    if not MIN_SCALE <= s <= MAX_SCALE:
        errors.append(f"Масштаб изображений по точкам различается в {max(s, 1 / s):.1f} раза — "
                      "вероятно, точки поставлены на разных деталях. Проверьте их.")
        return errors, warnings
    if len(full) >= 3 and _spread_ratio(P) >= COLLINEAR_RATIO and _affine_det(Q, P) <= 0:
        errors.append("Точки расставлены в зеркальном порядке (номера на двух изображениях перепутаны). "
                      "Проверьте нумерацию точек.")
    return errors, warnings


# ---------------------------------------------------------------- тонкая пластина

class TPS:
    """Интерполяция векторного поля d в точках m (комплексные): аффинная часть + r² log r."""

    def __init__(self, m, d, affine_only=False):
        self.center = m.mean()
        self.scale = float(np.max(np.abs(m - self.center))) or 1.0
        x = (m - self.center) / self.scale
        n = len(m)
        self.x = x
        if affine_only:
            A = np.column_stack([np.ones(n), x.real, x.imag])
            coef, *_ = np.linalg.lstsq(A, np.column_stack([d.real, d.imag]), rcond=None)
            self.w = np.zeros((n, 2))
            self.a = coef
            return
        K = _kernel(np.abs(x[:, None] - x[None, :]))
        Pm = np.column_stack([np.ones(n), x.real, x.imag])
        L = np.zeros((n + 3, n + 3))
        L[:n, :n] = K
        L[:n, n:] = Pm
        L[n:, :n] = Pm.T
        rhs = np.zeros((n + 3, 2))
        rhs[:n, 0], rhs[:n, 1] = d.real, d.imag
        try:
            sol = np.linalg.solve(L, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(L, rhs, rcond=None)[0]
        self.w, self.a = sol[:n], sol[n:]

    def __call__(self, z):
        """Значения поля (комплексные) в точках z (любая форма массива)."""
        z = np.asarray(z, dtype=complex)
        shape = z.shape
        z = ((z.ravel() - self.center) / self.scale)
        out = np.empty(z.size, dtype=complex)
        chunk = max(1, 4_000_000 // max(1, len(self.x)))
        for i in range(0, z.size, chunk):
            zz = z[i:i + chunk]
            v = (self.a[0, 0] + self.a[1, 0] * zz.real + self.a[2, 0] * zz.imag) + \
                1j * (self.a[0, 1] + self.a[1, 1] * zz.real + self.a[2, 1] * zz.imag)
            if np.any(self.w):
                U = _kernel(np.abs(zz[:, None] - self.x[None, :]))
                v = v + U @ self.w[:, 0] + 1j * (U @ self.w[:, 1])
            out[i:i + chunk] = v
        return out.reshape(shape)


def _kernel(r):
    with np.errstate(divide="ignore", invalid="ignore"):
        k = r * r * np.log(r)
    return np.nan_to_num(k, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------- расчёт совмещения

def _boundary(w, h, n=200):
    t = np.linspace(0, 1, n)
    return np.concatenate([t * w, w + 1j * t * h, (1 - t) * w + 1j * h, 1j * (1 - t) * h])


def _forward(y, u, sign, iters=30):
    """Решение x + sign·u(x) = y методом простой итерации (прямое отображение границы)."""
    x = y.copy()
    for _ in range(iters):
        x = y - sign * u(x)
    return x


def compute_alignment(points, sizes):
    """Расчёт совмещения по точкам. Возвращает (параметры для проекта, Field).
    Ошибки — AlignError с понятным текстом."""
    errors, warnings = check_points(points, sizes)
    if errors:
        raise AlignError("\n".join(errors))
    full = complete_points(points)
    numbers = [n for n, _, _ in full]
    P = np.array([complex(*p) for _, p, _ in full])
    Q = np.array([complex(*q) for _, _, q in full])
    a, b = fit_similarity(P, Q)
    alpha, beta = half_similarity(a, b)
    p1 = (P - beta) / alpha          # параллельные → C
    c1 = alpha * Q + beta            # скрещённые → C
    m = (p1 + c1) / 2
    d = (p1 - c1) / 2
    resid = np.abs(p1 - c1)          # расхождение точек после общего подобия, пиксели C

    if len(full) == 2:
        model = "similarity"
    elif _spread_ratio(m) < COLLINEAR_RATIO:
        model = "similarity"
        warnings.append("Точки лежат почти на одной прямой — исправлены только общие смещение, поворот "
                        "и масштаб. Для исправления локальных расхождений добавьте точки в стороне от этой линии.")
    elif len(full) == 3:
        model = "affine"
    else:
        model = "tps"

    params = {
        "version": 1,
        "points": [[P[i].real, P[i].imag, Q[i].real, Q[i].imag] for i in range(len(full))],
        "numbers": numbers,
        "sizes": {s: list(map(int, sizes[s])) for s in SIDES},
        "alpha": [alpha.real, alpha.imag],
        "beta": [beta.real, beta.imag],
        "scale": abs(a),
        "rotation_deg": math.degrees(math.atan2(a.imag, a.real)),
        "model": model,
        "resid_rms": float(np.sqrt(np.mean(resid ** 2))),
        "resid_max": float(resid.max()),
        "resid_point": int(numbers[int(resid.argmax())]),
        "warnings": warnings,
    }
    u = _make_u(model, m, d)

    # холст — охватывающий прямоугольник обоих деформированных изображений
    yp = (_boundary(*sizes["parallel"]) - beta) / alpha
    yc = alpha * _boundary(*sizes["crossed"]) + beta
    xp = _forward(yp, u, +1)
    xc = _forward(yc, u, -1)
    allx = np.concatenate([xp, xc])
    if not np.all(np.isfinite(allx)):
        raise AlignError(_fold_message(None, m, numbers))
    x0, y0 = math.floor(allx.real.min()), math.floor(allx.imag.min())
    x1, y1 = math.ceil(allx.real.max()), math.ceil(allx.imag.max())
    cw, ch = x1 - x0, y1 - y0
    if cw <= 0 or ch <= 0 or max(cw, ch) > 8 * max(max(sizes[s]) for s in SIDES):
        raise AlignError(_fold_message(None, m, numbers))
    params["canvas"] = [x0, y0, cw, ch]
    params["grid_step"] = max(2, math.ceil(max(cw, ch) / GRID_NODES))
    field = Field(params, u)
    field.check_folds(m, numbers)
    return params, field


def _make_u(model, m, d):
    if model == "similarity":
        return lambda z: np.zeros(np.shape(z), dtype=complex)
    return TPS(m, d, affine_only=(model == "affine"))


def _fold_message(where, m, numbers):
    text = "Преобразование получается со складкой или переворотом"
    if where is not None:
        k = int(np.argmin(np.abs(m - where)))
        text += f" около точки {numbers[k]}"
    return (text + ". Скорее всего, одна из точек поставлена не на ту деталь или соседние точки "
            "противоречат друг другу. Проверьте положение точек (особенно указанной), передвиньте или удалите "
            "неверную и снова нажмите «Усреднить геометрию».")


# ---------------------------------------------------------------- поле смещений

class Field:
    """Поле u на регулярной сетке узлов холста + отображения результата в исходники."""

    def __init__(self, params, u=None):
        self.params = params
        self.alpha = complex(*params["alpha"])
        self.beta = complex(*params["beta"])
        self.sizes = {s: tuple(params["sizes"][s]) for s in SIDES}
        x0, y0, cw, ch = params["canvas"]
        G = params["grid_step"]
        self.G = G
        self.gx0, self.gy0 = x0 - G, y0 - G
        nx, ny = math.ceil(cw / G) + 3, math.ceil(ch / G) + 3
        if u is None:
            P = np.array([complex(p[0], p[1]) for p in params["points"]])
            Q = np.array([complex(p[2], p[3]) for p in params["points"]])
            p1 = (P - self.beta) / self.alpha
            c1 = self.alpha * Q + self.beta
            u = _make_u(params["model"], (p1 + c1) / 2, (p1 - c1) / 2)
        gx = self.gx0 + G * np.arange(nx)
        gy = self.gy0 + G * np.arange(ny)
        if params["model"] == "similarity":
            self.ux = np.zeros((ny, nx))
            self.uy = np.zeros((ny, nx))
        else:
            v = u(gx[None, :] + 1j * gy[:, None])
            self.ux, self.uy = v.real.copy(), v.imag.copy()

    # --- плотные отображения
    def _interp(self, xs, ys):
        """Билинейная интерполяция поля в точках решётки xs × ys (координаты C)."""
        fx = np.clip((xs - self.gx0) / self.G, 0, self.ux.shape[1] - 1.000001)
        fy = np.clip((ys - self.gy0) / self.G, 0, self.ux.shape[0] - 1.000001)
        ix, iy = fx.astype(np.int64), fy.astype(np.int64)
        wx, wy = (fx - ix)[None, :], (fy - iy)[:, None]
        r0, r1 = int(iy.min()), int(iy.max()) + 2
        ry = iy - r0
        out = []
        for g in (self.ux, self.uy):
            rows = g[r0:r1]                                   # только нужные строки узлов
            gx = rows[:, ix] * (1 - wx) + rows[:, ix + 1] * wx  # интерполяция по x (малый массив)
            out.append(gx[ry] * (1 - wy) + gx[ry + 1] * wy)   # затем по y
        return out

    def source_coords(self, side, xs, ys):
        """Непрерывные координаты исходника side для точек решётки xs × ys системы C."""
        ux, uy = self._interp(xs, ys)
        X = xs[None, :]
        Y = ys[:, None]
        if side == "parallel":   # P = α·(x + u) + β
            a, b, c = self.alpha, 1.0, self.beta
        else:                     # Q = (x − u − β) / α
            a, b, c = 1 / self.alpha, -1.0, -self.beta / self.alpha
        X = X + b * ux
        Y = Y + b * uy
        sx = a.real * X - a.imag * Y
        sx += c.real
        sy = a.imag * X
        sy += a.real * Y
        sy += c.imag
        return sx, sy

    def to_common(self, side, x, y):
        """Точка исходника → система C (с учётом поля; обратное отображение итерациями)."""
        if side == "parallel":
            yv = (complex(x, y) - self.beta) / self.alpha
            sign = 1
        else:
            yv = self.alpha * complex(x, y) + self.beta
            sign = -1
        z = yv
        for _ in range(40):
            ux, uy = self._interp(np.array([z.real]), np.array([z.imag]))
            z = yv - sign * complex(ux[0, 0], uy[0, 0])
        return z.real, z.imag

    def check_folds(self, m, numbers):
        """Якобиан обратных отображений в узлах, где видно изображение: складки и сильные искажения."""
        G = self.G
        dux_dy, dux_dx = np.gradient(self.ux, G)
        duy_dy, duy_dx = np.gradient(self.uy, G)
        ny, nx = self.ux.shape
        xs = self.gx0 + G * np.arange(nx)
        ys = self.gy0 + G * np.arange(ny)
        for side, sign in (("parallel", 1), ("crossed", -1)):
            det = (1 + sign * dux_dx) * (1 + sign * duy_dy) - dux_dy * duy_dx
            sx, sy = self.source_coords(side, xs.astype(float), ys.astype(float))
            w, h = self.sizes[side]
            valid = (sx >= 0) & (sx <= w) & (sy >= 0) & (sy <= h)
            bad = valid & ((det < MIN_DET) | (det > MAX_DET))
            if np.any(bad):
                iy, ix = np.unravel_index(np.argmax(np.where(bad, np.abs(np.log(np.abs(det) + 1e-12)), -1)),
                                          det.shape)
                raise AlignError(_fold_message(complex(xs[ix], ys[iy]), m, numbers))

    # --- отрисовка
    def render(self, side, src, src_size, rect, out_scale=1.0, fill=None, interp=None, block=1024,
               progress=None, cancelled=None, sink=None):
        """Отрисовка результата side в области rect = (x, y, w, h) системы C с масштабом out_scale
        (1 — пиксели C, <1 — уменьшенный предпросмотр).

        src — исходник (numpy H×W×3 или изображение Pillow RGB) полного или уменьшенного размера;
        src_size — размер полного исходника
        (координаты пересчитываются под фактический размер src). fill(mask, x, y) — заполнение пустых
        областей (по умолчанию белый). Возвращает numpy H×W×3; если задан sink(tile, x, y), блоки передаются
        ему по мере готовности (весь результат в памяти не собирается).
        """
        import cv2

        x, y, w, h = rect
        W = max(1, round(w * out_scale))
        H = max(1, round(h * out_scale))
        kxo, kyo = W / w, H / h            # фактический масштаб по осям (размер округлён до пикселя)
        out = None if sink else np.empty((H, W, 3), np.uint8)
        sw, sh = (src.shape[1], src.shape[0]) if isinstance(src, np.ndarray) else src.size
        kx = sw / src_size[0]
        ky = sh / src_size[1]
        local = math.sqrt(abs(self.alpha) ** (2 if side == "parallel" else -2)) * min(kx, ky) / out_scale
        if interp is None:
            interp = cv2.INTER_CUBIC
        tiles = [(r, c) for r in range(0, H, block) for c in range(0, W, block)]
        for n, (r0, c0) in enumerate(tiles):
            if cancelled and cancelled():
                return None
            r1, c1 = min(H, r0 + block), min(W, c0 + block)
            xs = x + (np.arange(c0, c1) + 0.5) / kxo
            ys = y + (np.arange(r0, r1) + 0.5) / kyo
            sx, sy = self.source_coords(side, xs, ys)
            valid = (sx >= 0) & (sx <= src_size[0]) & (sy >= 0) & (sy <= src_size[1])
            mx = sx * kx - 0.5
            my = sy * ky - 0.5
            del sx, sy
            tile = np.empty((r1 - r0, c1 - c0, 3), np.uint8)
            all_valid = bool(valid.all())
            if all_valid or np.any(valid):
                vx, vy = (mx, my) if all_valid else (mx[valid], my[valid])
                pad = 4 + math.ceil(2 * local)
                bx0 = max(0, int(math.floor(vx.min())) - pad)
                by0 = max(0, int(math.floor(vy.min())) - pad)
                bx1 = min(sw, int(math.ceil(vx.max())) + pad + 1)
                by1 = min(sh, int(math.ceil(vy.max())) + pad + 1)
                del vx, vy
                if isinstance(src, np.ndarray):
                    crop = src[by0:by1, bx0:bx1]
                else:
                    crop = np.asarray(src.crop((bx0, by0, bx1, by1)))
                if local > 1.15:  # уменьшение: лёгкое сглаживание против муара (одно пересэмплирование)
                    sigma = 0.5 * math.sqrt(local * local - 1)
                    crop = cv2.GaussianBlur(np.ascontiguousarray(crop), (0, 0), sigma)
                tile[:] = cv2.remap(np.ascontiguousarray(crop), (mx - bx0).astype(np.float32),
                                    (my - by0).astype(np.float32), interp, borderMode=cv2.BORDER_REPLICATE)
            if not all_valid:
                inv = ~valid
                if fill is None:
                    tile[inv] = 255
                else:
                    fill(tile, inv, c0, r0)
            if sink:
                sink(tile, c0, r0)
            else:
                out[r0:r1, c0:c1] = tile
            if progress:
                progress(n + 1, len(tiles))
        return out


# ---------------------------------------------------------------- кэш полей

_field_cache = OrderedDict()
_field_lock = threading.Lock()


def params_key(params):
    return hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()


def get_field(params):
    """Поле по параметрам из проекта (детерминированно восстанавливается из точек) с небольшим кэшем."""
    key = params_key(params)
    with _field_lock:
        f = _field_cache.get(key)
        if f is not None:
            _field_cache.move_to_end(key)
            return f
    f = Field(params)
    with _field_lock:
        _field_cache[key] = f
        while len(_field_cache) > 6:
            _field_cache.popitem(last=False)
    return f


def put_field(params, field):
    with _field_lock:
        _field_cache[params_key(params)] = field


def common_points(params):
    """Промежуточные положения точек m_i в системе C: [(номер, x, y)]."""
    alpha, beta = complex(*params["alpha"]), complex(*params["beta"])
    out = []
    for n, p in zip(params["numbers"], params["points"]):
        z = ((complex(p[0], p[1]) - beta) / alpha + alpha * complex(p[2], p[3]) + beta) / 2
        out.append((n, z.real, z.imag))
    return out


def display_frame(points, sizes, params=None):
    """(α, β) общей системы C для показа исходников в одном окне без локальной деформации:
    параллельные C = (P − β)/α, скрещённые C = α·Q + β. После совмещения — сохранённые параметры
    (тот же кадр, что у результата); иначе — по уже поставленным точкам; без точек — по размерам кадров."""
    if params:
        return complex(*params["alpha"]), complex(*params["beta"])
    full = complete_points(points)
    if len(full) >= 2:
        P = np.array([complex(*p) for _, p, _ in full])
        Q = np.array([complex(*q) for _, _, q in full])
        try:
            a, b = fit_similarity(P, Q)
            if MIN_SCALE <= abs(a) <= MAX_SCALE:
                return half_similarity(a, b)
        except AlignError:
            pass
    (wp, hp), (wc, hc) = sizes["parallel"], sizes["crossed"]
    a = complex(math.sqrt(wp * hp / (wc * hc)), 0)
    return half_similarity(a, complex(wp / 2, hp / 2) - a * complex(wc / 2, hc / 2))


def predict_point(points, side, xy):
    """Ожидаемое положение на втором изображении для точки xy на side (по уже поставленным парам):
    аффинно при ≥3 невырожденных точках, по подобию при 2, иначе None."""
    full = complete_points(points)
    if len(full) < 2:
        return None
    P = np.array([complex(*p) for _, p, _ in full])
    Q = np.array([complex(*q) for _, _, q in full])
    src, dst = (P, Q) if side == "parallel" else (Q, P)
    z = complex(*xy)
    if len(full) >= 3 and _spread_ratio(src) >= COLLINEAR_RATIO:
        A = np.column_stack([src.real, src.imag, np.ones(len(src))])
        coef, *_ = np.linalg.lstsq(A, np.column_stack([dst.real, dst.imag]), rcond=None)
        v = np.array([z.real, z.imag, 1.0]) @ coef
        return [float(v[0]), float(v[1])]
    a, b = fit_similarity(dst, src)
    w = a * z + b
    return [w.real, w.imag]
