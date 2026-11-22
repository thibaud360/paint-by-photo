#!/usr/bin/env python3
"""paint-by-photo: turn a photo into a printable paint-by-numbers kit and a colouring page.

    python paint_by_photo.py photo.jpg
    python paint_by_photo.py photo.jpg --colors 24 --detail high --size A3

Outputs (in -o, default paint_out/):
    NAME_numbers.png   outlines with a number in every region
    NAME_palette.png   colour key (numbers, hex codes, optional mixing hints)
    NAME_preview.png   what the finished painting will look like
    NAME_coloring.png  outlines only, no numbers
    NAME_kit.pdf       printable: numbered canvas, palette, colouring page

Needs: pillow numpy opencv-python scipy scikit-learn
"""
import argparse
import itertools
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy import ndimage
from sklearn.cluster import KMeans

PX_PER_MM = 10.0     # output resolution (about 254 dpi)
WORK_SCALE = 2.5     # output px per working px
MARGIN_MM = 10.0

PAGES = {"A4": (210.0, 297.0), "A3": (297.0, 420.0), "LETTER": (215.9, 279.4)}

# detail: (min inradius mm, min area mm2, smoothing passes, label blur sigma in work px)
DETAIL = {
    "low": (3.6, 70.0, 5, 2.4),
    "medium": (2.7, 32.0, 4, 1.7),
    "high": (2.0, 14.0, 3, 1.2),
}

LINE_GREY = 55
TEXT_GREY = 40

# Simple paint box for the optional mixing hint (sRGB, roughly artist acrylics).
PAINTS = [
    ("white", (246, 246, 240)),
    ("black", (28, 27, 30)),
    ("yellow", (250, 214, 20)),
    ("red", (205, 30, 40)),
    ("blue", (28, 45, 150)),
    ("green", (20, 120, 80)),
    ("brown", (100, 55, 30)),
]


# ----------------------------------------------------------------- colour utils
def rgb2lab(rgb):
    a = np.asarray(rgb, np.uint8).reshape(-1, 1, 3)
    return cv2.cvtColor(a, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32) * np.array(
        [100 / 255, 1, 1], np.float32) - np.array([0, 128, 128], np.float32)


def lab2rgb(lab):
    l = np.asarray(lab, np.float32).reshape(-1, 1, 3)
    l = (l + np.array([0, 128, 128], np.float32)) * np.array([255 / 100, 1, 1], np.float32)
    return cv2.cvtColor(np.clip(l, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).reshape(-1, 3)


def hexof(c):
    return "#%02X%02X%02X" % tuple(int(v) for v in c)


def luma(c):
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


# ----------------------------------------------------------------- pipeline
def load_font(size):
    for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Regular.ttf", "Arial.ttf",
                 "Helvetica.ttc", "/System/Library/Fonts/Helvetica.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                 "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default(size)


def canvas_geometry(img_w, img_h, size):
    pw, ph = PAGES[size]
    if img_w < img_h:
        orient = "portrait"
    else:
        orient = "landscape"
        pw, ph = ph, pw
    aw, ah = pw - 2 * MARGIN_MM, ph - 2 * MARGIN_MM
    s = min(aw / img_w, ah / img_h)
    return (pw, ph), (img_w * s, img_h * s), orient


def smooth(rgb, passes):
    out = rgb
    for _ in range(passes):
        out = cv2.bilateralFilter(out, 9, 28, 7)
    out = cv2.pyrMeanShiftFiltering(out, 6, 18)
    return out


def kmeans_lab(rgb, k, seed):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100 / 255
    lab[..., 1:] -= 128
    flat = lab.reshape(-1, 3)
    rng = np.random.default_rng(seed)
    sample = flat[rng.choice(len(flat), min(len(flat), 60000), replace=False)]
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(sample)
    labels = km.predict(flat).reshape(rgb.shape[:2])
    return labels.astype(np.int32), lab


def blur_labels(labels, k, sigma):
    """Mode-style filter: blur each label's mask, keep the strongest."""
    best = np.full(labels.shape, -1.0, np.float32)
    out = np.zeros(labels.shape, np.int32)
    for l in range(k):
        m = (labels == l).astype(np.float32)
        if not m.any():
            continue
        b = cv2.GaussianBlur(m, (0, 0), sigma)
        sel = b > best
        best[sel] = b[sel]
        out[sel] = l
    return out


def upsample_labels(labels, k, sigma, shape):
    """Smooth, sub-pixel boundaries: blur one-hot maps, upscale, argmax."""
    H, W = shape
    best = np.full((H, W), -1.0, np.float32)
    out = np.zeros((H, W), np.int32)
    for l in range(k):
        m = (labels == l).astype(np.float32)
        if not m.any():
            continue
        b = cv2.GaussianBlur(m, (0, 0), sigma)
        b = cv2.resize(b, (W, H), interpolation=cv2.INTER_CUBIC)
        sel = b > best
        best[sel] = b[sel]
        out[sel] = l
    return out


def merge_small(labels, pal_lab, min_area, min_r, max_iter=60, log=None):
    """Merge every region that is too small or too thin to paint into its most similar neighbour."""
    H, W = labels.shape
    for it in range(max_iter):
        fails = []
        for l in np.unique(labels):
            m = np.pad((labels == l).astype(np.uint8), 1)
            n, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
            if n <= 1:
                continue
            dt = cv2.distanceTransform(m, cv2.DIST_L2, 3)
            mx = ndimage.maximum(dt, cc, index=np.arange(1, n))
            for i in range(1, n):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < min_area or mx[i - 1] < min_r:
                    x, y, w, h = stats[i, :4]
                    crop = (cc[y:y + h, x:x + w] == i)
                    fails.append((area, int(l), x - 1, y - 1, w, h, crop))
        if not fails:
            break
        if log:
            log(f"    merge pass {it + 1}: {len(fails)} regions too small")
        fails.sort(key=lambda f: f[0])
        changed = 0
        for area, l, x, y, w, h, crop in fails:
            x0, y0 = max(x - 1, 0), max(y - 1, 0)
            x1, y1 = min(x + w + 1, W), min(y + h + 1, H)
            sub = labels[y0:y1, x0:x1]
            mm = np.zeros(sub.shape, np.uint8)
            ys, xs = np.nonzero(crop)
            ys = ys + y
            xs = xs + x
            ok = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
            ys, xs = ys[ok] - y0, xs[ok] - x0
            if len(ys) == 0:
                continue
            # the component may have been altered by an earlier merge this pass
            valid = sub[ys, xs] == l
            ys, xs = ys[valid], xs[valid]
            if len(ys) == 0:
                continue
            mm[ys, xs] = 1
            ring = cv2.dilate(mm, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))) > mm
            nl = sub[ring]
            nl = nl[nl != l]
            if len(nl) == 0:
                continue
            cnt = np.bincount(nl)
            cand = np.nonzero(cnt)[0]
            de = np.linalg.norm(pal_lab[cand] - pal_lab[l], axis=1)
            share = cnt[cand] / cnt.sum()
            best = cand[np.argmin(de - 10.0 * share)]
            sub[ys, xs] = best
            changed += 1
        if not changed:
            break
    return labels


def compact(labels, pal_lab):
    used = np.unique(labels)
    remap = np.zeros(labels.max() + 1, np.int32)
    remap[used] = np.arange(len(used))
    return remap[labels], pal_lab[used]


def region_colours(labels, lab_img, n):
    out = np.zeros((n, 3), np.float32)
    flat = lab_img.reshape(-1, 3)
    lf = labels.ravel()
    for l in range(n):
        out[l] = flat[lf == l].mean(axis=0)
    return out


def merge_similar_colours(labels, pal_lab, thr=5.0):
    n = len(pal_lab)
    while True:
        best = None
        for i, j in itertools.combinations(range(n), 2):
            d = np.linalg.norm(pal_lab[i] - pal_lab[j])
            if d < thr and (best is None or d < best[0]):
                best = (d, i, j)
        if not best:
            return labels, pal_lab
        _, i, j = best
        labels = np.where(labels == j, i, labels)
        labels, pal_lab = compact(labels, pal_lab)
        n = len(pal_lab)


def order_by_lightness(labels, pal_lab):
    order = np.argsort(-pal_lab[:, 0])
    remap = np.zeros(len(order), np.int32)
    remap[order] = np.arange(len(order))
    return remap[labels], pal_lab[order]


# ----------------------------------------------------------------- drawing
def edge_mask(labels):
    e = np.zeros(labels.shape, np.uint8)
    d1 = labels[:, 1:] != labels[:, :-1]
    d2 = labels[1:, :] != labels[:-1, :]
    e[:, 1:] |= d1
    e[:, :-1] |= d1
    e[1:, :] |= d2
    e[:-1, :] |= d2
    e[0, :] = e[-1, :] = 1
    e[:, 0] = e[:, -1] = 1
    return e


def line_art(labels, grey, strength):
    e = edge_mask(labels).astype(np.float32)
    e = cv2.GaussianBlur(e, (0, 0), 0.8) * strength
    e = np.clip(e, 0, 1)
    return (255 - e * (255 - grey)).astype(np.uint8)


def place_numbers(labels, min_em, max_em):
    """One number per connected region, at its pole of inaccessibility."""
    items = []
    font_cache = {}

    def font(sz):
        if sz not in font_cache:
            font_cache[sz] = load_font(sz)
        return font_cache[sz]

    for l in np.unique(labels):
        m = np.pad((labels == l).astype(np.uint8), 1)
        n, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
        if n <= 1:
            continue
        dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
        idx = np.arange(1, n)
        mx = ndimage.maximum(dt, cc, index=idx)
        pos = ndimage.maximum_position(dt, cc, index=idx)
        txt = str(int(l) + 1)
        for i, r, (py, px) in zip(idx, mx, pos):
            x, y, w, h = stats[i, :4]
            crop = cc[y:y + h, x:x + w] == i
            em = max_em
            placed = False
            while em >= min_em:
                f = font(int(em))
                bb = f.getbbox(txt)
                tw, th = bb[2] - bb[0], bb[3] - bb[1]
                x0, y0 = int(round(px - tw / 2 - 2)), int(round(py - th / 2 - 2))
                x1, y1 = x0 + tw + 4, y0 + th + 4
                cx0, cy0, cx1, cy1 = x0 - x, y0 - y, x1 - x, y1 - y
                if cx0 >= 0 and cy0 >= 0 and cx1 <= w and cy1 <= h and crop[cy0:cy1, cx0:cx1].all():
                    placed = True
                    break
                em -= max(1, em * 0.06)
            if not placed:
                em = min_em
                f = font(int(em))
                bb = f.getbbox(txt)
                tw, th = bb[2] - bb[0], bb[3] - bb[1]
            items.append((txt, px - 1 - tw / 2 - bb[0], py - 1 - th / 2 - bb[1], int(em), r))
    return items, font


def draw_numbers(base_L, labels, min_em, max_em):
    img = Image.fromarray(base_L)
    d = ImageDraw.Draw(img)
    items, font = place_numbers(labels, min_em, max_em)
    for txt, x, y, em, r in items:
        d.text((x, y), txt, fill=TEXT_GREY, font=font(em))
    return img, len(items)


# ----------------------------------------------------------------- mixing hint
def mix_colour(parts):
    tot = sum(p for _, p in parts)
    lin = np.zeros(3)
    for rgb, p in parts:
        lin += (p / tot) * np.log(np.clip(np.array(rgb, float) / 255, 0.02, 1))
    return np.exp(lin) * 255


def mixing_hints(pal_rgb, max_parts=6):
    base = [(n, np.array(c)) for n, c in PAINTS]
    base_lab = rgb2lab([c for _, c in base])
    combos = []
    for k in (1, 2, 3):
        for idx in itertools.combinations(range(len(base)), k):
            for parts in itertools.product(range(1, max_parts + 1), repeat=k):
                if sum(parts) > max_parts + 2 or (k > 1 and np.gcd.reduce(parts) > 1):
                    continue
                combos.append((idx, parts))
    cols = np.array([mix_colour([(base[i][1], p) for i, p in zip(idx, parts)]) for idx, parts in combos])
    cl = rgb2lab(cols)
    out = []
    for c in pal_rgb:
        d = np.linalg.norm(cl - rgb2lab(c)[0], axis=1) + 0.8 * np.array([sum(p) for _, p in combos])
        idx, parts = combos[int(np.argmin(d))]
        out.append(", ".join(f"{p} {base[i][0]}" if len(idx) > 1 else base[i][0] for i, p in zip(idx, parts)))
    return out


# ----------------------------------------------------------------- palette page
def palette_page(pal_rgb, share, page_px, hints, title="Colour key"):
    W, H = page_px
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    m = int(MARGIN_MM * PX_PER_MM)
    d.text((m, m), title, fill=(40, 40, 40), font=load_font(int(7 * PX_PER_MM)))
    top = m + int(14 * PX_PER_MM)
    n = len(pal_rgb)
    aw, ah = W - 2 * m, H - top - m
    cols = max(1, int(np.ceil(np.sqrt(n * aw / ah))))
    cols = min(cols, 4)
    rows = int(np.ceil(n / cols))
    while rows * cols < n:
        rows += 1
    cw, ch = aw / cols, min(ah / rows, 42 * PX_PER_MM)
    rows_fit = ch
    dia = min(cw * 0.36, ch * 0.7, 24 * PX_PER_MM)
    for i, c in enumerate(pal_rgb):
        r, q = divmod(i, cols)
        x0, y0 = m + q * cw, top + r * ch
        cx, cy = x0 + dia / 2 + cw * 0.04, y0 + ch / 2
        d.ellipse((cx - dia / 2, cy - dia / 2, cx + dia / 2, cy + dia / 2), fill=tuple(int(v) for v in c),
                  outline=(120, 120, 120), width=2)
        f = load_font(int(dia * 0.42))
        col = (255, 255, 255) if luma(c) < 140 else (30, 30, 30)
        d.text((cx, cy), str(i + 1), fill=col, font=f, anchor="mm")
        tx = cx + dia / 2 + cw * 0.04
        sm = load_font(int(min(ch * 0.18, cw * 0.075, 3.6 * PX_PER_MM)))
        d.text((tx, cy - ch * 0.17), hexof(c), fill=(40, 40, 40), font=sm, anchor="lm")
        d.text((tx, cy + ch * 0.03), f"{share[i] * 100:.0f}% of canvas" if share[i] >= 0.005 else "<1% of canvas", fill=(120, 120, 120), font=sm,
               anchor="lm")
        if hints:
            hf = load_font(int(min(ch * 0.15, cw * 0.052, 3.0 * PX_PER_MM)))
            d.text((tx, cy + ch * 0.22), hints[i], fill=(90, 90, 90), font=hf, anchor="lm")
    return img


# ----------------------------------------------------------------- main
def process(path, a, log=print):
    name = os.path.splitext(os.path.basename(path))[0]
    src = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    (pw, ph), (cw_mm, ch_mm), orient = canvas_geometry(src.width, src.height, a.size)
    out_w, out_h = int(round(cw_mm * PX_PER_MM)), int(round(ch_mm * PX_PER_MM))
    wk_w, wk_h = int(round(out_w / WORK_SCALE)), int(round(out_h / WORK_SCALE))
    ppmm = PX_PER_MM / WORK_SCALE
    min_r_mm, min_a_mm, passes, sigma = DETAIL[a.detail]
    log(f"{name}: {src.width}x{src.height} -> canvas {cw_mm:.0f}x{ch_mm:.0f} mm on {a.size} {orient}, "
        f"{a.colors} colours, detail {a.detail}")

    rgb = np.array(src.resize((wk_w, wk_h), Image.LANCZOS))
    sm = smooth(rgb, passes)

    log("  clustering colours")
    labels, lab_img = kmeans_lab(sm, a.colors, a.seed)
    k = a.colors
    pal_lab = region_colours(labels, lab_img, k)
    labels = blur_labels(labels, k, sigma)

    log("  merging small regions")
    labels = merge_small(labels, pal_lab, min_a_mm * ppmm ** 2, min_r_mm * ppmm, log=None)
    labels, pal_lab = compact(labels, pal_lab)
    pal_lab = region_colours(labels, lab_img, len(pal_lab))
    labels, pal_lab = merge_similar_colours(labels, pal_lab)
    labels = merge_small(labels, pal_lab, min_a_mm * ppmm ** 2, min_r_mm * ppmm)
    labels, pal_lab = compact(labels, pal_lab)
    pal_lab = region_colours(labels, lab_img, len(pal_lab))

    log("  smoothing outlines")
    hi = upsample_labels(labels, len(pal_lab), 1.7, (out_h, out_w))
    hi = merge_small(hi, pal_lab, 0.5 * min_a_mm * PX_PER_MM ** 2, 0.8 * min_r_mm * PX_PER_MM)
    hi, pal_lab = compact(hi, pal_lab)
    hi, pal_lab = order_by_lightness(hi, pal_lab)
    n = len(pal_lab)
    pal_rgb = lab2rgb(pal_lab)
    share = np.bincount(hi.ravel(), minlength=n) / hi.size

    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, name)

    # numbered canvas
    L = line_art(hi, LINE_GREY, 2.2)
    min_em, max_em = 2.6 * PX_PER_MM, 3.8 * PX_PER_MM
    numbers, nreg = draw_numbers(L, hi, min_em, max_em)
    numbers.save(base + "_numbers.png", dpi=(254, 254))

    # colouring page
    col = Image.fromarray(line_art(hi, 35, 3.0))
    col.save(base + "_coloring.png", dpi=(254, 254))

    # preview (flat colours, faint outlines)
    flat = pal_rgb[hi].astype(np.float32)
    e = cv2.GaussianBlur(edge_mask(hi).astype(np.float32), (0, 0), 0.8)[..., None]
    prev = flat * (1 - 0.22 * np.clip(e * 2, 0, 1))
    Image.fromarray(np.clip(prev, 0, 255).astype(np.uint8)).save(base + "_preview.png", dpi=(254, 254))

    # palette
    page_px = (int(round(pw * PX_PER_MM)), int(round(ph * PX_PER_MM)))
    hints = mixing_hints(pal_rgb) if a.mix else None
    pal_img = palette_page(pal_rgb, share, page_px, hints)
    pal_img.save(base + "_palette.png", dpi=(254, 254))

    # pdf
    if not a.no_pdf:
        def page(img):
            p = Image.new("L", page_px, 255)
            p.paste(img.convert("L"), ((page_px[0] - img.width) // 2, (page_px[1] - img.height) // 2))
            return p
        pages = [page(numbers), pal_img, page(col)]
        pages[0].save(base + "_kit.pdf", save_all=True, append_images=pages[1:], resolution=254.0,
                      quality=95)
    log(f"  {n} colours, {nreg} regions -> {base}_*.png" + ("" if a.no_pdf else " + _kit.pdf"))

    if a.strip:
        make_strip(src, base)
    return base


def make_strip(src, base, height=900, gap=30):
    parts = [src.convert("RGB")]
    for suf in ("_preview", "_numbers"):
        parts.append(Image.open(base + suf + ".png").convert("RGB"))
    parts = [p.resize((int(p.width * height / p.height), height), Image.LANCZOS) for p in parts]
    W = sum(p.width for p in parts) + gap * 4
    strip = Image.new("RGB", (W, height + 2 * gap), (245, 242, 236))
    x = gap
    for p in parts:
        strip.paste(p, (x, gap))
        x += p.width + gap
    strip.save(base + "_strip.jpg", quality=88)


def main():
    ap = argparse.ArgumentParser(description="Photo -> paint-by-numbers kit + colouring page")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--colors", "--colours", type=int, default=16, help="number of paints (default 16)")
    ap.add_argument("--detail", choices=list(DETAIL), default="medium",
                    help="low = big easy regions, high = many small ones")
    ap.add_argument("--size", default="A4", type=str.upper, choices=list(PAGES), help="paper size")
    ap.add_argument("--no-pdf", action="store_true", help="skip the PDF")
    ap.add_argument("--mix", action="store_true", help="add a rough paint-mixing hint to the palette")
    ap.add_argument("--strip", action="store_true", help="also write original | preview | numbers strip")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out", default="paint_out")
    a = ap.parse_args()
    if not 2 <= a.colors <= 60:
        sys.exit("--colors must be between 2 and 60")
    for p in a.images:
        process(p, a)


if __name__ == "__main__":
    main()
