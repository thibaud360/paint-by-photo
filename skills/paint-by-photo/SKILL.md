---
name: paint-by-photo
description: Turn the user's own photo into a printable paint-by-numbers kit (numbered canvas, colour key, painted preview, A4 PDF) and a colouring page. Use for paint by numbers, colour-by-number, colouring page from a photo, or /paint-by-photo.
---

# paint-by-photo

Makes a paint-by-numbers kit from a photo with `scripts/paint_by_photo.py` (Python 3, Pillow, NumPy, OpenCV, SciPy, scikit-learn). No image model: it smooths the photo, reduces it to N colours, merges regions too small to paint, traces the outlines and puts a number in every region.

## Outputs (per photo, in `paint_out/`)

| File | What |
|---|---|
| `NAME_numbers.png` | Outlines with a number in every region, white background |
| `NAME_palette.png` | Colour key: numbered swatches, hex codes, share of canvas |
| `NAME_preview.png` | What the painting will look like when finished |
| `NAME_coloring.png` | Outlines only, no numbers |
| `NAME_kit.pdf` | Printable: page 1 numbered canvas, page 2 palette, page 3 colouring page |

## Steps

1. Find the user's photo. Dependencies: `pip install pillow numpy opencv-python scipy scikit-learn` (add `--break-system-packages` if pip asks).
2. Pick options from what they say:
   - "easy", "for a kid", "relaxing", "big areas" -> `--detail low`
   - "detailed", "for experienced painters", "a portrait with fine features" -> `--detail high --colors 24`
   - "few paints", "cheap" -> `--colors 8`; "realistic" -> `--colors 24` or more
   - Paper: `--size A3` or `--size letter` (default A4). US users usually want `letter`.
   - Wants to know what paints to buy/mix: add `--mix` (rough hint from a basic 7-paint box, to be tested on a scrap).
   - Only wants a colouring page: use `*_coloring.png` and say so; `--no-pdf` skips the PDF.
3. Run: `python scripts/paint_by_photo.py PHOTO --colors 16 --detail medium -o paint_out`
   Takes 20 to 40 seconds per photo. It prints the colour and region counts.
4. Look at `_numbers.png` and `_preview.png` before sending. If the preview lost something important (a face, the eyes), rerun with `--detail high` or more `--colors`. If there are too many tiny regions, use `--detail low`.
5. Send `_kit.pdf` first (it is what they print), then the preview and the colouring page. Mention: print at 100% scale, number 1 is the lightest paint, and the preview is what it will look like painted.

## Notes

- Portraits and pets work best with `--detail high`. Landscapes with big skies work with `medium`.
- Photos below about 1000 px wide give coarse outlines; say so rather than hiding it.
- Describe the result as a kit made from their own photo. Do not call it AI art.
- Same photo and flags always give the same result (`--seed` changes the clustering).

## If the script is missing

`git clone --depth 1 https://github.com/thibaud360/paint-by-photo /tmp/paint-by-photo` and use `/tmp/paint-by-photo/skills/paint-by-photo/scripts/paint_by_photo.py`.
