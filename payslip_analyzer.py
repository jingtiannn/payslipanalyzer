"""
payslip_analyzer.py
-------------------
Malaysian Payslip Analyzer — Phase 2 CV Pipeline
Processes a JPG/PNG payslip image and outputs:
  - Text summary with validation flags
  - salary_chart.png  (pie chart of salary breakdown)
  - loan_chart.png    (bar chart of DSR loan eligibility)

Usage:
    python payslip_analyzer.py payslip.jpg
    python payslip_analyzer.py payslip.png --output-dir ./results

Dependencies:
    pip install opencv-python pytesseract Pillow numpy matplotlib imutils
    Also requires Tesseract-OCR installed on your system.
    Windows: https://github.com/UB-Mannheim/tesseract/wiki
"""

import sys
import re
import argparse
import os
import cv2
import numpy as np
import pytesseract
import matplotlib
matplotlib.use("Agg")           # headless backend — no display window needed
import matplotlib.pyplot as plt
from PIL import Image

# Force UTF-8 output on Windows so box-drawing characters print correctly.
# Windows cmd/PowerShell defaults to cp1252 which can't encode them.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Auto-detect Tesseract on Windows ───────────────────────────────────────
def _find_tesseract() -> None:
    """
    Search common Windows install paths for tesseract.exe and configure
    pytesseract automatically. Exits with a helpful message if not found.
    """
    import shutil
    if shutil.which("tesseract"):
        return   # already on PATH — nothing to do

    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe"),
        os.path.expandvars(r"%USERPROFILE%\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            pytesseract.pytesseract.tesseract_cmd = path
            print(f"[Setup] Tesseract found at: {path}")
            return

    # Not found anywhere — print install instructions and exit cleanly
    sys.exit(
        "\n[ERROR] Tesseract-OCR is not installed or not in PATH.\n"
        "\nHow to install on Windows:\n"
        "  1. Download the installer from:\n"
        "     https://github.com/UB-Mannheim/tesseract/wiki\n"
        "     (choose the latest *-w64-setup.exe for 64-bit Windows)\n"
        "  2. Run the installer. On the 'Select Components' screen,\n"
        "     tick 'Additional language data' and select 'Malay (msa)'.\n"
        "  3. The default install path is:\n"
        "     C:\\Program Files\\Tesseract-OCR\\\n"
        "  4. Re-run this script — it will auto-detect the install.\n"
        "\nAlternatively add Tesseract to PATH:\n"
        "  System Properties → Environment Variables → Path\n"
        "  → Add: C:\\Program Files\\Tesseract-OCR\\\n"
    )

_find_tesseract()
# ───────────────────────────────────────────────────────────────────────────

BLUR_THRESHOLD        = 100      # Laplacian variance below this = blurry

# ── EPF rates ──────────────────────────────────────────────────────────────
EPF_RATE              = 0.11     # Employee: 11% of gross salary
EPF_EMPLOYER_RATE_LOW = 0.13     # Employer: 13% when gross ≤ RM 5,000
EPF_EMPLOYER_RATE_HI  = 0.12     # Employer: 12% when gross > RM 5,000
EPF_EMPLOYER_THRESHOLD= 5000.00  # Threshold separating 13% and 12% employer rate
EPF_TOLERANCE         = 5.00     # RM 5 tolerance before flagging employee EPF

# ── SOCSO / EIS ────────────────────────────────────────────────────────────
EIS_RATE              = 0.002    # Employee AND employer EIS rate: 0.2% each
EIS_CAP               = 4000.00  # EIS insured salary ceiling (max contribution RM 8 each)
SOCSO_SALARY_CAP      = 4000.00  # SOCSO not applicable above this gross salary


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — IMAGE PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def load_image(image_path: str) -> np.ndarray:
    """
    Load image from disk, auto-correct EXIF rotation, and resize if needed.

    WHY use Pillow instead of cv2.imread directly:
    Phone cameras store portrait images as landscape with an EXIF rotation
    tag. cv2.imread ignores that tag, so a phone payslip photo that looks
    upright in any image viewer arrives SIDEWAYS to OpenCV — Tesseract then
    reads columns as rows and returns gibberish.  Pillow's exif_transpose()
    reads the EXIF tag and physically rotates the pixel data first.

    WHY resize:
    Phone cameras produce images 3000–5000 px wide.  Tesseract is optimised
    for ~300 DPI printed text which, on a typical A4 payslip, corresponds to
    roughly 2480 px wide.  Feeding a 4000 px image gives slightly WORSE OCR
    (more noise, slower) than downscaling to ~2000 px.
    """
    from PIL import ImageOps as _Iops

    # ── Pillow load (EXIF-aware) ───────────────────────────────────────────
    try:
        pil_img = Image.open(image_path)
        pil_img = _Iops.exif_transpose(pil_img)   # honour rotation tag
        pil_img = pil_img.convert("RGB")
    except Exception as exc:
        sys.exit(f"[ERROR] Cannot open image: {image_path}\n{exc}")

    # ── Resize to OCR-friendly resolution ────────────────────────────────
    # Target: ~2400 px on the longer axis.
    #   • Downscale: phone photos (4000–6000 px) have more noise than signal
    #     at that resolution; 2400 px keeps character height ≥ 30 px (good
    #     for Tesseract) while reducing noise.
    #   • Upscale: screenshots or PDF exports can be very small (600–800 px).
    #     Tesseract needs ~30 px per character height for reliable LSTM OCR.
    #     Upscaling with LANCZOS preserves edge sharpness better than BILINEAR.
    TARGET_W = 2400
    MIN_W    = 1600    # below this, Tesseract character resolution is too low
    w, h = pil_img.size
    long_side = max(w, h)
    if long_side > TARGET_W:
        # Downscale
        ratio = TARGET_W / long_side
        new_w, new_h = int(w * ratio), int(h * ratio)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        print(f"[Preprocessing] Downscaled {w}×{h} → {new_w}×{new_h} px")
    elif long_side < MIN_W:
        # Upscale — image is too small for reliable OCR
        ratio = MIN_W / long_side
        new_w, new_h = int(w * ratio), int(h * ratio)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        print(f"[Preprocessing] Upscaled  {w}×{h} → {new_w}×{new_h} px "
              f"(image was too small for reliable OCR)")

    # ── Convert to BGR numpy array for OpenCV ─────────────────────────────
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    print(f"[Preprocessing] Image loaded: {img.shape[1]}×{img.shape[0]} px")
    return img


def convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to grayscale.
    Payslip text is monochrome so colour channels only add noise for OCR.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    """
    Enhance contrast with CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Unlike global histogram equalization, CLAHE works on small tiles so it
    lifts faded text regions without over-brightening already bright areas.
    clipLimit=2.0 caps the contrast amplification to avoid noise explosion.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def remove_background(gray: np.ndarray) -> np.ndarray:
    """
    Normalise uneven illumination so adaptive thresholding works correctly
    on phone photos.

    WHY this is needed for phone photos:
    Phone cameras often capture payslips with a lighting gradient — one corner
    near a window is bright, the opposite corner is dark. When the brightness
    difference across the page is large, adaptive thresholding (even with
    blockSize=31) struggles because it computes a local threshold relative to
    the local mean. In a very dark region the mean is already low, so the
    threshold is pulled so low that faint text ink (EPF row, SOCSO row)
    is classified as background and erased. In a very bright region caused by
    glare, the threshold overshoots and dark text gets missed too.

    HOW this normalisation fixes it:
    1. Morphologically DILATE the grayscale image with a large rectangular
       kernel (~1/40 of the image width). Dilation replaces each pixel with
       the MAXIMUM value in its neighbourhood, which has the effect of
       "filling in" dark text pixels with the surrounding lighter background.
       The result is a smooth background-only luminance map with no text.
    2. Divide every pixel in the original image by the corresponding background
       pixel, scaled to 0-255. Where the background is dark (shadow), both
       numerator and denominator are small → quotient ≈ 1.0 → pixel becomes
       mid-grey (normalised). Where the background is bright (glare), both
       are large → same result. Text pixels are darker than background →
       quotient < 1.0 → they stay dark after rescaling.
    3. The output is a flat, evenly-lit image: background ≈ 240-255, text ≈
       50-150, regardless of the original lighting. Adaptive thresholding
       then reliably separates text from background everywhere.

    Kernel size: clamped to odd numbers, minimum 51 px (~2 % of a 2400 px
    image), maximum reasonable for ~4000 px photos.
    """
    h, w = gray.shape
    # Kernel must be odd.  `| 1` sets the last bit → nearest odd >= computed.
    k = max(51, (w // 40) | 1)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    dilated = cv2.dilate(gray, kernel)          # background-only estimate
    # float64 division then rescale to 0-255 uint8
    result  = cv2.divide(gray.astype(np.float64),
                         dilated.astype(np.float64),
                         scale=255.0)
    return np.clip(result, 0, 255).astype(np.uint8)


def detect_blur(gray: np.ndarray) -> float:
    """
    Compute the variance of the Laplacian as a focus/sharpness score.
    The Laplacian highlights edges; a well-focused image has strong,
    high-variance edges. A blurry image has weak edges → low variance.
    Returns the variance score (higher = sharper).
    """
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def deskew(gray: np.ndarray) -> np.ndarray:
    """
    Correct camera-angle tilt so OCR bounding boxes align with text rows.
    Method:
      1. Binarize with Otsu — makes text pixels white, background black.
      2. Find coordinates of all white (text) pixels.
      3. Fit a minimum-area bounding rectangle around those pixels.
      4. Extract the rotation angle from that rectangle.
      5. Rotate the grayscale image by that angle.
    Angles outside ±45° are ignored (likely a portrait/landscape flip,
    not a skew) to avoid catastrophic mis-rotation.
    """
    # Otsu binarization: automatically finds the best threshold
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Coordinates of all non-zero (text) pixels
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) == 0:
        return gray   # empty image — nothing to deskew

    # Minimum-area bounding rectangle gives us the skew angle
    angle = cv2.minAreaRect(coords)[-1]

    # cv2.minAreaRect returns angles in (-90, 0]; map to (-45, 45]
    if angle < -45:
        angle = 90 + angle

    # Only correct meaningful skew to avoid introducing new distortion
    if abs(angle) < 0.5:
        return gray

    # Rotate around the image centre
    (h, w) = gray.shape
    centre = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
    deskewed = cv2.warpAffine(
        gray, rotation_matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    print(f"[Preprocessing] Deskew applied — corrected angle: {angle:.2f}°")
    return deskewed


def binarize(gray: np.ndarray) -> np.ndarray:
    """
    Binarize using Adaptive Gaussian Thresholding instead of global Otsu.

    WHY adaptive is better for payslips:
    Global Otsu picks ONE threshold for the entire image. When a payslip
    has a large dark-coloured header (common in Malaysian corporate payslips),
    the dark pixels dominate the histogram and push the threshold so high
    that lighter body text — deduction rows, allowance rows — falls below it
    and becomes invisible (white on white). Tesseract then silently skips it.

    Adaptive thresholding computes a separate threshold for each small
    neighbourhood (blockSize × blockSize pixels), so it adapts to local
    contrast. Dark header → correct threshold for header. Light body text
    → correct threshold for body. Every region stays readable.

    Dynamic blockSize:
    A fixed blockSize=31 is correct for a 2400 px image (~300 DPI A4) but
    too small for higher-resolution images (blockSize becomes < 1 character
    width, so text blobs merge with background) or too large for small
    images.  We scale blockSize to ~1/80 of image width so it always covers
    roughly one full character height regardless of resolution.
      2400 px → 2400//80 = 30 → rounded to odd = 31  (original value)
       800 px →  800//80 = 10 → rounded to odd = 11
      3500 px → 3500//80 = 43 → rounded to odd = 43

    C=10 : subtract 10 from the local mean so faint text (which sits just
           above the mean) still becomes black ink after normalisation.
    """
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # Dynamic blockSize — must be odd and at least 11
    block = max(11, (gray.shape[1] // 80) | 1)
    binary  = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=block,
        C=10
    )
    return binary


def preprocess(image_path: str) -> np.ndarray:
    """
    Full preprocessing pipeline.
    Returns a cleaned, binarized image ready for OCR,
    or exits with a message if the image is too blurry.

    Pipeline order (each step explained):
      1. load_image      — EXIF rotation fix + resize to ~2400 px wide
      2. grayscale       — colour channels add noise, not information, for OCR
      3. blur detection  — halt early if photo is too shaky to read
      4. CLAHE           — lift faded/dark regions without blowing out bright ones
      5. remove_background — divide by morphological background estimate to
                             flatten lighting gradients from phone cameras
      6. deskew          — correct camera tilt so OCR bounding boxes align
      7. binarize        — adaptive threshold with dynamic blockSize
    """
    print("[Preprocessing] Loading image...")
    img  = load_image(image_path)

    print("[Preprocessing] Converting to grayscale...")
    gray = convert_to_grayscale(img)

    print("[Preprocessing] Detecting blur...")
    blur_score = detect_blur(gray)
    print(f"[Preprocessing] Blur score (Laplacian variance): {blur_score:.2f}")
    if blur_score < BLUR_THRESHOLD:
        # Soft warning only — do NOT exit.
        # Phone photos naturally score lower than scanned documents because
        # JPEG compression and slight hand-shake reduce high-frequency edges.
        # A score below the threshold doesn't mean OCR will fail; it just means
        # the photo may be slightly soft. We continue and let Tesseract decide.
        print(f"[Preprocessing] Note: low sharpness score ({blur_score:.1f}) — "
              "if results are poor, retake in brighter lighting.")

    print("[Preprocessing] Enhancing contrast with CLAHE...")
    enhanced = apply_clahe(gray)

    print("[Preprocessing] Removing background lighting gradient...")
    bg_removed = remove_background(enhanced)

    print("[Preprocessing] Deskewing...")
    corrected = deskew(bg_removed)

    print("[Preprocessing] Binarizing for OCR...")
    binary = binarize(corrected)

    print("[Preprocessing] Done.\n")
    return binary


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — OCR & FIELD EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def run_ocr(binary_img: np.ndarray) -> str:
    """
    Run Tesseract OCR and return the COMBINED text from three passes:

      Pass A  binary,   PSM 6 — single block, good for clean body rows
      Pass B  binary,   PSM 4 — single column, better for slanted photos
      Pass C  inverted, PSM 6 — catches white-on-dark sections

    WHY three passes:
    Malaysian payslips often have a coloured net-pay box with WHITE text.
    After adaptive binarization, that box becomes black background + white
    text outlines — effectively invisible to Tesseract, which expects dark
    text on a light background. Inverting the whole image and running OCR
    again makes those inverted regions readable. Since body rows are now
    light-on-dark in the inverted pass, their text is detected again too
    (with slightly different tokens). Concatenating all three passes gives
    the parser the widest possible set of lines to search through.
    """
    inverted = cv2.bitwise_not(binary_img)

    segments: list[str] = []
    for img_arr, psm in [
        (binary_img, 6),    # Pass A: single uniform block (clean prints)
        (binary_img, 4),    # Pass B: single column (slightly skewed photos)
        (inverted,   6),    # Pass C: inverted — catches white-on-dark boxes
        (binary_img, 11),   # Pass D: sparse text — finds isolated labels/values
                            #   in wide two-column layouts where PSM 6/4 merge
                            #   left and right columns into garbled single lines
    ]:
        pil_img = Image.fromarray(img_arr)
        config  = f"--psm {psm} --oem 3 -l eng"
        try:
            text = pytesseract.image_to_string(pil_img, config=config)
            segments.append(text)
        except Exception:
            pass

    return "\n".join(segments)


def _clean_number(raw: str) -> float | None:
    """
    Convert a raw OCR number string to a float, handling common OCR artifacts:
      - "2 500.00"  → spaces inside numbers (Tesseract column gap artefact)
      - "2,500.00"  → thousands comma (normal)
      - "RM 275.00" → RM/MYR currency prefix
      - "27S.00"    → OCR misread of digit as letter (filtered out)
      - "234,00"    → decimal COMMA (OCR reads "." as "," — phone photo artifact)
      - "12,75"     → same decimal-comma artifact on 2-digit cent values

    Decimal-comma detection rule:
      A comma followed by EXACTLY 2 digits at the END of the string is treated
      as a decimal separator, not a thousands separator.
        "234,00"   → "234.00"   ✓  (2 digits after comma, end of string)
        "2,500.00" → "2500.00"  ✓  (3 digits after comma → thousands comma)
        "2,500,00" → "2500.00"  ✓  (last comma decimal, first thousands)

    Returns None if the string cannot be safely converted.
    """
    s = raw.strip()
    # Strip currency prefix
    s = re.sub(r"(?i)^(RM|MYR)\s*", "", s)
    # Collapse spaces between digits (Tesseract wide-column artefact)
    s = re.sub(r"(\d)\s+(\d)", r"\1\2", s)
    # Detect decimal comma: comma + exactly 2 digits at end of string
    # Replace that trailing ",XX" with ".XX" BEFORE removing thousands commas
    if re.search(r",\d{2}$", s):
        s = re.sub(r",(\d{2})$", r".\1", s)
    # Remove remaining commas (now only thousands separators remain)
    s = s.replace(",", "")
    # Reject strings that still contain non-numeric chars after cleaning
    if not re.match(r"^\d+(\.\d{1,2})?$", s):
        return None
    try:
        val = float(s)
        return val if val >= 1.0 else None   # ignore noise values < RM 1
    except ValueError:
        return None


def _parse_amount(text: str, *keywords) -> float | None:
    """
    Locate a salary/deduction amount near any of the given keywords.

    Three strategies in order of confidence:

      Strategy 1 — First number AFTER the keyword on the same line:
            Single column:  "EPF / KWSP      275.00"  → 275.00
            Two-column:     "BASIC SALARY  2,500.00  EPF  234.00"
                            Searching "BASIC" → takes first number after
                            "BASIC" = 2,500.00  ✓  (not the last = 234.00)
                            Searching "EPF"   → takes first number after
                            "EPF"   = 234.00   ✓

      Strategy 2 — Next-line value (wide-gap single-column layouts):
            "EPF / KWSP"
            "275.00"        → 275.00

      Strategy 3 — Colon/tab separator:
            "EPF : 275.00"  or  "KWSP\t275.00"

    WHY first-number-after-keyword instead of last-number-on-line:
    Single-column payslips put the amount far to the right; two-column
    payslips interleave labels and amounts from two sections on the same
    text line. Taking the LAST number on a two-column line gives the wrong
    field's value. Taking the FIRST number immediately after the keyword
    is correct for both layouts.
    """
    kw_pattern = "|".join(re.escape(kw) for kw in keywords)
    # Require EXACTLY 2 decimal places (decimal-point OR decimal-comma).
    # Malaysian payslips always show cents (e.g. 275.00, 24.75).
    # This prevents matching integers like "11" embedded in label text
    # such as "EPF / KWSP (Employee 11%)" — "11" has no decimal, so it
    # is skipped and the real amount "495.00" further along is returned.
    # Also accept "234,00" (OCR reads "." as "," on phone photos).
    NUM = r"(?:RM\s*)?[\d][\d,]*[.,]\d{2}"

    lines = text.splitlines()

    for i, line in enumerate(lines):
        if not re.search(kw_pattern, line, re.IGNORECASE):
            continue

        # ── Strategy 1: first 2-decimal amount AFTER the keyword ──────────
        # Reading left-to-right from the keyword position handles two-column
        # payslips: "BASIC SALARY  2,500.00  EPF  234.00" — searching
        # "BASIC" finds 2,500.00 (first after keyword), not 234.00 (last).
        kw_match = re.search(kw_pattern, line, re.IGNORECASE)
        if kw_match:
            after = line[kw_match.end():]
            for raw in re.findall(NUM, after):
                val = _clean_number(raw)
                if val:
                    return val

        # ── Strategy 2: standalone number within next 10 lines ───────────
        # WHY look ahead 10 lines:
        # PSM 11 (sparse text) outputs each word/number on its own line with
        # blank separator lines between items.  The value for a keyword can be
        # 2-8 lines down — e.g. "socso" at line N, blank, "OVERTIME" (another
        # label from the left column), blank, "eet" (OCR noise), blank, "12.75"
        # at line N+8.  We scan ahead, skip blank lines and OCR noise, and
        # accept the first standalone decimal number we encounter.
        # We also accept decimal-comma format "234,00" (OCR reads "." as ",").
        for look in range(1, 11):
            if i + look >= len(lines):
                break
            ahead = lines[i + look].strip()
            if not ahead:
                continue   # skip blank separator lines
            # Match standalone number: decimal-point OR decimal-comma format
            if re.match(r"^(?:RM\s*)?[\d,]+[.,]\d{2}$", ahead):
                val = _clean_number(ahead)
                if val:
                    return val
                break      # parsed but out of sane range — don't search further

        # ── Strategy 3: colon or tab separator ────────────────────────────
        colon_match = re.search(r"(?::|　|\t)\s*" + NUM, line, re.IGNORECASE)
        if colon_match:
            val = _clean_number(colon_match.group())
            if val:
                return val

    return None


def _parse_allowances(text: str) -> float | None:
    """
    Allowances need special treatment because real payslips list them as
    multiple individual rows (Transport Allowance, Housing Allowance, Meal
    Allowance…) rather than a single field — unlike EPF or SOCSO which
    always appear once.

    Strategy 1 — Total row first:
      Some payslips print a "Total Allowance / Jumlah Elaun" summary row.
      If found, return that single value directly.

    Strategy 2 — Sum individual rows:
      When no total row exists, find EVERY line that mentions any allowance
      keyword, extract the rightmost number on that line, and sum them all.
      Skip lines that are section headers, column headers, or gross/total
      rows to avoid double-counting.

    Real-world allowance label examples covered:
      Transport Allowance, Elaun Pengangkutan, Petrol Allowance,
      Housing Allowance, Elaun Perumahan, Meal Allowance, Elaun Makan,
      Hand Phone Allowance, Travelling Allowance, Medical Allowance,
      Parking Allowance, Shift Allowance, Fixed Allowance, etc.
    """
    lo, hi = FIELD_SANE_RANGE.get("allowances", (1, 20_000))

    # ── Strategy 1: explicit total row ────────────────────────────────────
    total_row_keywords = [
        "Total Allowance", "TOTAL ALLOWANCE",
        "Jumlah Elaun", "JUMLAH ELAUN",
        "Total Allowances", "Allowance Total",
    ]
    for kw in total_row_keywords:
        val = _parse_amount(text, kw)
        if val is not None and lo <= val <= hi:
            return val

    # ── Strategy 2: sum all individual allowance rows ─────────────────────
    # Pattern that matches any allowance / additional-earnings line.
    # Anchored with \b so short tokens like "OT" don't match mid-word.
    # Overtime pay (OT, Normal OT, Rest OT, Rest Day OT) is included because
    # it forms part of gross salary alongside named allowances, and users
    # expect the "Allowances" field to account for all non-basic earnings.
    allowance_line_re = re.compile(
        r"\b(allowance|elaun|hand\s*phone|handphone|transport|housing|"
        r"meal|makan|travel|petrol|parking|medical|telephone|mobile|"
        r"shift|fixed\s+allow|incentive|cola|attendance|outstation|"
        r"laundry|uniform|"
        r"overtime|over\s+time|\bOT\b|"        # standalone OT / Normal OT / Rest OT
        r"rest\s+day|rest\s+ot|"               # rest day / rest day OT
        r"night\s*(shift|allowance|diff|hour)?|"# night shift / night allowance
        r"daily(\s+allowance|\s+attendance)?|"  # daily / daily attendance
        r"public\s+holiday|ph\s+ot|"           # public holiday OT
        r"sunday|commission|bonus)\b",
        re.IGNORECASE,
    )

    # Lines to skip: section headers, column headers, totals, gross rows.
    # We don't want to accidentally grab the GROSS SALARY line which might
    # also contain an allowance component on some payslip formats.
    skip_line_re = re.compile(
        r"\b(total|jumlah|gross|kasar|description|keterangan|"
        r"amount|pendapatan|earnings|deduction|potongan)\b",
        re.IGNORECASE,
    )

    collected: list[float] = []
    seen_amounts: set[float] = set()          # deduplicate identical rows

    for line in text.splitlines():
        if not allowance_line_re.search(line):
            continue
        if skip_line_re.search(line):
            continue

        # Take the last decimal number on this line (rightmost = amount col)
        nums = re.findall(r"[\d,]+\.\d{2}", line)
        if not nums:
            continue

        val = _clean_number(nums[-1])
        if val is None or not (lo <= val <= hi):
            continue
        if val in seen_amounts:              # skip exact duplicates from
            continue                         # multi-pass OCR
        seen_amounts.add(val)
        collected.append(val)

    if collected:
        total = round(sum(collected), 2)
        if lo <= total <= hi:
            return total

    return None


# ── Sanity ranges: (min, max) in RM ───────────────────────────────────────
# If a parsed value falls outside this range it is almost certainly an OCR
# false-positive (e.g. a year like "2017" matching near the "EIS" footer).
# Ranges are intentionally wide to cover all realistic Malaysian salaries.
FIELD_SANE_RANGE: dict[str, tuple[float, float]] = {
    "gross_salary": (500,    100_000),
    "basic_salary": (500,    100_000),
    "allowances":   (1,       20_000),
    "epf":          (1,       11_000),   # 11 % of RM 100k
    "socso":        (1,           45),   # new 6-band table max = RM 39.75 (band 3501-4000)
    "pcb":          (0,        3_000),
    "eis":          (0.5,          9),   # 0.2 % × RM 4,000 cap = RM 8.00
    "net_pay":      (200,    100_000),
}

# ── Keyword definitions shared by both text and spatial extractors ─────────
# Maps field name → list of keywords Tesseract might produce for that label.
# Covers English, BM, abbreviations, and common OCR character misreads.
FIELD_KEYWORDS: dict[str, list[str]] = {
    # ── Gross / total income ───────────────────────────────────────────────
    # Real payslips use many different labels. "TOTAL INCOME" is extremely
    # common in Malaysian SME payslips (SQL Payroll, MYOB, Kakitangan).
    "gross_salary": [
        "Gross Salary", "GROSS SALARY", "Gross Pay", "GROSS PAY",
        "Gaji Kasar", "GAJI KASAR",
        "Total Income", "TOTAL INCOME",          # ← very common on real slips
        "TOTALINCOME", "TotalIncome",            # OCR drops the space sometimes
        "Total Earnings", "TOTAL EARNINGS",
        "Jumlah Pendapatan", "Jumlah Gaji",
        "Total Emolument", "Gross Emolument",
        "Gross",
    ],
    # ── Basic salary ──────────────────────────────────────────────────────
    "basic_salary": [
        "Basic Salary", "BASIC SALARY", "Basic Pay", "BASIC PAY",
        "Gaji Pokok", "GAJI POKOK", "Gaji Asas",
        "Basic",
    ],
    # ── Allowances (individual rows summed by _parse_allowances) ──────────
    "allowances": [
        "Total Allowance", "TOTAL ALLOWANCE", "Jumlah Elaun",
        "Total Allowances", "Allowance", "ALLOWANCE", "Elaun",
    ],
    # ── EPF — DEDUCTIONS column only ──────────────────────────────────────
    # Real payslips show EPF twice: employee (Deductions section, smaller)
    # and employer (Employer Contribution section, larger). We rely on the
    # spatial sort-by-y and first-match-wins to prefer the higher placement.
    "epf": [
        "EPF", "KWSP",
        "Kumpulan Wang Simpanan Pekerja", "Employees Provident Fund",
        "EFF", "EPE", "EAF",                    # common OCR misreads
        "EPF Employee", "KWSP Pekerja", "Caruman KWSP",
    ],
    # ── SOCSO ─────────────────────────────────────────────────────────────
    "socso": [
        "SOCSO", "PERKESO",
        "Pertubuhan Keselamatan Sosial", "Social Security",
        "S0CSO", "S0CS0",                       # zero-for-O OCR misread
        "SOCSO Employee", "PERKESO Pekerja",
    ],
    # ── PCB / MTD ─────────────────────────────────────────────────────────
    "pcb": [
        "PCB", "MTD",
        "Potongan Cukai Berjadual", "Monthly Tax Deduction",
        "Income Tax", "Cukai Pendapatan",
        "Tax Deduction", "Potongan Cukai",
        "PCB/MTD", "MTD/PCB",
    ],
    # ── EIS ───────────────────────────────────────────────────────────────
    "eis": [
        "EIS", "SIP",
        "Employment Insurance System", "Insurans Pekerjaan",
        "EIS Employee", "SIP Pekerja", "E1S",
        "Els", "EIS~", "Els~",      # OCR misreads: I→l, trailing noise char
    ],
    # ── Net pay ───────────────────────────────────────────────────────────
    "net_pay": [
        "Net Pay", "NET PAY",
        "Gaji Bersih", "GAJI BERSIH",
        "Jumlah Bersih", "JUMLAH BERSIH",
        "Bawa Balik", "Take Home", "Take-Home Pay",
        "Pendapatan Bersih", "Net Salary",
        "Amount Payable", "Jumlah Dibayar",
        "NETP", "NetPay", "NETPAY",              # OCR drops space or garbles
    ],
}
# ───────────────────────────────────────────────────────────────────────────


def _extract_fields_spatial(binary_img: np.ndarray) -> dict:
    """
    Fallback extractor using Tesseract word-level bounding boxes.

    Why this is needed:
      Many real payslips — and our dummy payslip — use a two-column layout:
      labels on the far left (x≈50) and amounts on the far right (x≈700).
      When the gap is large, Tesseract's line assembler sometimes drops the
      right-hand column entirely from its text output, so line-by-line regex
      finds the keyword but never finds the number.

    How it works:
      1. Call image_to_data() to get every word with its pixel coordinates.
      2. For each keyword, find the word box on the image that matches it.
      3. Look for any number-like word that sits to the RIGHT of that box
         and on the SAME horizontal row (mid-Y within ±20 px).
      4. Take the rightmost matching number — that is the amount column.

    This works regardless of column gap width, font size, or label language,
    making it robust for any Malaysian payslip layout.
    """
    pil_img = Image.fromarray(binary_img)
    data = pytesseract.image_to_data(
        pil_img,
        output_type=pytesseract.Output.DICT,
        config="--psm 11 --oem 3",   # sparse text: better word-box positions
    )                                 # for two-column and complex layouts

    # Build a clean word list: skip empty strings and very low-confidence hits
    word_boxes = []
    for i in range(len(data["text"])):
        txt  = str(data["text"][i]).strip()
        conf = int(data["conf"][i]) if str(data["conf"][i]) != "-1" else 0
        if not txt or conf < 15:
            continue
        left = data["left"][i]
        top  = data["top"][i]
        w    = data["width"][i]
        h    = data["height"][i]
        word_boxes.append({
            "text":  txt,
            "left":  left,
            "top":   top,
            "right": left + w,
            "mid_y": top + h // 2,
        })

    results: dict[str, float | None] = {}

    # Also run on the inverted image so white-on-dark sections are covered
    inverted = cv2.bitwise_not(binary_img)
    pil_inv  = Image.fromarray(inverted)
    inv_data = pytesseract.image_to_data(
        pil_inv,
        output_type=pytesseract.Output.DICT,
        config="--psm 11 --oem 3",
    )
    for i in range(len(inv_data["text"])):
        txt  = str(inv_data["text"][i]).strip()
        conf = int(inv_data["conf"][i]) if str(inv_data["conf"][i]) != "-1" else 0
        if not txt or conf < 15:
            continue
        left = inv_data["left"][i]; top = inv_data["top"][i]
        w = inv_data["width"][i];   h  = inv_data["height"][i]
        word_boxes.append({
            "text": txt, "left": left, "top": top,
            "right": left + w, "mid_y": top + h // 2,
        })

    # Keywords that mark a box as a section/column header or grand total —
    # used when summing allowance rows to avoid double-counting.
    allowance_skip_re = re.compile(
        r"\b(total|jumlah|gross|kasar|description|keterangan|"
        r"amount|pendapatan|earnings|deduction|potongan)\b",
        re.IGNORECASE,
    )
    # Pattern for individual allowance / other-earnings word boxes.
    # Used by the spatial extractor to identify allowance keyword boxes.
    allowance_line_re_spatial = re.compile(
        r"\b(allowance|elaun|hand\s*phone|handphone|transport|housing|"
        r"meal|makan|travel|petrol|parking|medical|telephone|mobile|"
        r"shift|fixed\s+allow|incentive|cola|attendance|outstation|"
        r"laundry|uniform|"
        r"overtime|over\s+time|\bOT\b|"
        r"rest\s+day|rest\s+ot|"
        r"night\s*(shift|allowance|diff|hour)?|"
        r"daily(\s+allowance|\s+attendance)?|"
        r"public\s+holiday|ph\s+ot|"
        r"sunday|commission|bonus)\b",
        re.IGNORECASE,
    )

    for field, keywords in FIELD_KEYWORDS.items():
        # Word-boundary pattern so short labels like "SIP" don't match
        # substrings inside longer words like "PAYSLIP" or "BERSIH".
        # (?<!\w) = not preceded by a word character
        # (?!\w)  = not followed by a word character
        kw_pattern = "(?<!" + r"\w)(" + "|".join(
            re.escape(k) for k in keywords
        ) + r")(?!\w)"

        lo, hi = FIELD_SANE_RANGE.get(field, (0, float("inf")))
        found_value: float | None = None

        # Find all word boxes whose text FULLY or BOUNDARY matches a keyword.
        # Sort by mid_y ASCENDING so matches higher on the page (deductions
        # section) are tried BEFORE matches lower on the page (employer
        # contribution section). This prevents grabbing the larger employer
        # EPF/SOCSO instead of the smaller employee deduction values.
        kw_boxes = sorted(
            [wb for wb in word_boxes
             if re.search(kw_pattern, wb["text"], re.IGNORECASE)],
            key=lambda wb: wb["mid_y"]          # top of page first
        )

        # ── Allowances: sum ALL matching rows ─────────────────────────────
        # Every other field appears once; allowances appear once per line
        # (Transport, Housing, Meal, Overtime…) so we collect and sum them.
        # We use the broader allowance_line_re_spatial (which includes OT,
        # night shift, daily attendance, rest day, etc.) in addition to the
        # narrow FIELD_KEYWORDS list so overtime rows are captured too.
        if field == "allowances":
            seen_vals: set[float] = set()
            row_amounts: list[float] = []

            # Extend kw_boxes with any word box matching the broad pattern
            # that wasn't already captured by FIELD_KEYWORDS
            extra_boxes = [
                wb for wb in word_boxes
                if allowance_line_re_spatial.search(wb["text"])
                and wb not in kw_boxes
            ]
            all_allowance_boxes = kw_boxes + extra_boxes

            for kw_box in all_allowance_boxes:
                # Skip total/header rows — we want individual rows only
                if allowance_skip_re.search(kw_box["text"]):
                    continue

                right_numbers = []
                for wb in word_boxes:
                    if wb["left"] <= kw_box["right"]:
                        continue
                    if abs(wb["mid_y"] - kw_box["mid_y"]) > 25:
                        continue
                    # Require a decimal separator (point OR comma) so we don't
                    # grab integers like "11" from "Employee 11%" labels.
                    # Also accept "234,00" (OCR decimal-comma artifact).
                    if not re.search(r"\d[.,]\d", wb["text"]):
                        continue
                    val = _clean_number(wb["text"])
                    if val and lo <= val <= hi:
                        right_numbers.append((wb["left"], val))

                if right_numbers:
                    right_numbers.sort(key=lambda x: x[0], reverse=True)
                    best_val = right_numbers[0][1]
                    if best_val not in seen_vals:
                        seen_vals.add(best_val)
                        row_amounts.append(best_val)

            # If a "Total Allowance" box exists, prefer its value directly
            total_kw_re = re.compile(
                r"\b(total\s+allow|jumlah\s+elaun)\b", re.IGNORECASE
            )
            total_boxes = [wb for wb in kw_boxes
                           if total_kw_re.search(wb["text"])]
            for tb in total_boxes:
                right_numbers = []
                for wb in word_boxes:
                    if wb["left"] <= tb["right"]:
                        continue
                    if abs(wb["mid_y"] - tb["mid_y"]) > 25:
                        continue
                    val = _clean_number(wb["text"])
                    if val and lo <= val <= hi:
                        right_numbers.append((wb["left"], val))
                if right_numbers:
                    right_numbers.sort(key=lambda x: x[0], reverse=True)
                    found_value = right_numbers[0][1]
                    break

            if found_value is None and row_amounts:
                total = round(sum(row_amounts), 2)
                found_value = total if lo <= total <= hi else None

        # ── All other fields: nearest valid number wins ────────────────────
        # WHY nearest instead of rightmost:
        # Two-column payslips put label + value + label + value on one row:
        #   "BASIC SALARY  2,500.00  |  EPF  234.00"
        # Taking the rightmost number for "BASIC SALARY" gives 234.00 (wrong).
        # Taking the NEAREST number to the right gives 2,500.00 (correct).
        # For single-column payslips there is usually only one number to the
        # right of any label, so nearest == rightmost — no regression.
        else:
            for kw_box in kw_boxes:
                right_numbers = []
                for wb in word_boxes:
                    if wb["left"] <= kw_box["right"]:
                        continue                 # not to the right
                    if abs(wb["mid_y"] - kw_box["mid_y"]) > 25:
                        continue                 # not the same row
                    # Require a decimal separator (point OR comma) to avoid
                    # matching integers like "11" from "Employee 11%".
                    # Also accept "234,00" (OCR decimal-comma artifact).
                    if not re.search(r"\d[.,]\d", wb["text"]):
                        continue
                    val = _clean_number(wb["text"])
                    if val:
                        right_numbers.append((wb["left"], val))

                if right_numbers:
                    # Nearest (leftmost of those to the right) first
                    right_numbers.sort(key=lambda x: x[0])
                    for _, candidate in right_numbers:
                        if lo <= candidate <= hi:
                            found_value = candidate
                            break

                if found_value is not None:
                    break

        results[field] = found_value

    return results


def extract_fields(raw_text: str,
                   binary_img: np.ndarray | None = None) -> dict:
    """
    Parse OCR text and map values to payslip field names.

    Pass 1 — Line-by-line text regex (fast; works when OCR outputs
             label and value on the same text line).
    Pass 2 — Spatial bounding-box fallback (used when Pass 1 misses a
             field because the label and value were in separate columns
             that Tesseract split across lines).

    If a field still shows 'not detected' after both passes, run with
    --debug to see the raw OCR text and add the exact label wording
    from that payslip to FIELD_KEYWORDS above.
    """
    print("[OCR] Extracting payslip fields from text...")

    # ── Pass 1: line-by-line text regex ──────────────────────────────────
    raw_fields: dict[str, float | None] = {
        field: _parse_amount(raw_text, *kws)
        for field, kws in FIELD_KEYWORDS.items()
    }
    # Allowances use a dedicated parser that sums multiple individual rows
    # instead of stopping at the first match (e.g. Transport + Housing).
    raw_fields["allowances"] = _parse_allowances(raw_text)

    # Apply sanity range filter to Pass 1 results — discard values that are
    # clearly wrong (e.g. "EIS" matching a year like 2017 in the footer).
    fields: dict[str, float | None] = {}
    for field, val in raw_fields.items():
        if val is not None:
            lo, hi = FIELD_SANE_RANGE.get(field, (0, float("inf")))
            fields[field] = val if lo <= val <= hi else None
        else:
            fields[field] = None

    # Auto-derive gross from basic + allowances when gross not found directly
    if fields["gross_salary"] is None:
        if fields["basic_salary"] is not None and fields["allowances"] is not None:
            fields["gross_salary"] = round(
                fields["basic_salary"] + fields["allowances"], 2
            )
            print("[OCR] Gross salary derived from basic + allowances.")

    # Sanity: gross must be >= basic salary.
    # In two-column layouts, OCR noise can corrupt "2,900.00" → "900.00"
    # (the leading digit gets merged with separator noise and dropped).
    # If detected gross < basic, it's clearly wrong — recompute.
    if (fields["gross_salary"] is not None
            and fields["basic_salary"] is not None
            and fields["gross_salary"] < fields["basic_salary"]):
        if fields["allowances"] is not None:
            corrected = round(fields["basic_salary"] + fields["allowances"], 2)
            print(f"[OCR] Gross corrected: detected RM {fields['gross_salary']:,.2f} "
                  f"< basic RM {fields['basic_salary']:,.2f}; "
                  f"recomputed from basic + allowances = RM {corrected:,.2f}")
            fields["gross_salary"] = corrected
        else:
            # No allowances known — at minimum gross equals basic
            print(f"[OCR] Gross corrected: detected RM {fields['gross_salary']:,.2f} "
                  f"< basic; set to basic RM {fields['basic_salary']:,.2f}")
            fields["gross_salary"] = fields["basic_salary"]

    # Math-consistency check: basic + allowances should equal gross.
    # If they don't (because some overtime/allowance rows were missed by OCR),
    # re-derive allowances as gross − basic.  This covers any row type that
    # the keyword list doesn't yet recognise: new allowance names, custom OT
    # labels, etc.  Only overrides when the discrepancy is > RM 1.00.
    if (fields["gross_salary"] is not None
            and fields["basic_salary"] is not None):
        derived_allow = round(fields["gross_salary"] - fields["basic_salary"], 2)
        lo_a, hi_a = FIELD_SANE_RANGE.get("allowances", (1, 20_000))
        if derived_allow > 0:
            if fields["allowances"] is None:
                fields["allowances"] = derived_allow
                print(f"[OCR] Allowances derived from gross − basic: "
                      f"RM {derived_allow:,.2f}")
            elif abs(fields["allowances"] - derived_allow) > 1.00:
                print(f"[OCR] Allowances adjusted: detected "
                      f"RM {fields['allowances']:,.2f} but gross − basic = "
                      f"RM {derived_allow:,.2f}; using derived value.")
                fields["allowances"] = derived_allow

    # ── Pass 2: spatial bounding-box fallback ─────────────────────────────
    # Run only when some fields are still missing AND we have the image.
    # We skip allowances/basic_salary from "must find" because not every
    # payslip shows them separately from gross.
    must_find = {"epf", "socso", "pcb", "eis", "net_pay", "gross_salary"}
    missing   = [k for k in must_find if fields[k] is None]

    if missing and binary_img is not None:
        print(f"[OCR] Pass 1 missed: {missing} — trying spatial analysis...")
        spatial = _extract_fields_spatial(binary_img)
        for key in missing:
            if spatial.get(key) is not None:
                fields[key] = spatial[key]
                print(f"[OCR] Spatial found {key}: RM {fields[key]:,.2f}")

    # ── Net pay computed fallback ─────────────────────────────────────────
    # When the net pay box has white-on-dark text that OCR cannot read,
    # compute net pay from gross minus all detected deductions. This is
    # arithmetically reliable if at least gross and one deduction are found.
    if fields["net_pay"] is None and fields["gross_salary"] is not None:
        deduction_keys = ("epf", "socso", "pcb", "eis")
        detected_deductions = [
            fields[k] for k in deduction_keys if fields[k] is not None
        ]
        if detected_deductions:
            computed = round(
                fields["gross_salary"] - sum(detected_deductions), 2
            )
            lo, hi = FIELD_SANE_RANGE["net_pay"]
            if lo <= computed <= hi:
                fields["net_pay"] = computed
                print(f"[OCR] Net pay computed from gross − deductions: "
                      f"RM {computed:,.2f}")

    # ── Summary ───────────────────────────────────────────────────────────
    for name, value in fields.items():
        status = f"RM {value:,.2f}" if value is not None else "not detected"
        print(f"         {name:<15}: {status}")

    print()
    return fields


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — STATUTORY VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def validate_epf(gross: float | None, epf: float | None) -> str:
    """
    EPF employee contribution is normally 11% of gross salary.
    Reference: EPF Act 1991, Third Schedule.

    Special cases handled:
    • 9% reduced rate — Malaysian government allowed employees to opt for
      a temporary 9% rate (down from 11%) from April 2020 to December 2022
      as a COVID-19 cash-flow relief measure. Payslips from that period
      will show ~9% and are still CORRECT.
    • RM 5 tolerance — absorbs rounding differences between payroll systems.
    """
    if gross is None or epf is None:
        return "unable to verify (field not detected)"

    expected_11 = round(gross * 0.11, 2)   # standard rate
    expected_9  = round(gross * 0.09, 2)   # COVID reduced rate (2020-2022)

    diff_11 = abs(epf - expected_11)
    diff_9  = abs(epf - expected_9)

    if diff_11 <= EPF_TOLERANCE:
        return f"CORRECT  (11% rate — expected RM {expected_11:,.2f}, found RM {epf:,.2f})"
    if diff_9 <= EPF_TOLERANCE:
        return (f"CORRECT  (9% reduced rate — expected RM {expected_9:,.2f}, "
                f"found RM {epf:,.2f}; note: standard rate is 11%)")
    return (f"WARNING  (expected RM {expected_11:,.2f} at 11% or "
            f"RM {expected_9:,.2f} at 9%, found RM {epf:,.2f} "
            f"— difference RM {diff_11:,.2f}; verify with HR)")


# SOCSO (PERKESO) First Category contribution table.
# Each entry: (wage_ceiling_inclusive, employee_RM, employer_RM)
# Source: Employees' Social Security Act 1969, Second Schedule.
# Covers Employment Injury Insurance + Invalidity Pension (combined).
# Employee rate ≈ 1.0% of salary; Employer rate ≈ 1.75% of salary.
# SOCSO is NOT applicable for gross salary above RM 4,000.
_SOCSO_TABLE: list[tuple[float, float, float]] = [
    (1500, 14.75, 26.10),
    (2000, 19.75, 34.85),
    (2500, 24.75, 43.60),
    (3000, 29.75, 52.35),
    (3500, 34.75, 61.10),
    (4000, 39.75, 69.85),
]


def _get_socso_rates(gross: float) -> tuple[float | None, float | None]:
    """
    Look up (employee, employer) SOCSO contributions for a given gross salary.
    Returns (None, None) when gross > RM 4,000 — SOCSO not applicable.
    """
    if gross > SOCSO_SALARY_CAP:
        return None, None
    for ceiling, emp, empr in _SOCSO_TABLE:
        if gross <= ceiling:
            return emp, empr
    return None, None   # fallback (shouldn't be reached)


def _expected_socso(gross: float) -> float | None:
    """Return expected employee SOCSO only (used by validation)."""
    emp, _ = _get_socso_rates(gross)
    return emp


def validate_socso(gross: float | None, socso: float | None) -> str:
    """
    Validate SOCSO employee contribution against the PERKESO salary band table.
    Reference: Employees' Social Security Act 1969, Second Schedule.
    SOCSO is only applicable for gross salaries up to RM 4,000.
    """
    if gross is None or socso is None:
        return "unable to verify (field not detected)"

    if gross > SOCSO_SALARY_CAP:
        return (f"N/A  (gross RM {gross:,.2f} exceeds SOCSO salary cap "
                f"of RM {SOCSO_SALARY_CAP:,.2f})")

    expected  = _expected_socso(gross)
    if expected is None:
        return "unable to verify"
    tolerance = 1.00            # RM 1.00 tolerance for rounding / band edge
    if abs(socso - expected) <= tolerance:
        return f"CORRECT  (expected RM {expected:.2f}, found RM {socso:,.2f})"
    return (f"WARNING  (expected RM {expected:.2f} for gross RM {gross:,.2f}, "
            f"found RM {socso:,.2f} — verify with HR)")


def validate_pcb(pcb: float | None) -> str:
    """
    PCB (Potongan Cukai Berjadual / Monthly Tax Deduction) depends on
    personal reliefs, marital status, and number of dependants — details
    not visible on the payslip. We flag it for self-verification instead
    of attempting a calculation that could be misleading.
    Reference: Income Tax Act 1967, LHDN PCB schedule.
    """
    if pcb is None:
        return "not detected"
    return (f"RM {pcb:,.2f} — please verify with LHDN (exact PCB depends "
            "on your personal tax reliefs; use e-PCB at https://lhdn.gov.my)")


def validate_eis(gross: float | None, eis: float | None) -> str:
    """
    EIS employee contribution = 0.2% of insured salary (capped at RM 4,000).
    Employer also contributes 0.2% — both sides are capped at RM 8.00/month.
    Reference: Employment Insurance System Act 2017.
    """
    if gross is None or eis is None:
        return "unable to verify (field not detected)"
    insured  = min(gross, EIS_CAP)
    expected = round(insured * EIS_RATE, 2)
    if abs(eis - expected) <= 0.10:
        return f"CORRECT  (expected RM {expected:,.2f}, found RM {eis:,.2f})"
    return (f"WARNING  (expected RM {expected:,.2f}, found RM {eis:,.2f}; "
            "verify with HR)")


def calculate_employer_contributions(gross: float | None) -> dict:
    """
    Calculate employer-side statutory contributions for a given gross salary.

    EPF employer:
      13% of gross when gross ≤ RM 5,000 (most employees / fresh graduates)
      12% of gross when gross > RM 5,000
      Reference: EPF Act 1991, Third Schedule.

    SOCSO employer:
      Looked up from the PERKESO salary-band table (≈ 1.75% of salary).
      Not applicable when gross > RM 4,000.
      Reference: Employees' Social Security Act 1969, Second Schedule.

    EIS employer:
      0.2% of insured salary, capped at RM 4,000 (max RM 8.00/month).
      Mirrors the employee rate exactly.
      Reference: Employment Insurance System Act 2017.

    Returns a dict with all numeric amounts (float) or None when not applicable.
    """
    if gross is None:
        return {
            "epf_employer":           None,
            "epf_employee_expected":  None,
            "epf_total":              None,
            "epf_employer_rate":      None,
            "socso_employer":         None,
            "socso_employee_expected":None,
            "socso_total":            None,
            "socso_applicable":       None,
            "eis_employer":           None,
            "eis_employee_expected":  None,
            "eis_total":              None,
        }

    # ── EPF ──────────────────────────────────────────────────────────────────
    epf_employer_rate    = EPF_EMPLOYER_RATE_LOW if gross <= EPF_EMPLOYER_THRESHOLD \
                           else EPF_EMPLOYER_RATE_HI
    epf_employee_exp     = round(gross * EPF_RATE, 2)
    epf_employer         = round(gross * epf_employer_rate, 2)
    epf_total            = round(epf_employee_exp + epf_employer, 2)

    # ── SOCSO ─────────────────────────────────────────────────────────────────
    socso_applicable     = gross <= SOCSO_SALARY_CAP
    socso_emp, socso_empr = _get_socso_rates(gross)
    socso_total          = round(socso_emp + socso_empr, 2) \
                           if (socso_emp is not None and socso_empr is not None) \
                           else None

    # ── EIS ───────────────────────────────────────────────────────────────────
    insured              = min(gross, EIS_CAP)
    eis_each             = round(insured * EIS_RATE, 2)   # same rate both sides
    eis_total            = round(eis_each * 2, 2)

    return {
        "epf_employer":            epf_employer,
        "epf_employee_expected":   epf_employee_exp,
        "epf_total":               epf_total,
        "epf_employer_rate":       epf_employer_rate,
        "socso_employer":          socso_empr,
        "socso_employee_expected": socso_emp,
        "socso_total":             socso_total,
        "socso_applicable":        socso_applicable,
        "eis_employer":            eis_each,
        "eis_employee_expected":   eis_each,
        "eis_total":               eis_total,
    }


def run_validation(fields: dict) -> dict:
    """
    Run all statutory validation checks.
    Returns a dict containing:
      - validation status strings for each deduction field
      - 'employer' sub-dict with all calculated employer contribution amounts
    """
    print("[Validation] Checking statutory deductions...")
    employer = calculate_employer_contributions(fields["gross_salary"])
    results  = {
        "epf":      validate_epf(fields["gross_salary"], fields["epf"]),
        "socso":    validate_socso(fields["gross_salary"], fields["socso"]),
        "pcb":      validate_pcb(fields["pcb"]),
        "eis":      validate_eis(fields["gross_salary"], fields["eis"]),
        "employer": employer,
    }
    print()
    return results


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — FINANCIAL CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════

def calculate_financials(net_pay: float | None) -> dict:
    """
    Compute DSR-based loan eligibility and 50/30/20 budget breakdown.
    DSR (Debt Service Ratio): Malaysian banks typically allow up to 60%
    of net pay for total monthly loan repayments.
    50/30/20 rule: 50% needs, 30% wants, 20% savings.
    Returns None values when net pay is not available.
    """
    if net_pay is None:
        return {
            "max_loan_installment": None,
            "needs_50":             None,
            "wants_30":             None,
            "savings_20":           None,
        }
    return {
        "max_loan_installment": round(net_pay * 0.60, 2),
        "needs_50":             round(net_pay * 0.50, 2),
        "wants_30":             round(net_pay * 0.30, 2),
        "savings_20":           round(net_pay * 0.20, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — TEXT SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

def fmt(value: float | None, prefix: str = "RM ") -> str:
    """Format a float as currency string, or 'not detected' if None."""
    if value is None:
        return "not detected"
    return f"{prefix}{value:,.2f}"


def print_summary(fields: dict, validation: dict, financials: dict) -> None:
    """Print the full analysis report to stdout."""
    divider = "─" * 62
    emp     = validation["employer"]   # employer contribution dict

    print("\n" + "═" * 62)
    print("  PAYSLIP ANALYSIS REPORT")
    print("  OpenClaw Payslip Analyzer v1.0.0 — Malaysian Edition")
    print("═" * 62)

    # ── Earnings ─────────────────────────────────────────────────────────────
    print(f"\nEARNINGS")
    print(divider)
    print(f"  Basic Salary        : {fmt(fields['basic_salary'])}")
    print(f"  Allowances / OT     : {fmt(fields['allowances'])}")
    print(f"  GROSS SALARY        : {fmt(fields['gross_salary'])}")

    # ── Employee deductions ───────────────────────────────────────────────────
    print(f"\nEMPLOYEE DEDUCTIONS  (deducted from your pay)")
    print(divider)

    # EPF
    emp_rate_pct = int((emp["epf_employer_rate"] or 0) * 100) if emp["epf_employer_rate"] else "?"
    print(f"  EPF / KWSP  (Employee 11%)")
    print(f"    From payslip      : {fmt(fields['epf'])}")
    print(f"    └─ Validation     : {validation['epf']}")

    # SOCSO
    print(f"  SOCSO / PERKESO")
    print(f"    From payslip      : {fmt(fields['socso'])}")
    print(f"    └─ Validation     : {validation['socso']}")
    if emp["socso_applicable"] is False:
        print(f"    └─ Note           : SOCSO not applicable (gross > "
              f"RM {SOCSO_SALARY_CAP:,.0f})")

    # PCB
    print(f"  PCB / MTD  (Income Tax)")
    print(f"    From payslip      : {fmt(fields['pcb'])}")
    print(f"    └─ Validation     : {validation['pcb']}")

    # EIS
    print(f"  EIS / SIP  (Employee 0.2%)")
    print(f"    From payslip      : {fmt(fields['eis'])}")
    print(f"    └─ Validation     : {validation['eis']}")

    # ── Net pay ───────────────────────────────────────────────────────────────
    print(f"\nNET PAY  (take-home)")
    print(divider)
    print(f"  NET PAY             : {fmt(fields['net_pay'])}")

    # ── Employer contributions ────────────────────────────────────────────────
    print(f"\nEMPLOYER CONTRIBUTIONS  (paid by employer, not deducted from you)")
    print(divider)

    # EPF employer
    print(f"  EPF / KWSP  (Employer {emp_rate_pct}%)")
    print(f"    Employee 11%      : {fmt(emp['epf_employee_expected'])}")
    print(f"    Employer {emp_rate_pct}%      : {fmt(emp['epf_employer'])}")
    print(f"    Total EPF         : {fmt(emp['epf_total'])}")

    # SOCSO employer
    if emp["socso_applicable"] is True:
        print(f"  SOCSO / PERKESO")
        print(f"    Employee          : {fmt(emp['socso_employee_expected'])}")
        print(f"    Employer          : {fmt(emp['socso_employer'])}")
        print(f"    Total SOCSO       : {fmt(emp['socso_total'])}")
    elif emp["socso_applicable"] is False:
        print(f"  SOCSO / PERKESO     : Not applicable (gross > "
              f"RM {SOCSO_SALARY_CAP:,.0f})")
    else:
        print(f"  SOCSO / PERKESO     : not detected")

    # EIS employer
    print(f"  EIS / SIP  (0.2% each side)")
    print(f"    Employee 0.2%     : {fmt(emp['eis_employee_expected'])}")
    print(f"    Employer 0.2%     : {fmt(emp['eis_employer'])}")
    print(f"    Total EIS         : {fmt(emp['eis_total'])}")

    # Total employment cost
    gross = fields["gross_salary"]
    if gross is not None:
        extra = sum(
            v for v in [emp["epf_employer"], emp["socso_employer"], emp["eis_employer"]]
            if v is not None
        )
        total_cost = round(gross + extra, 2)
        print(f"\n  ── Total Cost to Employer ──────────────────────────")
        print(f"  Gross Salary      : {fmt(gross)}")
        print(f"  + Employer EPF    : {fmt(emp['epf_employer'])}")
        if emp["socso_applicable"]:
            print(f"  + Employer SOCSO  : {fmt(emp['socso_employer'])}")
        print(f"  + Employer EIS    : {fmt(emp['eis_employer'])}")
        print(f"  = TOTAL COST      : RM {total_cost:,.2f}")

    # ── Financial insights ────────────────────────────────────────────────────
    print(f"\nFINANCIAL INSIGHTS  (DSR + 50/30/20)")
    print(divider)
    print(f"  Max Monthly Loan Repayment (60%) : {fmt(financials['max_loan_installment'])}")
    print(f"  Needs Budget              (50%) : {fmt(financials['needs_50'])}")
    print(f"  Wants Budget              (30%) : {fmt(financials['wants_30'])}")
    print(f"  Savings Target            (20%) : {fmt(financials['savings_20'])}")

    # ── Disclaimer ────────────────────────────────────────────────────────────
    print(f"\nDISCLAIMER")
    print(divider)
    print(
        "  Results are estimates based on publicly available Malaysian\n"
        "  statutory rates (EPF Act 1991, SOCSO Act 1969, EIS Act 2017).\n"
        "  PCB depends on personal tax reliefs — verify at lhdn.gov.my.\n"
        "  This tool does not store or transmit your payslip image.\n"
        "  Verify all figures with your HR department or a licensed\n"
        "  financial advisor before making financial decisions."
    )
    print("═" * 62 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6 — CHART GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def _safe_amount(value: float | None, default: float = 0.0) -> float:
    """Return 0.0 for missing fields so charts can still render."""
    return value if value is not None else default


def generate_pie_chart(fields: dict, output_path: str) -> None:
    """
    Employee salary breakdown pie chart.
    Shows how the gross salary is split into employee deductions and take-home.
    """
    gross = _safe_amount(fields["gross_salary"])
    if gross <= 0:
        print("[Chart] Gross salary not available — skipping salary pie chart.")
        return

    epf   = _safe_amount(fields["epf"])
    socso = _safe_amount(fields["socso"])
    pcb   = _safe_amount(fields["pcb"])
    eis   = _safe_amount(fields["eis"])
    net   = _safe_amount(fields["net_pay"])

    if net <= 0:
        net = max(gross - epf - socso - pcb - eis, 0)

    labels  = ["EPF Employee (11%)", "SOCSO Employee", "PCB / Tax",
               "EIS Employee (0.2%)", "Take-Home Pay"]
    sizes   = [epf, socso, pcb, eis, net]
    colors  = ["#FF5722", "#FF9800", "#9C27B0", "#795548", "#2196F3"]
    explode = [0, 0, 0, 0, 0.05]

    filtered = [(l, s, c, e)
                for l, s, c, e in zip(labels, sizes, colors, explode)
                if s > 0]
    if not filtered:
        print("[Chart] No non-zero values — skipping salary pie chart.")
        return
    labels, sizes, colors, explode = zip(*filtered)

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, explode=explode,
        autopct="%1.1f%%", startangle=140, pctdistance=0.82,
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_color("white")
        at.set_fontweight("bold")

    ax.set_title(
        f"Employee Salary Breakdown\n(Gross: RM {gross:,.2f})",
        fontsize=13, fontweight="bold", pad=20
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Chart] Salary pie chart saved  → {output_path}")


def generate_employer_cost_chart(fields: dict, validation: dict,
                                 output_path: str) -> None:
    """
    Employer total cost pie chart.
    Shows how the total employment cost (gross + employer statutory
    contributions) is broken down, helping employees understand the
    true cost of employing them beyond just their take-home pay.
    """
    gross = _safe_amount(fields["gross_salary"])
    if gross <= 0:
        print("[Chart] Gross salary not available — skipping employer cost chart.")
        return

    emp           = validation["employer"]
    epf_employer  = _safe_amount(emp["epf_employer"])
    socso_empr    = _safe_amount(emp["socso_employer"])
    eis_employer  = _safe_amount(emp["eis_employer"])
    net           = _safe_amount(fields["net_pay"])

    if net <= 0:
        epf_e = _safe_amount(fields["epf"])
        soc_e = _safe_amount(fields["socso"])
        pcb_e = _safe_amount(fields["pcb"])
        eis_e = _safe_amount(fields["eis"])
        net   = max(gross - epf_e - soc_e - pcb_e - eis_e, 0)

    epf_employee  = _safe_amount(fields["epf"])
    socso_emp_val = _safe_amount(fields["socso"])
    pcb           = _safe_amount(fields["pcb"])
    eis_employee  = _safe_amount(fields["eis"])

    total_cost = gross + epf_employer + socso_empr + eis_employer

    labels = [
        f"Take-Home Pay",
        f"EPF Employee (11%)",
        f"EPF Employer ({int((emp['epf_employer_rate'] or 0.13)*100)}%)",
        f"SOCSO Employee",
        f"SOCSO Employer",
        f"EIS Employee (0.2%)",
        f"EIS Employer (0.2%)",
        f"PCB / Tax",
    ]
    sizes = [
        net, epf_employee, epf_employer,
        socso_emp_val, socso_empr,
        eis_employee, eis_employer,
        pcb,
    ]
    colors = [
        "#2196F3",   # net pay — blue
        "#FF5722",   # EPF employee — deep orange
        "#FF8A65",   # EPF employer — light orange
        "#FF9800",   # SOCSO employee — amber
        "#FFCC02",   # SOCSO employer — yellow
        "#4CAF50",   # EIS employee — green
        "#81C784",   # EIS employer — light green
        "#9C27B0",   # PCB — purple
    ]
    explode = [0.05, 0, 0, 0, 0, 0, 0, 0]

    # Remove zero/None slices
    filtered = [(l, s, c, e)
                for l, s, c, e in zip(labels, sizes, colors, explode)
                if s > 0]
    if not filtered:
        print("[Chart] No employer cost data — skipping employer cost chart.")
        return
    labels, sizes, colors, explode = zip(*filtered)

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, explode=explode,
        autopct="%1.1f%%", startangle=140, pctdistance=0.80,
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color("white")
        at.set_fontweight("bold")

    ax.set_title(
        f"Total Employment Cost Breakdown\n"
        f"(Gross RM {gross:,.2f}  |  Total Cost to Employer: RM {total_cost:,.2f})",
        fontsize=12, fontweight="bold", pad=20
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Chart] Employer cost chart saved → {output_path}")


def generate_bar_chart(fields: dict, financials: dict, output_path: str) -> None:
    """
    Bar chart: net pay vs max monthly loan repayment vs savings target.
    Side-by-side bars make it easy to see DSR headroom at a glance.
    """
    net       = _safe_amount(fields["net_pay"])
    max_loan  = _safe_amount(financials["max_loan_installment"])
    savings   = _safe_amount(financials["savings_20"])

    if net <= 0:
        print("[Chart] Net pay not available — skipping bar chart.")
        return

    categories = ["Net Monthly Pay", "Max Loan\nRepayment (60%)", "Savings\nTarget (20%)"]
    values     = [net, max_loan, savings]
    colors     = ["#2196F3", "#4CAF50", "#FF9800"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(categories, values, color=colors, width=0.5, edgecolor="white")

    # Add RM labels on top of each bar
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + net * 0.01,
            f"RM {val:,.2f}",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold"
        )

    ax.set_ylabel("Amount (RM)", fontsize=11)
    ax.set_title(
        "Loan Eligibility vs Net Pay (DSR 60%)",
        fontsize=13, fontweight="bold", pad=15
    )
    ax.set_ylim(0, net * 1.20)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"RM {x:,.0f}")
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Chart] Bar chart saved  → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API  (called by OpenClaw tool wrapper)
# ═══════════════════════════════════════════════════════════════════════════

def analyze_payslip(image_path: str, output_dir: str = "output") -> dict:
    """
    OpenClaw entry point — runs the full pipeline and returns a structured dict.

    Parameters
    ----------
    image_path : str
        Absolute or relative path to the payslip image (JPG / PNG).
    output_dir : str
        Directory where the three chart PNGs will be saved.
        Created automatically if it does not exist.

    Returns
    -------
    dict with keys:
        status_code              "OK" | "ERROR_FILE_NOT_FOUND" |
                                 "ERROR_NO_TEXT" | "ERROR_PIPELINE"
        text_summary             Full analysis report as a string (None on error).
        salary_chart_path        Absolute path to salary_chart.png (None on error).
        loan_chart_path          Absolute path to loan_chart.png (None on error).
        employer_cost_chart_path Absolute path to employer_cost_chart.png (None on error).
        error_message            Human-readable error detail (None on success).
        fields                   Raw extracted field dict (None on hard error).
    """
    import io as _io

    # ── Guard: file must exist and be a supported format ───────────────────
    if not os.path.isfile(image_path):
        return {
            "status_code": "ERROR_FILE_NOT_FOUND",
            "text_summary": None,
            "salary_chart_path": None,
            "loan_chart_path": None,
            "employer_cost_chart_path": None,
            "error_message": (
                "Could not read image, please try again. "
                f"(File not found: {image_path})"
            ),
            "fields": None,
        }

    ext = os.path.splitext(image_path)[1].lower()
    if ext not in (".jpg", ".jpeg", ".png"):
        return {
            "status_code": "ERROR_PIPELINE",
            "text_summary": None,
            "salary_chart_path": None,
            "loan_chart_path": None,
            "employer_cost_chart_path": None,
            "error_message": (
                "Could not read image, please try again. "
                f"(Unsupported format '{ext}' — only JPG and PNG are accepted.)"
            ),
            "fields": None,
        }

    os.makedirs(output_dir, exist_ok=True)
    pie_path      = os.path.abspath(os.path.join(output_dir, "salary_chart.png"))
    bar_path      = os.path.abspath(os.path.join(output_dir, "loan_chart.png"))
    employer_path = os.path.abspath(os.path.join(output_dir, "employer_cost_chart.png"))

    try:
        # ── Step 1: Preprocess ─────────────────────────────────────────────
        binary = preprocess(image_path)

        # ── Step 2: OCR ───────────────────────────────────────────────────
        raw_text = run_ocr(binary)

        if not raw_text or not raw_text.strip():
            return {
                "status_code": "ERROR_NO_TEXT",
                "text_summary": None,
                "salary_chart_path": None,
                "loan_chart_path": None,
                "employer_cost_chart_path": None,
                "error_message": (
                    "No text detected, please send a clearer image. "
                    "Tip: retake the photo in bright, even lighting with "
                    "the payslip lying flat."
                ),
                "fields": None,
            }

        # ── Step 3: Field extraction ───────────────────────────────────────
        fields = extract_fields(raw_text, binary)

        # If neither gross nor net pay could be found, OCR effectively failed
        if fields.get("gross_salary") is None and fields.get("net_pay") is None:
            return {
                "status_code": "ERROR_NO_TEXT",
                "text_summary": None,
                "salary_chart_path": None,
                "loan_chart_path": None,
                "employer_cost_chart_path": None,
                "error_message": (
                    "No text detected, please send a clearer image. "
                    "Could not locate gross salary or net pay on the payslip."
                ),
                "fields": fields,
            }

        # ── Step 4-5: Validation + financials ─────────────────────────────
        validation = run_validation(fields)
        financials = calculate_financials(fields["net_pay"])

        # ── Capture text summary (print_summary writes to stdout) ──────────
        buf         = _io.StringIO()
        _old_stdout = sys.stdout
        sys.stdout  = buf
        try:
            print_summary(fields, validation, financials)
        finally:
            sys.stdout = _old_stdout
        text_summary = buf.getvalue()

        # ── Step 6: Charts (each failure is non-fatal) ─────────────────────
        salary_chart_path        = None
        loan_chart_path          = None
        employer_cost_chart_path = None

        try:
            generate_pie_chart(fields, pie_path)
            salary_chart_path = pie_path
        except Exception as chart_err:
            print(f"[Warning] Salary chart skipped: {chart_err}", file=sys.stderr)

        try:
            generate_bar_chart(fields, financials, bar_path)
            loan_chart_path = bar_path
        except Exception as chart_err:
            print(f"[Warning] Loan chart skipped: {chart_err}", file=sys.stderr)

        try:
            generate_employer_cost_chart(fields, validation, employer_path)
            employer_cost_chart_path = employer_path
        except Exception as chart_err:
            print(f"[Warning] Employer cost chart skipped: {chart_err}",
                  file=sys.stderr)

        return {
            "status_code": "OK",
            "text_summary": text_summary,
            "salary_chart_path": salary_chart_path,
            "loan_chart_path": loan_chart_path,
            "employer_cost_chart_path": employer_cost_chart_path,
            "error_message": None,
            "fields": fields,
        }

    except Exception as exc:
        return {
            "status_code": "ERROR_PIPELINE",
            "text_summary": None,
            "salary_chart_path": None,
            "loan_chart_path": None,
            "employer_cost_chart_path": None,
            "error_message": f"Pipeline error: {exc}",
            "fields": None,
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Malaysian Payslip Analyzer — OpenClaw Phase 2 Pipeline"
    )
    parser.add_argument(
        "image",
        help="Path to the payslip image (JPG or PNG)"
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to save chart PNGs (default: current directory)"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Print raw OCR text so you can see exactly what Tesseract read. "
             "Use this whenever a field shows 'not detected' to diagnose why."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.image):
        sys.exit(f"[ERROR] File not found: {args.image}")

    ext = os.path.splitext(args.image)[1].lower()
    if ext not in (".jpg", ".jpeg", ".png"):
        sys.exit("[ERROR] Only JPG and PNG files are supported.")

    os.makedirs(args.output_dir, exist_ok=True)
    pie_path      = os.path.join(args.output_dir, "salary_chart.png")
    bar_path      = os.path.join(args.output_dir, "loan_chart.png")
    employer_path = os.path.join(args.output_dir, "employer_cost_chart.png")

    # ── Pipeline ──────────────────────────────────────────────────────────
    binary   = preprocess(args.image)

    # Save the preprocessed binary image when debugging.
    # This is the EXACT image Tesseract receives — open it to verify:
    #   ✓  Black text on white background
    #   ✓  All rows visible (EPF, SOCSO, PCB, EIS, Net Pay)
    #   ✗  If mostly black or white → preprocessing parameters need tuning
    #   ✗  If sideways → EXIF rotation not applied (re-save with correct orientation)
    if args.debug:
        debug_img_path = os.path.join(args.output_dir, "debug_preprocessed.png")
        cv2.imwrite(debug_img_path, binary)
        print(f"[Debug] Preprocessed binary image → {debug_img_path}")
        print("[Debug] Open it: it should show crisp black text on a white")
        print("[Debug] background. If it looks wrong, that explains OCR failures.\n")

    raw_text = run_ocr(binary)

    if args.debug:
        print("─" * 62)
        print("  RAW OCR TEXT (--debug mode)")
        print("─" * 62)
        # Number every line so you can cross-reference with the payslip
        for n, line in enumerate(raw_text.splitlines(), start=1):
            print(f"  {n:>3}: {line}")
        print("─" * 62 + "\n")
        print("  TIP: Find your field labels in the lines above.")
        print("  If a label is missing or garbled, the keyword list")
        print("  in extract_fields() needs updating to match.\n")

    fields      = extract_fields(raw_text, binary)
    validation  = run_validation(fields)
    financials  = calculate_financials(fields["net_pay"])

    print_summary(fields, validation, financials)
    generate_pie_chart(fields, pie_path)
    generate_bar_chart(fields, financials, bar_path)
    generate_employer_cost_chart(fields, validation, employer_path)

    print("\n[Done] Analysis complete.")
    print(f"        Salary chart        : {pie_path}")
    print(f"        Loan / DSR chart    : {bar_path}")
    print(f"        Employer cost chart : {employer_path}\n")


if __name__ == "__main__":
    main()
