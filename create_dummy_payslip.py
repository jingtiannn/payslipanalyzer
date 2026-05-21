"""
create_dummy_payslip.py
-----------------------
Generates realistic-looking FAKE Malaysian payslip images for testing.
All names, figures, and company details are completely fictitious.

Produces three test payslips:
  payslip_tc01.png  — Fresh grad, RM 2,500 basic, no allowances
  payslip_tc02.png  — Mid-level,  RM 4,000 basic, with allowances
  payslip_tc03.png  — Blurry version of TC01 (tests blur detection)

Run:
    python create_dummy_payslip.py

Requires:
    pip install Pillow
"""

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os

# ── Try to load a clean monospace font, fall back to PIL default ───────────
def get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """
    Try Courier New (Windows) → DejaVu Mono (Linux) → PIL default.
    Monospace fonts produce cleaner column alignment, making OCR easier.
    """
    candidates = [
        "courbd.ttf" if bold else "cour.ttf",           # Courier New — Windows
        "DejaVuSansMono-Bold.ttf" if bold
            else "DejaVuSansMono.ttf",                  # DejaVu — Linux
        "LiberationMono-Bold.ttf" if bold
            else "LiberationMono-Regular.ttf",          # Liberation — Ubuntu
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    # Last resort: PIL built-in bitmap font (no size control)
    return ImageFont.load_default()


# ── Canvas helpers ─────────────────────────────────────────────────────────
CANVAS_W  = 800
BG_COLOR  = (255, 255, 255)   # white background
FG_COLOR  = (10,  10,  10)    # near-black text
GRAY_COLOR= (130, 130, 130)   # muted labels
RED_COLOR = (180,  20,  20)   # warning accents
LINE_COLOR= (180, 180, 180)   # separator lines

FONT_SM   = 14
FONT_MD   = 16
FONT_LG   = 20
FONT_HDR  = 22


def new_canvas(height: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img  = Image.new("RGB", (CANVAS_W, height), BG_COLOR)
    draw = ImageDraw.Draw(img)
    return img, draw


def hline(draw: ImageDraw.ImageDraw, y: int,
          x1: int = 30, x2: int = 770, color=LINE_COLOR, width: int = 1):
    draw.line([(x1, y), (x2, y)], fill=color, width=width)


def row(draw: ImageDraw.ImageDraw, y: int,
        label: str, value: str,
        lx: int = 50, rx: int = 750,
        font=None, color=FG_COLOR) -> int:
    """Draw a label on the left and a value right-aligned. Returns next y."""
    if font is None:
        font = get_font(FONT_MD)
    draw.text((lx, y), label, fill=color, font=font)
    draw.text((rx, y), value, fill=color, font=font, anchor="ra")
    return y + 26


def centered(draw: ImageDraw.ImageDraw, y: int, text: str,
             font=None, color=FG_COLOR) -> int:
    if font is None:
        font = get_font(FONT_MD)
    draw.text((CANVAS_W // 2, y), text, fill=color, font=font, anchor="mt")
    return y + 28


# ── Payslip renderer ───────────────────────────────────────────────────────
def build_payslip(
    company_name:    str,
    company_address: str,
    emp_name:        str,
    emp_id:          str,
    department:      str,
    pay_period:      str,
    basic_salary:    float,
    allowances:      dict,       # {"label": amount, ...}
    epf:             float,
    socso:           float,
    pcb:             float,
    eis:             float,
) -> Image.Image:
    """
    Draws a full payslip image with sections:
        Header → Employee Info → Earnings → Deductions → Net Pay → Footer
    All label strings exactly match the keywords in payslip_analyzer.py so
    that OCR field extraction works reliably on these dummy images.
    """

    # ── Pre-compute totals ─────────────────────────────────────────────────
    allowances_total  = sum(allowances.values())
    gross_salary      = basic_salary + allowances_total
    total_deductions  = epf + socso + pcb + eis
    net_pay           = gross_salary - total_deductions

    # ── Fonts ──────────────────────────────────────────────────────────────
    f_sm   = get_font(FONT_SM)
    f_md   = get_font(FONT_MD)
    f_bold = get_font(FONT_MD,  bold=True)
    f_hdr  = get_font(FONT_HDR, bold=True)
    f_lg   = get_font(FONT_LG,  bold=True)

    # ── Estimate canvas height dynamically ────────────────────────────────
    n_allowance_rows = max(len(allowances), 1)
    canvas_height = 720 + n_allowance_rows * 28
    img, draw = new_canvas(canvas_height)

    y = 20

    # ═══════════════════════════════════════
    # HEADER — company info
    # ═══════════════════════════════════════
    draw.rectangle([(0, 0), (CANVAS_W, 90)], fill=(30, 60, 114))
    draw.text((CANVAS_W // 2, 18), company_name,
              fill=(255, 255, 255), font=f_hdr, anchor="mt")
    draw.text((CANVAS_W // 2, 48), company_address,
              fill=(200, 215, 255), font=f_sm,  anchor="mt")
    draw.text((CANVAS_W // 2, 68), "PAYSLIP / KENYATAAN GAJI",
              fill=(255, 220, 100), font=f_bold, anchor="mt")
    y = 105

    # ═══════════════════════════════════════
    # EMPLOYEE DETAILS
    # ═══════════════════════════════════════
    draw.rectangle([(30, y), (770, y + 80)], fill=(245, 247, 252),
                   outline=LINE_COLOR, width=1)
    y += 10
    draw.text((50,  y), f"Employee Name : {emp_name}",   fill=FG_COLOR, font=f_md)
    draw.text((430, y), f"Pay Period : {pay_period}",    fill=FG_COLOR, font=f_md)
    y += 26
    draw.text((50,  y), f"Employee ID   : {emp_id}",     fill=FG_COLOR, font=f_md)
    draw.text((430, y), f"Department : {department}",    fill=FG_COLOR, font=f_md)
    y += 40

    # ═══════════════════════════════════════
    # EARNINGS TABLE
    # ═══════════════════════════════════════
    draw.text((50, y), "EARNINGS / PENDAPATAN",
              fill=(30, 60, 114), font=f_bold)
    y += 24
    hline(draw, y, width=2)
    y += 8

    # Column headers
    draw.text((50,  y), "Description",   fill=GRAY_COLOR, font=f_sm)
    draw.text((750, y), "Amount (RM)",   fill=GRAY_COLOR, font=f_sm, anchor="ra")
    y += 22
    hline(draw, y)
    y += 8

    y = row(draw, y, "Basic Salary / Gaji Pokok",
            f"{basic_salary:>10,.2f}", font=f_md)

    for label, amount in allowances.items():
        y = row(draw, y, f"  {label}", f"{amount:>10,.2f}", font=f_md,
                color=(60, 60, 60))

    hline(draw, y + 4)
    y += 14
    y = row(draw, y, "GROSS SALARY / Gaji Kasar",
            f"{gross_salary:>10,.2f}", font=f_bold)
    y += 10

    # ═══════════════════════════════════════
    # DEDUCTIONS TABLE
    # ═══════════════════════════════════════
    draw.text((50, y), "DEDUCTIONS / POTONGAN",
              fill=(180, 30, 30), font=f_bold)
    y += 24
    hline(draw, y, width=2)
    y += 8

    draw.text((50,  y), "Description",   fill=GRAY_COLOR, font=f_sm)
    draw.text((750, y), "Amount (RM)",   fill=GRAY_COLOR, font=f_sm, anchor="ra")
    y += 22
    hline(draw, y)
    y += 8

    # Each label deliberately uses the same keywords that payslip_analyzer.py
    # searches for, so OCR extraction matches correctly.
    deduction_rows = [
        ("EPF / KWSP (Employee 11%)",           epf),
        ("SOCSO / PERKESO",                     socso),
        ("PCB / MTD (Cukai Pendapatan)",        pcb),
        ("EIS / SIP (0.2%)",                    eis),
    ]
    for label, amount in deduction_rows:
        y = row(draw, y, label, f"{amount:>10,.2f}", font=f_md)

    hline(draw, y + 4)
    y += 14
    y = row(draw, y, "Total Deductions / Jumlah Potongan",
            f"{total_deductions:>10,.2f}", font=f_bold, color=RED_COLOR)
    y += 10

    # ═══════════════════════════════════════
    # NET PAY BOX
    # ═══════════════════════════════════════
    box_top = y
    draw.rectangle([(30, box_top), (770, box_top + 56)],
                   fill=(30, 60, 114), outline=(20, 40, 90), width=2)
    draw.text((55,  box_top + 14), "NET PAY / Gaji Bersih",
              fill=(200, 215, 255), font=f_bold)
    draw.text((750, box_top + 10), f"RM  {net_pay:,.2f}",
              fill=(255, 220, 100), font=f_lg, anchor="ra")
    y = box_top + 70

    # ═══════════════════════════════════════
    # FOOTER — statutory notice
    # ═══════════════════════════════════════
    hline(draw, y + 6)
    y += 18
    footer_lines = [
        "This payslip is computer-generated. No signature required.",
        "EPF contributions are made under the Employees Provident Fund Act 1991.",
        "SOCSO contributions are made under the Employees Social Security Act 1969.",
        "EIS contributions are made under the Employment Insurance System Act 2017.",
        "*** FOR TESTING PURPOSES ONLY — ALL DATA IS FICTITIOUS ***",
    ]
    for line in footer_lines:
        color = RED_COLOR if "FICTITIOUS" in line else GRAY_COLOR
        draw.text((CANVAS_W // 2, y), line, fill=color, font=f_sm, anchor="mt")
        y += 20

    return img


# ── Test case definitions ──────────────────────────────────────────────────

TEST_CASES = [
    {
        "filename":        "payslip_tc01.png",
        "label":           "TC01 — Fresh graduate, RM 2,500, no allowances",
        "company_name":    "SYARIKAT TEKNOLOGI MAJU SDN BHD",
        "company_address": "No. 12, Jalan Teknologi 3, Taman Sains Selangor, 47810 Petaling Jaya",
        "emp_name":        "MUHAMMAD AMIR BIN ZAINAL",
        "emp_id":          "EMP-2025-001",
        "department":      "Software Engineering",
        "pay_period":      "April 2025",
        "basic_salary":    2500.00,
        "allowances":      {},
        # Correct statutory values (PERKESO Second Schedule):
        #   EPF   = 2500 x 11%           = 275.00
        #   SOCSO = band RM2401-2500     =  12.25  (NOT 24.75 — that is for RM4900-5000)
        #   PCB   = estimate             =  28.00  (single, no children)
        #   EIS   = 2500 x 0.2%         =   5.00
        "epf":             275.00,
        "socso":            12.25,
        "pcb":              28.00,
        "eis":               5.00,
    },
    {
        "filename":        "payslip_tc02.png",
        "label":           "TC02 — Mid-level, RM 4,000 + allowances",
        "company_name":    "PERDANA INOVASI BERHAD",
        "company_address": "Level 18, Menara Perdana, Jalan Ampang, 50450 Kuala Lumpur",
        "emp_name":        "NURUL AINA BINTI HASSAN",
        "emp_id":          "EMP-2024-047",
        "department":      "Product Management",
        "pay_period":      "April 2025",
        "basic_salary":    4000.00,
        "allowances":      {
            "Transport Allowance / Elaun Pengangkutan": 300.00,
            "Housing Allowance / Elaun Perumahan":      200.00,
        },
        # Gross = 4500
        #   EPF   = 4500 x 11%           = 495.00
        #   SOCSO = band RM4401-4500     =  22.25  (PERKESO cap is RM5000, not RM4000)
        #   PCB   = estimate             = 120.00
        #   EIS   = min(4500,4000)x0.2%  =   8.00  (EIS insured salary capped at RM4000)
        "epf":             495.00,
        "socso":            22.25,
        "pcb":             120.00,
        "eis":               8.00,
    },
    {
        "filename":        "payslip_tc03_warning.png",
        "label":           "TC03 — WARNING: EPF underdeducted (8% instead of 11%)",
        "company_name":    "BETA RESOURCES SDN BHD",
        "company_address": "Lot 5, Jalan Industri 7, Kawasan Perindustrian Nilai, Negeri Sembilan",
        "emp_name":        "RAJESH A/L KUMAR",
        "emp_id":          "EMP-2023-112",
        "department":      "Operations",
        "pay_period":      "April 2025",
        "basic_salary":    3200.00,
        "allowances":      {
            "Meal Allowance / Elaun Makan": 150.00,
        },
        # Gross = 3350
        #   EPF   deliberately wrong: 3350 x 8% = 268.00  (should be 368.50)
        #   SOCSO = correct band RM3301-3400     =  16.75
        #   PCB   = estimate                     =  80.00
        #   EIS   = 3350 x 0.2%                 =   6.70
        "epf":             268.00,   # <- intentionally wrong (8% not 11%)
        "socso":            16.75,
        "pcb":              80.00,
        "eis":               6.70,
    },
]


def make_blurry(img: Image.Image, passes: int = 6) -> Image.Image:
    """Apply repeated Gaussian blur to simulate a shaky camera photo."""
    for _ in range(passes):
        img = img.filter(ImageFilter.GaussianBlur(radius=3))
    return img


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("Generating dummy payslip images...\n")
    generated = []

    for tc in TEST_CASES:
        img = build_payslip(
            company_name    = tc["company_name"],
            company_address = tc["company_address"],
            emp_name        = tc["emp_name"],
            emp_id          = tc["emp_id"],
            department      = tc["department"],
            pay_period      = tc["pay_period"],
            basic_salary    = tc["basic_salary"],
            allowances      = tc["allowances"],
            epf             = tc["epf"],
            socso           = tc["socso"],
            pcb             = tc["pcb"],
            eis             = tc["eis"],
        )

        img.save(tc["filename"], format="PNG", dpi=(150, 150))
        generated.append(tc["filename"])
        print(f"  [OK] {tc['filename']:35s} — {tc['label']}")

    # ── TC04: blurry image (should trigger WARN_BLURRY_IMAGE in analyzer) ─
    blurry_src = TEST_CASES[0]
    blurry_img = build_payslip(
        company_name    = blurry_src["company_name"],
        company_address = blurry_src["company_address"],
        emp_name        = blurry_src["emp_name"],
        emp_id          = blurry_src["emp_id"],
        department      = blurry_src["department"],
        pay_period      = blurry_src["pay_period"],
        basic_salary    = blurry_src["basic_salary"],
        allowances      = blurry_src["allowances"],
        epf             = blurry_src["epf"],
        socso           = blurry_src["socso"],
        pcb             = blurry_src["pcb"],
        eis             = blurry_src["eis"],
    )
    blurry_img = make_blurry(blurry_img, passes=8)
    blurry_img.save("payslip_tc04_blurry.png", format="PNG")
    generated.append("payslip_tc04_blurry.png")
    print(f"  [OK] {'payslip_tc04_blurry.png':35s} — TC04 — Blurry (expect blur warning)")

    print(f"\nAll {len(generated)} dummy payslips saved to current directory.")
    print("\nNow run the analyzer on each one:\n")
    for f in generated:
        print(f"    python payslip_analyzer.py {f}")

    print()
    print("What each test case checks:")
    print("  TC01 — All deductions correct, fresh grad salary")
    print("  TC02 — Correct deductions with allowances, mid-level salary")
    print("  TC03 — EPF WARNING (8% used instead of 11%)")
    print("  TC04 — BLUR WARNING (pipeline should halt before OCR)")


if __name__ == "__main__":
    main()
