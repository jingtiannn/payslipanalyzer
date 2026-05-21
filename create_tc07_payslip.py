"""
create_tc07_payslip.py
----------------------
Generates payslip_tc07.png — a test payslip with a deliberately different
layout from TC01-TC04, to verify payslip_analyzer.py handles variety.

KEY DIFFERENCES FROM TC01-TC04:
  1. Three-column table  : Description | Rate | Amount (RM)
                           (TC01-TC04 use two-column: Label | Amount)
  2. BM-first labels     : "Gaji Pokok", "KWSP", "PERKESO", "Gaji Bersih"
                           (TC01-TC04 use English-first labels)
  3. OT rows             : Normal OT + Rest Day OT in the earnings section
  4. Night shift row     : "Elaun Syif Malam / Night Shift Allowance"
  5. Gross label         : "JUMLAH PENDAPATAN / TOTAL INCOME" (not "GROSS SALARY")
  6. Net pay label       : "GAJI BERSIH / NET PAY"
  7. Teal/green header   : Different colour scheme
  8. "–" rate placeholder: Used for flat allowances (no rate column value)

EXPECTED ANALYSIS RESULTS:
  Basic Salary   : RM 2,800.00
  Allowances/OT  : RM 700.00   (transport 200 + meal 150 + night 200 + OT 150)
  Gross          : RM 3,500.00
  EPF (11%)      : RM 385.00   → CORRECT (11%)
  SOCSO          : RM 34.75    → CORRECT (band 3001-3500)
  PCB            : RM 60.00    → user to verify with LHDN
  EIS (0.2%)     : RM 7.00     → CORRECT (0.2% × 3500)
  Net Pay        : RM 3,013.25

EMPLOYER CONTRIBUTIONS (calculated by analyzer):
  EPF employer 13%  : RM 455.00
  SOCSO employer    : RM 61.10   (band 3001-3500)
  EIS employer 0.2% : RM 7.00
  Total employer cost: RM 3,500 + 455 + 61.10 + 7 = RM 4,023.10

Run:
    python create_tc07_payslip.py
    python -X utf8 payslip_analyzer.py payslip_tc07.png

Requires:
    pip install Pillow
"""

from PIL import Image, ImageDraw, ImageFont

# ── Font helper ────────────────────────────────────────────────────────────
def get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "courbd.ttf" if bold else "cour.ttf",
        "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf",
        "LiberationMono-Bold.ttf" if bold else "LiberationMono-Regular.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


# ── Colour palette ─────────────────────────────────────────────────────────
CANVAS_W    = 860
BG          = (255, 255, 255)
FG          = (15,  15,  15)
GRAY        = (110, 110, 110)
RED         = (180,  20,  20)
LINE        = (190, 190, 190)
HDR_BG      = (0,  110,  90)    # teal (different from TC01-04 navy)
HDR_FG      = (255, 255, 255)
HDR_ACCENT  = (160, 255, 210)   # mint green accent
SUB_BG      = (235, 248, 244)   # very light teal for section titles
TEAL        = (0,  130, 105)


def hline(draw, y, x1=30, x2=830, color=LINE, width=1):
    draw.line([(x1, y), (x2, y)], fill=color, width=width)


def three_col(draw, y, col1, col2, col3,
              font=None, color=FG,
              cx1=55, cx2=590, cx3=830) -> int:
    """
    Draw a three-column row:
      col1 left-aligned  at cx1
      col2 right-aligned at cx2   (rate / period column)
      col3 right-aligned at cx3   (amount column)
    """
    if font is None:
        font = get_font(16)
    draw.text((cx1,  y), col1, fill=color, font=font)
    draw.text((cx2,  y), col2, fill=color, font=font, anchor="ra")
    draw.text((cx3,  y), col3, fill=color, font=font, anchor="ra")
    return y + 27


# ── Payslip data ───────────────────────────────────────────────────────────
COMPANY_NAME    = "MEGA DYNAMIC SDN BHD"
COMPANY_ADDR    = "No. 88, Jalan Bukit Bintang, 55100 Kuala Lumpur"
COMPANY_REG     = "Reg. No. 201901043210"
EMP_NAME        = "LEE WEI KANG"
EMP_ID          = "EMP-2024-088"
DEPARTMENT      = "Teknologi Maklumat"
PAY_PERIOD      = "Ogos 2024"
BANK_ACCOUNT    = "MAYBANK ****7821"

BASIC_SALARY    = 2_800.00

INCOME_ROWS = [
    # (BM description,                 Rate label,      Amount)
    ("Gaji Pokok / Basic Salary",      "26 hari",       2_800.00),
    ("Elaun Pengangkutan",             "sebulan",         200.00),
    ("Elaun Makan / Meal Allowance",   "sebulan",         150.00),
    ("Elaun Syif Malam / Night Shift", "10 syif",         200.00),
    ("Kerja Lebih Masa (Normal OT)",   "8 jam",           100.00),
    ("Kerja Lebih Masa (Rest Day OT)", "2 jam",            50.00),
]

GROSS = sum(amt for _, _, amt in INCOME_ROWS)   # 3,500.00

# Statutory deductions (all correct for gross RM 3,500):
EPF   = round(GROSS * 0.11, 2)   # 385.00  (11%)
SOCSO = 34.75                     # band 3001-3500
PCB   = 60.00                     # estimate (single, no children)
EIS   = round(min(GROSS, 4000) * 0.002, 2)   # 7.00  (0.2%)

TOTAL_DEDUCTIONS = round(EPF + SOCSO + PCB + EIS, 2)
NET_PAY          = round(GROSS - TOTAL_DEDUCTIONS, 2)

DEDUCTION_ROWS = [
    # (BM description,                       Rate label,  Amount)
    ("KWSP / EPF (Pekerja 11%)",             "11%",       EPF),
    ("PERKESO / SOCSO",                      "–",         SOCSO),
    ("Cukai Pendapatan / PCB",               "–",         PCB),
    ("SIP / EIS (0.2%)",                     "0.2%",      EIS),
]


# ── Build image ────────────────────────────────────────────────────────────
def build() -> Image.Image:
    f_sm   = get_font(13)
    f_md   = get_font(16)
    f_bold = get_font(16, bold=True)
    f_hdr  = get_font(22, bold=True)
    f_sub  = get_font(14, bold=True)
    f_lg   = get_font(20, bold=True)

    n_rows  = len(INCOME_ROWS) + len(DEDUCTION_ROWS)
    height  = 820 + n_rows * 28
    img     = Image.new("RGB", (CANVAS_W, height), BG)
    draw    = ImageDraw.Draw(img)

    # ════════════════════════════════════════════════
    # HEADER — teal band
    # ════════════════════════════════════════════════
    draw.rectangle([(0, 0), (CANVAS_W, 100)], fill=HDR_BG)
    draw.text((CANVAS_W // 2, 14),  COMPANY_NAME,
              fill=HDR_FG,     font=f_hdr, anchor="mt")
    draw.text((CANVAS_W // 2, 46),  COMPANY_ADDR,
              fill=HDR_ACCENT, font=f_sm,  anchor="mt")
    draw.text((CANVAS_W // 2, 64),  COMPANY_REG,
              fill=HDR_ACCENT, font=f_sm,  anchor="mt")
    draw.text((CANVAS_W // 2, 80),  "PENYATA GAJI / PAYSLIP",
              fill=(255, 230, 150), font=f_bold, anchor="mt")

    y = 115

    # ════════════════════════════════════════════════
    # EMPLOYEE INFO block
    # ════════════════════════════════════════════════
    draw.rectangle([(30, y), (830, y + 90)],
                   fill=SUB_BG, outline=LINE, width=1)
    y2 = y + 12
    draw.text((50,  y2), f"Nama / Name       : {EMP_NAME}",    fill=FG,   font=f_md)
    draw.text((490, y2), f"Tempoh Gaji : {PAY_PERIOD}",        fill=FG,   font=f_md)
    y2 += 27
    draw.text((50,  y2), f"No. Pekerja       : {EMP_ID}",      fill=FG,   font=f_md)
    draw.text((490, y2), f"Bahagian    : {DEPARTMENT}",        fill=FG,   font=f_md)
    y2 += 27
    draw.text((50,  y2), f"Akaun Bank        : {BANK_ACCOUNT}",fill=GRAY, font=f_sm)
    y = y + 106

    # ════════════════════════════════════════════════
    # EARNINGS TABLE
    # ════════════════════════════════════════════════
    draw.rectangle([(30, y), (830, y + 26)], fill=TEAL)
    draw.text((55, y + 4), "PENDAPATAN / INCOME",
              fill=HDR_FG, font=f_sub)
    y += 30

    # Column headers
    hline(draw, y)
    y += 6
    draw.text((55,  y), "Penerangan / Description", fill=GRAY, font=f_sm)
    draw.text((590, y), "Kadar / Rate",             fill=GRAY, font=f_sm, anchor="ra")
    draw.text((830, y), "Amaun (RM)",               fill=GRAY, font=f_sm, anchor="ra")
    y += 20
    hline(draw, y, width=2)
    y += 8

    for desc, rate, amount in INCOME_ROWS:
        clr = (40, 40, 40) if "Gaji Pokok" in desc else FG
        y = three_col(draw, y, desc, rate, f"{amount:,.2f}", font=f_md, color=clr)

    hline(draw, y + 2)
    y += 12

    # TOTAL INCOME row (bold, teal text)
    y = three_col(draw, y,
                  "JUMLAH PENDAPATAN / TOTAL INCOME",
                  "",
                  f"{GROSS:,.2f}",
                  font=f_bold, color=TEAL)
    y += 14

    # ════════════════════════════════════════════════
    # DEDUCTIONS TABLE
    # ════════════════════════════════════════════════
    draw.rectangle([(30, y), (830, y + 26)], fill=(160, 30, 30))
    draw.text((55, y + 4), "POTONGAN / DEDUCTIONS",
              fill=HDR_FG, font=f_sub)
    y += 30

    hline(draw, y)
    y += 6
    draw.text((55,  y), "Penerangan / Description", fill=GRAY, font=f_sm)
    draw.text((590, y), "Kadar / Rate",             fill=GRAY, font=f_sm, anchor="ra")
    draw.text((830, y), "Amaun (RM)",               fill=GRAY, font=f_sm, anchor="ra")
    y += 20
    hline(draw, y, width=2)
    y += 8

    for desc, rate, amount in DEDUCTION_ROWS:
        y = three_col(draw, y, desc, rate, f"{amount:,.2f}", font=f_md)

    hline(draw, y + 2)
    y += 12

    y = three_col(draw, y,
                  "JUMLAH POTONGAN / TOTAL DEDUCTIONS",
                  "",
                  f"{TOTAL_DEDUCTIONS:,.2f}",
                  font=f_bold, color=RED)
    y += 14

    # ════════════════════════════════════════════════
    # NET PAY box
    # ════════════════════════════════════════════════
    box_y = y
    draw.rectangle([(30, box_y), (830, box_y + 60)],
                   fill=HDR_BG, outline=(0, 80, 65), width=2)
    draw.text((55, box_y + 16), "GAJI BERSIH / NET PAY",
              fill=HDR_ACCENT, font=f_bold)
    draw.text((820, box_y + 10), f"RM  {NET_PAY:,.2f}",
              fill=(255, 230, 150), font=f_lg, anchor="ra")
    y = box_y + 76

    # ════════════════════════════════════════════════
    # SUMMARY BOX (employee + employer overview)
    # ════════════════════════════════════════════════
    y += 10
    draw.rectangle([(30, y), (830, y + 110)],
                   fill=(245, 252, 248), outline=LINE, width=1)
    y2 = y + 10
    draw.text((55, y2), "Ringkasan Caruman / Contribution Summary",
              fill=TEAL, font=f_sub)
    y2 += 24
    hline(draw, y2, x1=40, x2=820, color=(210, 230, 220))
    y2 += 8

    summary_rows = [
        ("Caruman KWSP Pekerja / EPF Employee 11%",
         f"RM {EPF:,.2f}"),
        ("Caruman KWSP Majikan / EPF Employer 13%",
         f"RM {round(GROSS * 0.13, 2):,.2f}"),
        ("Caruman PERKESO Pekerja / SOCSO Employee",
         f"RM {SOCSO:,.2f}"),
        ("Caruman PERKESO Majikan / SOCSO Employer",
         f"RM 61.10"),
        ("Caruman SIP Pekerja & Majikan / EIS Employee & Employer",
         f"RM {EIS:,.2f} each"),
    ]
    for label, val in summary_rows:
        draw.text((55,  y2), label, fill=GRAY, font=f_sm)
        draw.text((820, y2), val,   fill=FG,   font=f_sm, anchor="ra")
        y2 += 18

    y = y + 126

    # ════════════════════════════════════════════════
    # FOOTER
    # ════════════════════════════════════════════════
    hline(draw, y + 6)
    y += 20
    footer = [
        "Penyata gaji ini dijana secara komputer. / This payslip is computer-generated.",
        "KWSP — Akta KWSP 1991  |  PERKESO — Akta PERKESO 1969  |  SIP — Akta SIP 2017",
        "*** UNTUK TUJUAN UJIAN SAHAJA — SEMUA DATA ADALAH REKAAN / FOR TESTING ONLY ***",
    ]
    for line in footer:
        clr = RED if "REKAAN" in line else GRAY
        draw.text((CANVAS_W // 2, y), line,
                  fill=clr, font=f_sm, anchor="mt")
        y += 20

    # Crop to actual content
    return img.crop((0, 0, CANVAS_W, y + 20))


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    out = "payslip_tc07.png"
    img = build()
    img.save(out, format="PNG", dpi=(150, 150))

    print(f"[OK] {out} saved.")
    print()
    print("TC07 expected values:")
    print(f"  Basic Salary   : RM {BASIC_SALARY:,.2f}")
    print(f"  Gross          : RM {GROSS:,.2f}")
    print(f"  EPF   (11%)    : RM {EPF:,.2f}")
    print(f"  SOCSO          : RM {SOCSO:,.2f}  (band 3001-3500)")
    print(f"  PCB            : RM {PCB:,.2f}")
    print(f"  EIS   (0.2%)   : RM {EIS:,.2f}")
    print(f"  Net Pay        : RM {NET_PAY:,.2f}")
    print()
    print("Run the analyzer:")
    print(f"  python -X utf8 payslip_analyzer.py {out}")
