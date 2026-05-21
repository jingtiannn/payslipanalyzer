"""
create_tc08_payslip.py
----------------------
Generates payslip_tc08.png — a TRUE TWO-COLUMN layout where EARNINGS and
DEDUCTIONS sit side-by-side on the same horizontal rows.

This is the HARDEST layout for OCR because Tesseract reads left-to-right,
so "BASIC SALARY   2,200.00   |   EPF / KWSP   242.00" becomes one merged
line. The analyzer must use the spatial bounding-box fallback (PSM 11 +
right-of-keyword search) to correctly separate both values.

LAYOUT:
  ┌────────────────────────┬────────────────────────┐
  │ EARNINGS               │ DEDUCTIONS             │
  ├────────────────────────┼────────────────────────┤
  │ Basic Salary  2,200.00 │ EPF / KWSP     242.00  │
  │ Transport Allow  200.00│ SOCSO           29.75  │
  │ Housing Allow    300.00│ PCB / MTD        30.00  │
  │ Meal Allowance   200.00│ EIS / SIP         5.40  │
  │                        │                        │
  │ TOTAL INCOME  2,900.00 │ TOTAL DEDUCTIONS 307.15 │
  └────────────────────────┴────────────────────────┘
  ┌──────────────────────────────────────────────────┐
  │  NET PAY / GAJI BERSIH              RM 2,592.85 │
  └──────────────────────────────────────────────────┘

EXPECTED ANALYSIS RESULTS:
  Basic Salary    : RM 2,200.00
  Allowances / OT : RM 700.00   (transport 200 + housing 300 + meal 200)
  Gross           : RM 2,900.00
  EPF   (11%)     : RM 319.00   → CORRECT  (11%)
  SOCSO           : RM 24.75    → CORRECT  (band 2001-2500... wait, 2900 is band 2501-3000)
  PCB             : RM 30.00
  EIS   (0.2%)    : RM 5.80     → CORRECT  (0.2% × 2900)
  Net Pay         : RM 2,520.45

EMPLOYER CONTRIBUTIONS:
  EPF employer 13%  : RM 377.00
  SOCSO employer    : RM 52.35   (band 2501-3000)
  EIS employer 0.2% : RM 5.80
  Total cost        : RM 2,900 + 377 + 52.35 + 5.80 = RM 3,335.15

Run:
    python create_tc08_payslip.py
    python -X utf8 payslip_analyzer.py payslip_tc08.png
"""

from PIL import Image, ImageDraw, ImageFont

def get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    for name in (
        ("courbd.ttf" if bold else "cour.ttf"),
        ("DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"),
        ("LiberationMono-Bold.ttf" if bold else "LiberationMono-Regular.ttf"),
    ):
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            pass
    return ImageFont.load_default()


# ── Colours ────────────────────────────────────────────────────────────────
W          = 900
BG         = (255, 255, 255)
FG         = (20,  20,  20)
GRAY       = (120, 120, 120)
RED        = (170,  25,  25)
LINE       = (180, 180, 180)
HDR_BG     = (45,  85, 150)    # corporate blue
HDR_FG     = (255, 255, 255)
HDR_ACC    = (210, 225, 255)
EARN_BG    = (230, 240, 255)   # light blue tint — earnings side
DED_BG     = (255, 232, 232)   # light red tint  — deductions side
DIVIDER    = (100, 130, 190)   # vertical divider colour
NET_BG     = (40,  75, 140)

# Column boundaries
LX1 = 30   # left edge
MID = 460  # centre divider x
RX2 = 870  # right edge
ROW_H = 26 # row height
PAD = 10   # inner horizontal padding

# ── Payslip data ───────────────────────────────────────────────────────────
COMPANY    = "VISTA PRIMA SDN BHD"
ADDR       = "Wisma Vista, Jalan Semantan, Damansara Heights, 50490 Kuala Lumpur"
EMP_NAME   = "SITI NURHALIZA BINTI RAMLI"
EMP_ID     = "VP-2023-055"
DEPT       = "Human Resources"
PERIOD     = "September 2024"
IC         = "980321-14-5678"

BASIC      = 2_200.00
ALLOWANCES = [
    ("Transport Allowance",    200.00),
    ("Housing Allowance",      300.00),
    ("Meal Allowance",         200.00),
]
GROSS   = BASIC + sum(a for _, a in ALLOWANCES)   # 2,900.00
EPF     = round(GROSS * 0.11, 2)                  # 319.00
SOCSO   = 29.75                                    # band 2501-3000
PCB     = 30.00
EIS     = round(min(GROSS, 4000) * 0.002, 2)       # 5.80
TOTAL_D = round(EPF + SOCSO + PCB + EIS, 2)
NET     = round(GROSS - TOTAL_D, 2)


def hline(draw, y, x1=LX1, x2=RX2, color=LINE, width=1):
    draw.line([(x1, y), (x2, y)], fill=color, width=width)

def vline(draw, x, y1, y2, color=DIVIDER, width=2):
    draw.line([(x, y1), (x, y2)], fill=color, width=width)

def ltext(draw, x, y, text, font, color=FG):
    draw.text((x, y), text, fill=color, font=font)

def rtext(draw, x, y, text, font, color=FG):
    draw.text((x, y), text, fill=color, font=font, anchor="ra")


def two_col_row(draw, y,
                left_label, left_val,
                right_label, right_val,
                font, lcolor=FG, rcolor=FG,
                left_val_x=MID - PAD - 5,
                right_val_x=RX2 - PAD) -> int:
    """
    Draw one row spanning both columns:
      left side:  label left-aligned, value right-aligned before MID
      right side: label left-aligned after MID, value right-aligned at RX2
    """
    ltext(draw, LX1 + PAD, y, left_label,  font, lcolor)
    if left_val:
        rtext(draw, left_val_x,   y, left_val,  font, lcolor)
    ltext(draw, MID + PAD,  y, right_label, font, rcolor)
    if right_val:
        rtext(draw, right_val_x,  y, right_val, font, rcolor)
    return y + ROW_H


def build() -> Image.Image:
    f_sm   = get_font(13)
    f_md   = get_font(15)
    f_bold = get_font(15, bold=True)
    f_hdr  = get_font(21, bold=True)
    f_lg   = get_font(19, bold=True)

    # Canvas — estimate height
    earn_rows = 1 + len(ALLOWANCES)   # basic + allowances
    ded_rows  = 4                      # EPF, SOCSO, PCB, EIS
    max_rows  = max(earn_rows, ded_rows)
    H = 680 + max_rows * ROW_H
    img  = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # ════════════════════════════════════════════════
    # HEADER
    # ════════════════════════════════════════════════
    draw.rectangle([(0, 0), (W, 95)], fill=HDR_BG)
    draw.text((W//2, 12),  COMPANY,
              fill=HDR_FG, font=f_hdr, anchor="mt")
    draw.text((W//2, 44),  ADDR,
              fill=HDR_ACC, font=f_sm, anchor="mt")
    draw.text((W//2, 64),
              "SALARY STATEMENT / KENYATAAN GAJI",
              fill=(255, 220, 100), font=f_bold, anchor="mt")
    draw.text((W//2, 80), f"Pay Period: {PERIOD}",
              fill=HDR_ACC, font=f_sm, anchor="mt")

    y = 108

    # ════════════════════════════════════════════════
    # EMPLOYEE INFO (two-up)
    # ════════════════════════════════════════════════
    draw.rectangle([(LX1, y), (RX2, y+80)],
                   fill=(245, 247, 253), outline=LINE)
    y2 = y + 10
    ltext(draw, 50,  y2, f"Employee Name : {EMP_NAME}",  f_md)
    ltext(draw, 520, y2, f"Department   : {DEPT}",       f_md)
    y2 += 26
    ltext(draw, 50,  y2, f"Employee ID   : {EMP_ID}",    f_md)
    ltext(draw, 520, y2, f"IC Number    : {IC}",         f_md)
    y += 92

    # ════════════════════════════════════════════════
    # TWO-COLUMN TABLE HEADER ROW
    # ════════════════════════════════════════════════
    draw.rectangle([(LX1, y), (MID, y+28)], fill=HDR_BG)
    draw.rectangle([(MID, y), (RX2, y+28)], fill=(150, 30, 30))
    draw.text(((LX1+MID)//2, y+4), "EARNINGS / PENDAPATAN",
              fill=HDR_FG,          font=f_bold, anchor="mt")
    draw.text(((MID+RX2)//2, y+4), "DEDUCTIONS / POTONGAN",
              fill=HDR_FG,          font=f_bold, anchor="mt")
    y += 32

    # Sub-header row (Description | Amount || Description | Amount)
    draw.rectangle([(LX1, y), (RX2, y+22)],
                   fill=(230, 235, 250))
    ltext(draw, LX1+PAD,        y+3, "Description",  f_sm, GRAY)
    rtext(draw, MID-PAD-5,      y+3, "Amount (RM)",  f_sm, GRAY)
    ltext(draw, MID+PAD,        y+3, "Description",  f_sm, GRAY)
    rtext(draw, RX2-PAD,        y+3, "Amount (RM)",  f_sm, GRAY)
    y += 24
    hline(draw, y, width=2)
    y += 6

    # ── Fill earnings & deductions in parallel rows ────────────────────────
    earn_items = [("Basic Salary",       BASIC)] + list(ALLOWANCES)
    ded_items  = [
        ("EPF / KWSP (Employee 11%)",   EPF),
        ("SOCSO / PERKESO",             SOCSO),
        ("PCB / MTD",                   PCB),
        ("EIS / SIP (0.2%)",            EIS),
    ]

    table_top = y
    n_rows    = max(len(earn_items), len(ded_items))
    for i in range(n_rows):
        row_y = y + i * ROW_H
        # Plain white rows — coloured tints create OCR noise after
        # adaptive thresholding, corrupting right-aligned numbers.
        draw.rectangle([(LX1, row_y), (RX2, row_y+ROW_H)], fill=BG)

        el = earn_items[i] if i < len(earn_items) else ("", None)
        dl = ded_items[i]  if i < len(ded_items)  else ("", None)

        lbl_e, amt_e = el
        lbl_d, amt_d = dl

        ltext(draw, LX1+PAD, row_y+5, lbl_e, f_md)
        if amt_e is not None:
            rtext(draw, MID-PAD-5, row_y+5, f"{amt_e:,.2f}", f_md)

        ltext(draw, MID+PAD, row_y+5, lbl_d, f_md)
        if amt_d is not None:
            rtext(draw, RX2-PAD, row_y+5, f"{amt_d:,.2f}", f_md)

    y += n_rows * ROW_H
    table_bottom = y

    # Draw vertical centre divider over full row area
    vline(draw, MID, table_top, table_bottom)

    # ── Totals row ──────────────────────────────────────────────────────────
    hline(draw, y, width=2)
    y += 6
    draw.rectangle([(LX1, y), (MID, y+ROW_H+4)], fill=(210, 225, 255))
    draw.rectangle([(MID, y), (RX2, y+ROW_H+4)], fill=(255, 210, 210))
    vline(draw, MID, y, y+ROW_H+4)

    ltext(draw, LX1+PAD,   y+5, "TOTAL INCOME",      f_bold, HDR_BG)
    rtext(draw, MID-PAD-5, y+5, f"{GROSS:,.2f}",     f_bold, HDR_BG)
    ltext(draw, MID+PAD,   y+5, "TOTAL DEDUCTIONS",  f_bold, RED)
    rtext(draw, RX2-PAD,   y+5, f"{TOTAL_D:,.2f}",   f_bold, RED)
    y += ROW_H + 14

    # ════════════════════════════════════════════════
    # NET PAY box — full width
    # ════════════════════════════════════════════════
    draw.rectangle([(LX1, y), (RX2, y+58)],
                   fill=NET_BG, outline=(30, 60, 120), width=2)
    ltext(draw, LX1+16, y+16, "NET PAY / GAJI BERSIH",
          f_bold, (190, 210, 255))
    rtext(draw, RX2-16, y+10, f"RM  {NET:,.2f}",
          f_lg, (255, 225, 100))
    y += 72

    # ════════════════════════════════════════════════
    # EMPLOYER CONTRIBUTIONS mini-table
    # ════════════════════════════════════════════════
    y += 8
    draw.rectangle([(LX1, y), (RX2, y+24)], fill=(50, 90, 160))
    draw.text((W//2, y+3), "EMPLOYER STATUTORY CONTRIBUTIONS",
              fill=HDR_FG, font=f_bold, anchor="mt")
    y += 28

    empr_epf   = round(GROSS * 0.13, 2)
    empr_socso = 52.35   # band 2501-3000
    empr_eis   = round(min(GROSS, 4000) * 0.002, 2)
    total_empr = round(empr_epf + empr_socso + empr_eis, 2)
    total_cost = round(GROSS + total_empr, 2)

    empr_rows = [
        ("EPF Employer Contribution (13%)",  f"RM {empr_epf:,.2f}"),
        ("SOCSO Employer Contribution",      f"RM {empr_socso:,.2f}"),
        ("EIS Employer Contribution (0.2%)", f"RM {empr_eis:,.2f}"),
        ("Total Employer Contribution",      f"RM {total_empr:,.2f}"),
        ("TOTAL COST TO EMPLOYER",           f"RM {total_cost:,.2f}"),
    ]
    for i, (lbl, val) in enumerate(empr_rows):
        bg = (232, 238, 255) if i % 2 == 0 else (242, 246, 255)
        if "TOTAL COST" in lbl:
            bg = (210, 220, 255)
        draw.rectangle([(LX1, y), (RX2, y+ROW_H)], fill=bg)
        col = (40, 70, 150) if "TOTAL" in lbl else GRAY
        ltext(draw, LX1+PAD, y+5, lbl, f_md if "TOTAL COST" in lbl else f_sm, col)
        rtext(draw, RX2-PAD, y+5, val, f_bold if "TOTAL COST" in lbl else f_sm, col)
        y += ROW_H

    # ════════════════════════════════════════════════
    # FOOTER
    # ════════════════════════════════════════════════
    y += 10
    hline(draw, y)
    y += 12
    for line in [
        "This is a computer-generated payslip. No signature required.",
        "EPF Act 1991  |  PERKESO Act 1969  |  EIS Act 2017  |  Income Tax Act 1967",
        "*** FOR TESTING PURPOSES ONLY — ALL DETAILS ARE FICTITIOUS ***",
    ]:
        clr = RED if "FICTITIOUS" in line else GRAY
        draw.text((W//2, y), line, fill=clr, font=f_sm, anchor="mt")
        y += 19

    return img.crop((0, 0, W, y + 15))


if __name__ == "__main__":
    out = "payslip_tc08.png"
    build().save(out, format="PNG", dpi=(150, 150))
    print(f"[OK] {out} saved.")
    print()
    print("TC08 — Two-column layout expected values:")
    print(f"  Basic Salary  : RM {BASIC:,.2f}")
    print(f"  Gross         : RM {GROSS:,.2f}")
    print(f"  EPF  (11%)    : RM {EPF:,.2f}")
    print(f"  SOCSO         : RM {SOCSO:,.2f}   (band 2501-3000)")
    print(f"  PCB           : RM {PCB:,.2f}")
    print(f"  EIS  (0.2%)   : RM {EIS:,.2f}")
    print(f"  Net Pay       : RM {NET:,.2f}")
    print()
    print("Run:")
    print(f"  python -X utf8 payslip_analyzer.py {out}")
