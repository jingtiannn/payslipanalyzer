"""
analyze_payslip_tool.py
-----------------------
OpenClaw tool wrapper for the Malaysian Payslip Analyzer.

Tool name   : analyze_payslip
Input       : image_path (str, required)
              output_dir  (str, optional — default "output")
Output      : dict  {
                  status_code              : "OK" | "ERROR_FILE_NOT_FOUND" |
                                             "ERROR_NO_TEXT" | "ERROR_PIPELINE"
                  text_summary             : str | None
                  salary_chart_path        : str | None
                  loan_chart_path          : str | None
                  employer_cost_chart_path : str | None
                  error_message            : str | None
                  fields                   : dict | None
              }

File placement
--------------
Place this file in the SAME directory as payslip_analyzer.py:

    PayslipAnalyzer/
    ├── payslip_analyzer.py          ← existing pipeline
    ├── analyze_payslip_tool.py      ← THIS FILE  (OpenClaw tool wrapper)
    ├── payslip_analyzer.skill.yaml  ← skill definition
    └── output/                      ← charts are saved here

Register in OpenClaw's skill loader:
    from analyze_payslip_tool import TOOL_SCHEMA, run_tool

Quick smoke-test (run directly):
    python analyze_payslip_tool.py payslip_tc07.png
    python analyze_payslip_tool.py payslip_tc07.png --output-dir results/
"""

from __future__ import annotations

import os
import sys

# ── Ensure payslip_analyzer.py is importable regardless of cwd ─────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from payslip_analyzer import analyze_payslip   # noqa: E402  (import after path fix)


# ═══════════════════════════════════════════════════════════════════════════
# TOOL SCHEMA  —  registered with OpenClaw so it knows how to call this tool
# ═══════════════════════════════════════════════════════════════════════════

TOOL_SCHEMA: dict = {
    "name": "analyze_payslip",
    "description": (
        "Analyzes a Malaysian payslip image using OCR and computer vision. "
        "Extracts salary components (basic salary, allowances, gross salary), "
        "validates statutory deductions (EPF 11%, SOCSO, PCB / MTD, EIS 0.2%) "
        "against official Malaysian rates, calculates employer contributions "
        "(EPF 13%, SOCSO employer share, EIS 0.2%), estimates DSR loan "
        "eligibility (60% rule), and generates three PNG charts: salary "
        "breakdown pie chart, loan eligibility bar chart, and employer total "
        "cost chart. Supports English and Bahasa Malaysia payslip labels."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": (
                    "Absolute or relative path to the payslip image file. "
                    "Accepted formats: JPG, JPEG, PNG."
                ),
            },
            "output_dir": {
                "type": "string",
                "description": (
                    "Directory where the three chart PNGs will be saved. "
                    "Created automatically if it does not exist. "
                    "Default: 'output' (relative to the tool directory)."
                ),
                "default": "output",
            },
        },
        "required": ["image_path"],
    },
    "returns": {
        "type": "object",
        "description": "Structured analysis result.",
        "properties": {
            "status_code": {
                "type": "string",
                "enum": [
                    "OK",
                    "ERROR_FILE_NOT_FOUND",
                    "ERROR_NO_TEXT",
                    "ERROR_PIPELINE",
                ],
            },
            "text_summary": {
                "type": ["string", "null"],
                "description": "Full plain-text analysis report.",
            },
            "salary_chart_path": {
                "type": ["string", "null"],
                "description": "Absolute path to salary_chart.png.",
            },
            "loan_chart_path": {
                "type": ["string", "null"],
                "description": "Absolute path to loan_chart.png.",
            },
            "employer_cost_chart_path": {
                "type": ["string", "null"],
                "description": "Absolute path to employer_cost_chart.png.",
            },
            "error_message": {
                "type": ["string", "null"],
                "description": "Human-readable error detail (null on success).",
            },
            "fields": {
                "type": ["object", "null"],
                "description": "Raw extracted field values (gross, basic, etc.).",
            },
        },
    },
    # Maps OpenClaw status codes to user-facing messages
    "error_responses": {
        "ERROR_FILE_NOT_FOUND": "Could not read image, please try again.",
        "ERROR_NO_TEXT":        "No text detected, please send a clearer image.",
        "ERROR_PIPELINE":       (
            "An error occurred while processing the payslip. "
            "Please try again or contact support."
        ),
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# TOOL ENTRY POINT  —  called by OpenClaw when the skill is triggered
# ═══════════════════════════════════════════════════════════════════════════

def run_tool(image_path: str, output_dir: str = "output") -> dict:
    """
    OpenClaw calls this function with the parameters from TOOL_SCHEMA.

    Parameters
    ----------
    image_path : str
        Path to the payslip image (JPG or PNG).
    output_dir : str
        Directory to save generated charts (default: 'output').

    Returns
    -------
    dict — see TOOL_SCHEMA['returns'] for the full shape.
    """
    return analyze_payslip(image_path=image_path, output_dir=output_dir)


# ═══════════════════════════════════════════════════════════════════════════
# OPENCLAW REGISTRATION HELPER
# ═══════════════════════════════════════════════════════════════════════════

def register(openclaw_instance) -> None:
    """
    Register this tool with an OpenClaw instance.

    Usage (in your OpenClaw skill loader or main.py):

        from analyze_payslip_tool import register
        import openclaw

        app = openclaw.App()
        register(app)          # ← registers analyze_payslip tool
        app.run()

    OpenClaw will then route payslip image messages to run_tool() automatically.
    """
    openclaw_instance.register_tool(
        name=TOOL_SCHEMA["name"],
        schema=TOOL_SCHEMA,
        handler=run_tool,
    )


# ═══════════════════════════════════════════════════════════════════════════
# SMOKE-TEST  (run directly: python analyze_payslip_tool.py <image> [outdir])
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Smoke-test the OpenClaw analyze_payslip tool wrapper."
    )
    parser.add_argument("image_path", help="Path to payslip image (JPG/PNG)")
    parser.add_argument(
        "--output-dir", default="output",
        help="Directory to save charts (default: output/)"
    )
    args = parser.parse_args()

    print(f"[Tool] Calling analyze_payslip('{args.image_path}', '{args.output_dir}')")
    print()

    result = run_tool(args.image_path, args.output_dir)

    if result["status_code"] == "OK":
        print(result["text_summary"])
        print("Charts saved:")
        print(f"  Salary chart        : {result['salary_chart_path']}")
        print(f"  Loan chart          : {result['loan_chart_path']}")
        print(f"  Employer cost chart : {result['employer_cost_chart_path']}")
    else:
        user_msg = TOOL_SCHEMA["error_responses"].get(
            result["status_code"], "Unexpected error."
        )
        print(f"[{result['status_code']}] {user_msg}")
        if result["error_message"]:
            print(f"  Detail: {result['error_message']}")
        sys.exit(1)
