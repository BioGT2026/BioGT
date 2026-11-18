import argparse
import csv
import re
from pathlib import Path


TOP_DISEASE_HEADER_RE = re.compile(
    r"--- (?:Reranked )?Top \d+ Predicted Associations for '(.*)' \((.*)\) ---"
)
TOP_TYPE_HEADER_RE = re.compile(r"--- Top \d+ for type: (.*) ---")
FORCED_HEADER_RE = re.compile(r"--- Forced Associations for '(.*)' \((.*)\) -> type: (.*) ---")
RESULT_LINE_RE = re.compile(
    r"^(\d+)\. ID: (.*?), Name: (.*?) \((.*?)\) - Score: ([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)
FINAL_PATH_RE = re.compile(r"^Final Displayed Path:\s*(.*)$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="从预测日志中提取结果 rank、ID、名称、分数和最终显示路径，并导出为 CSV。"
    )
    parser.add_argument(
        "--input-log",
        type=Path,
        required=True,
        help="待解析的预测日志文件路径。",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="输出 CSV 文件路径；不传则默认写到同目录同名 .csv。",
    )
    return parser.parse_args()


def normalize_line(raw_line: str) -> str:
    return raw_line.rstrip("\n").strip()


def flush_record(rows, pending_record):
    if pending_record is None:
        return None
    rows.append(pending_record)
    return None


def parse_log(log_path: Path):
    rows = []
    current_disease = None
    current_input_type = None
    current_output_type = None
    current_section = None
    pending_record = None

    with log_path.open("r", encoding="utf-8", errors="replace") as infile:
        for raw_line in infile:
            line = normalize_line(raw_line)
            if not line:
                continue

            top_disease_match = TOP_DISEASE_HEADER_RE.match(line)
            if top_disease_match:
                pending_record = flush_record(rows, pending_record)
                current_disease = top_disease_match.group(1)
                current_input_type = top_disease_match.group(2)
                current_output_type = None
                current_section = "top"
                continue

            top_type_match = TOP_TYPE_HEADER_RE.match(line)
            if top_type_match:
                pending_record = flush_record(rows, pending_record)
                current_output_type = top_type_match.group(1).strip()
                continue

            forced_match = FORCED_HEADER_RE.match(line)
            if forced_match:
                pending_record = flush_record(rows, pending_record)
                current_disease = forced_match.group(1)
                current_input_type = forced_match.group(2)
                current_output_type = forced_match.group(3).strip()
                current_section = "forced"
                continue

            result_match = RESULT_LINE_RE.match(line)
            if result_match:
                pending_record = flush_record(rows, pending_record)
                pending_record = {
                    "disease": current_disease,
                    "input_type": current_input_type,
                    "output_type": current_output_type,
                    "section": current_section,
                    "rank": int(result_match.group(1)),
                    "result_id": result_match.group(2),
                    "result_name": result_match.group(3),
                    "result_type": result_match.group(4),
                    "score": result_match.group(5),
                    "final_displayed_path": "",
                }
                continue

            final_path_match = FINAL_PATH_RE.match(line)
            if final_path_match and pending_record is not None:
                pending_record["final_displayed_path"] = final_path_match.group(1).strip()
                continue

        flush_record(rows, pending_record)

    return rows


def write_csv(output_path: Path, rows):
    fieldnames = [
        "disease",
        "input_type",
        "output_type",
        "section",
        "rank",
        "result_id",
        "result_name",
        "result_type",
        "score",
        "final_displayed_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    output_csv = args.output_csv or args.input_log.with_suffix(".csv")
    rows = parse_log(args.input_log)
    write_csv(output_csv, rows)
    print(f"extracted_rows={len(rows)}")
    print(f"output_csv={output_csv}")


if __name__ == "__main__":
    main()