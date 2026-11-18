import argparse
import csv
import re
from pathlib import Path


DISEASE_HEADER_RE = re.compile(r"--- Top 1 Predicted Associations for '(.*)' \(disease\) ---")
TYPE_HEADER_RE = re.compile(r"--- Top 1 for type: (.*) ---")
LOG_LINE_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - ")
INLINE_LOG_RE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - ")
SCORE_VALUE_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def extract_path_from_block(block_lines):
    if not block_lines:
        return None

    joined = "\n".join(line.rstrip("\n") for line in block_lines)
    path_marker = " Path: "
    marker_index = joined.find(path_marker)
    if marker_index == -1:
        return None

    path_text = joined[marker_index + len(path_marker):].strip()
    inline_log_match = INLINE_LOG_RE.search(path_text)
    if inline_log_match:
        path_text = path_text[:inline_log_match.start()].rstrip()

    path_text = re.sub(r"\s+", " ", path_text).strip()
    path_text = re.sub(r"\(\s+\|", "(|", path_text)
    path_text = re.sub(r"\s+\)", ")", path_text)
    path_text = re.sub(r"\s+->\s+", " -> ", path_text)
    return path_text


def extract_drug_names_from_path(path_text):
    drug_names = []
    for segment in path_text.split(" -> "):
        if not segment.lstrip().startswith("drug:"):
            continue

        start = segment.find("(")
        end = segment.rfind(")")
        if start == -1 or end == -1 or end <= start:
            continue

        raw_name = segment[start + 1:end]
        cleaned_name = re.sub(r"\s+", " ", raw_name).strip().strip("|").strip()
        if cleaned_name:
            drug_names.append(cleaned_name)

    return " || ".join(drug_names)

def extract_score_from_block(block_lines):
    if not block_lines:
        return None

    joined = "\n".join(line.rstrip("\n") for line in block_lines)
    score_marker = " Score: "
    marker_index = joined.find(score_marker)
    if marker_index == -1:
        return None
    
    score_text = joined[marker_index + len(score_marker):].strip()
    inline_log_match = INLINE_LOG_RE.search(score_text)
    if inline_log_match:
        score_text = score_text[:inline_log_match.start()].rstrip()
    score_text = re.sub(r"\s+", " ", score_text).strip()

    score_match = SCORE_VALUE_RE.match(score_text)
    if not score_match:
        return None

    return score_match.group(0)

def flush_result(rows, current_disease, current_type, result_lines):
    if current_disease is None or current_type != "protein":
        return

    path_text = extract_path_from_block(result_lines)
    if path_text and "drug:" in path_text:
        rows.append({
            "target_disease": current_disease,
            "Score": extract_score_from_block(result_lines),
            "drug_name": extract_drug_names_from_path(path_text),
            "path_with_drug": path_text,
        })


def parse_log(log_path):
    rows = []
    current_disease = None
    current_type = None
    current_result_lines = []

    with log_path.open("r", encoding="utf-8", errors="replace") as infile:
        for raw_line in infile:
            disease_match = DISEASE_HEADER_RE.search(raw_line)
            type_match = TYPE_HEADER_RE.search(raw_line)
            stripped = raw_line.lstrip()

            if disease_match:
                flush_result(rows, current_disease, current_type, current_result_lines)
                current_result_lines = []
                current_disease = disease_match.group(1)
                current_type = None
                continue

            if type_match:
                flush_result(rows, current_disease, current_type, current_result_lines)
                current_result_lines = []
                current_type = type_match.group(1).strip().lower()
                continue

            if stripped.startswith("1. ID:"):
                flush_result(rows, current_disease, current_type, current_result_lines)
                current_result_lines = [raw_line]
                continue

            if current_result_lines:
                if LOG_LINE_RE.match(raw_line):
                    flush_result(rows, current_disease, current_type, current_result_lines)
                    current_result_lines = []
                    continue
                current_result_lines.append(raw_line)

    flush_result(rows, current_disease, current_type, current_result_lines)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="提取日志中 Top 1 protein 结果里包含 drug 节点的路径。"
    )
    parser.add_argument(
        "--log-file",
        default="predict_protein_metabolic_disease_20260307_224634.log",
        type=Path,
        help="待解析的日志文件路径。",
    )
    parser.add_argument(
        "--output-file",
        default="top1_protein_results_with_drug_paths.csv",
        type=Path,
        help="输出 CSV 文件路径。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = parse_log(args.log_file)

    with args.output_file.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["target_disease", "Score", "drug_name", "path_with_drug"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"extracted_rows={len(rows)}")
    print(f"output_file={args.output_file}")


if __name__ == "__main__":
    main()