import argparse
import csv
import re
from pathlib import Path

# 增加 CSV 字段长度限制，防止长路径报错
csv.field_size_limit(1024 * 1024 * 1024)

def load_mondo_diseases(mondo_node_path):
    """加载 MONDO 疾病节点文件，建立疾病名称到 MONDO ID 的映射"""
    diseases = {}
    if not mondo_node_path or not mondo_node_path.exists():
        return diseases
    
    try:
        with mondo_node_path.open("r", encoding="utf-8", newline="") as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                disease_id = (row.get("ID") or "").strip()
                disease_name = (row.get("Name") or "").strip()
                disease_type = (row.get("Type") or "").strip()
                
                if disease_id.startswith("MONDO:") and disease_type.lower() == "disease":
                    diseases[disease_name] = disease_id
    except Exception as e:
        print(f"Warning: Failed to load MONDO file: {e}")
    
    return diseases

# -----------------------------------------------------------------------------
# 1. 高置信度中药单体关键词 (High Confidence TCM Monomers)
# 这些成分在中药现代化研究中具有代表性，且通常作为单一活性成分研究
# -----------------------------------------------------------------------------
HIGH_CONFIDENCE_TCM = {
    "artemisinin": "青蒿素", "artesunate": "青蒿琥酯", "artemether": "蒿甲醚",
    "berberine": "小檗碱", "curcumin": "姜黄素", "triptolide": "雷公藤甲素",
    "celastrol": "雷公藤红素", "baicalein": "黄芩素", "baicalin": "黄芩苷",
    "wogonin": "汉黄芩素", "matrine": "苦参碱", "oxymatrine": "氧化苦参碱",
    "tetramethylpyrazine": "四甲基吡嗪", "ligustrazine": "川芎嗪",
    "tanshinone": "丹参酮", "cryptotanshinone": "隐丹参酮",
    "salvianolic acid": "丹酚酸", "ginsenoside": "人参皂苷",
    "notoginsenoside": "三七皂苷", "paeoniflorin": "芍药苷",
    "paeonol": "丹皮酚", "emodin": "大黄素", "rhein": "大黄酸",
    "shikonin": "紫草素", "oridonin": "冬凌草甲素", "huperzine": "石杉碱甲",
    "andrographolide": "穿心莲内酯", "icariin": "淫羊藿苷",
    "camptothecin": "喜树碱", "homoharringtonine": "高三尖杉酯碱",
    "evodiamine": "吴茱萸碱", "sinomenine": "青藤碱", "osthole": "蛇床子素",
    "honokiol": "和厚朴酚", "magnolol": "厚朴酚", "salidroside": "红景天苷",
    "astragaloside": "黄芪甲苷", "astragalus polysaccharide": "黄芪多糖",
    "ganoderma polysaccharide": "灵芝多糖", "lentinan": "香菇多糖",
    "cordyceps polysaccharide": "虫草多糖", "lycium barbarum polysaccharide": "枸杞多糖",
    "puerarin": "葛根素", "tetrandrine": "粉防己碱", "stephanine": "头花千金藤胺",
    "rotundine": "罗通定", "colchicine": "秋水仙碱", # 秋水仙碱虽源自植物，但在中药单体研究中常见
    "glycyrrhizin": "甘草酸", "glycyrrhetinic acid": "甘草次酸",
    "psoralen": "补骨脂素", "isopsoralen": "异补骨脂素", "bakuchiol": "补骨脂酚",
    "schisandrin": "五味子素", "arctiin": "牛蒡苷", "podophyllotoxin": "鬼臼毒素",
    "bufalin": "蟾毒灵", "cinobufagin": "华蟾毒精", "crocin": "藏红花素",
    "capsaicin": "辣椒素", "gingerol": "姜辣素", "piperine": "胡椒碱",
    "allicin": "大蒜素", "sulforaphane": "萝卜硫素", "quercetin": "槲皮素",
    "kaempferol": "山奈酚", "luteolin": "木犀草素", "naringenin": "柚皮素",
    "genistein": "染料木素", "resveratrol": "白藜芦醇", "ferulic acid": "阿魏酸",
    "chlorogenic acid": "绿原酸", "oleanolic acid": "齐墩果酸", "ursolic acid": "熊果酸",
    "borneol": "冰片", "menthol": "薄荷醇", "camphor": "樟脑", # 天然香料类，需结合上下文
    "epigallocatechin": "表没食子儿茶素", "catechin": "儿茶素"
}

# -----------------------------------------------------------------------------
# 2. 低置信度/通用天然成分 (Low Confidence / Common Natural Products)
# 这些成分广泛存在于食物、西药和保健品中，仅当药物名称主要指代该成分时才标记
# -----------------------------------------------------------------------------
LOW_CONFIDENCE_NATURAL = {
    "caffeine": "咖啡因", "theophylline": "茶碱", "salicylic acid": "水杨酸",
    "choline": "胆碱", "vitamin": "维生素", "omega-3": "欧米伽 3",
    "fish oil": "鱼油", "melatonin": "褪黑素", "biotin": "生物素"
}

# -----------------------------------------------------------------------------
# 3. 西药/保健品/制剂 排除关键词 (Exclusion Patterns)
# 如果药物名称包含这些词，即使匹配到天然成分，也优先判定为非中药单体
# -----------------------------------------------------------------------------
WESTERN_EXCLUSION_PATTERNS = [
    r"\bmultivitamin\b", r"\bmineral\b", r"\bsupplement\b", r"\bvitamin\s*\d+\b",
    r"\bacetaminophen\b", r"\bparacetamol\b", r"\bibuprofen\b", r"\baspirin\b",
    r"\bamoxicillin\b", r"\bpenicillin\b", r"\bmetformin\b", r"\binsulin\b",
    r"\bcodeine\b", r"\bmorphine\b", r"\bdiazepam\b", r"\bolanzapine\b",
    r"\brisperidone\b", r"\bfluoxetine\b", r"\bsertraline\b", r"\bclozapine\b",
    r"\bcream\b", r"\blotion\b", r"\bshampoo\b", r"\bsoap\b", r"\bcleanser\b",
    r"\binjection\b", r"\binjectable\b", r"\bvaccine\b", r"\bantibody\b",
    r"\bmab\b", r"\bnib\b", r"\btinib\b", r"\bmap\b", r"\bgene\b", r"\btherapy\b",
    r"\bpatch\b", r"\btablet\b", r"\bcaplet\b", r"\bcapsule\b", # 剂型词若与品牌连用需小心，这里作为辅助判断
    r"\btylenol\b", r"\badvil\b", r"\bmotrin\b", r"\bvoxzogo\b", r"\bbotox\b",
    r"\bfiyarro\b", r"\bkinevac\b", r"\balunbrig\b", r"\bcubicin\b", r"\bacticlate\b"
]

# 编译正则以提高效率
EXCLUSION_REGEX = [re.compile(pattern, re.IGNORECASE) for pattern in WESTERN_EXCLUSION_PATTERNS]

def normalize_text(value):
    return (value or "").strip()

def parse_score(value):
    text = normalize_text(value)
    if not text:
        return None

    try:
        return float(text)
    except ValueError:
        return None

def calculate_path_length(path_with_drug):
    path_text = normalize_text(path_with_drug)
    if not path_text:
        return 0
    return path_text.count("->") + 1

def is_excluded_drug(drug_name):
    """检查药物名称是否属于明确的西药、保健品或制剂"""
    lowered = drug_name.lower()
    for pattern in EXCLUSION_REGEX:
        if pattern.search(lowered):
            return True
    return False

def match_tcm_keywords(drug_name):
    """
    匹配中药单体关键词
    返回：(matched_keywords_list, confidence_level)
    confidence_level: 'high' (高置信度), 'low' (低置信度), None (无匹配)
    """
    lowered = drug_name.lower()
    matched_high = []
    matched_low = []

    # 1. 匹配高置信度关键词 (使用单词边界，避免部分匹配)
    for keyword, cn_name in HIGH_CONFIDENCE_TCM.items():
        # 使用正则确保匹配的是独立单词或词组，避免匹配到长词的一部分
        pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
        if pattern.search(lowered):
            matched_high.append(f"{keyword} ({cn_name})")
    
    # 2. 匹配低置信度关键词 (仅当药物名主要由该成分构成时)
    for keyword, cn_name in LOW_CONFIDENCE_NATURAL.items():
        pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
        if pattern.search(lowered):
            # 对于低置信度词，检查药物名是否过于复杂（如复方）
            # 简单启发式：如果药物名长度很长且包含多个成分，降低权重
            if len(drug_name.split()) < 6: 
                matched_low.append(f"{keyword} ({cn_name})")

    if matched_high:
        return matched_high, 'high'
    elif matched_low:
        return matched_low, 'low'
    return [], None

def screen_candidates(input_file, mondo_diseases):
    rows = []
    tcm_count = 0
    excluded_count = 0
    
    with input_file.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            drug_name = normalize_text(row.get("drug_name"))
            target_disease = normalize_text(row.get("target_disease"))
            score = normalize_text(row.get("Score"))
            score_value = parse_score(score)
            path_with_drug = normalize_text(row.get("path_with_drug"))
            path_length = calculate_path_length(path_with_drug)
            
            # 1. 优先检查排除列表 (Exclusion Check)
            is_western = is_excluded_drug(drug_name)
            
            # 2. 匹配关键词 (Keyword Matching)
            matched_terms, confidence = match_tcm_keywords(drug_name)
            
            # 3. 决策逻辑 (Decision Logic)
            tcm_label = "非中药"
            reason = "未匹配到中药单体关键词"
            
            if is_western:
                tcm_label = "非中药 (西药/保健品/制剂)"
                reason = f"命中排除规则 (Exclusion Pattern). Matched: {matched_terms if matched_terms else 'None'}"
                excluded_count += 1
            elif matched_terms:
                if confidence == 'high':
                    tcm_label = "潜在中药单体"
                    reason = f"高置信度匹配 (High Confidence): {', '.join(matched_terms)}"
                    tcm_count += 1
                elif confidence == 'low':
                    # 低置信度需要人工复核，标记为“天然成分 (需核实)"
                    tcm_label = "天然成分 (需核实)"
                    reason = f"低置信度匹配 (Low Confidence/Common): {', '.join(matched_terms)}. 可能为西药或营养补充剂。"
            else:
                tcm_label = "非中药"
                reason = "无关键词匹配"

            # 4. 查找 MONDO ID
            mondo_id = mondo_diseases.get(target_disease, "")

            rows.append({
                "target_disease": target_disease,
                "Score": score,
                "path_length": path_length,
                "mondo_id": mondo_id,
                "tcm_label": tcm_label,
                "matched_keyword": " || ".join(matched_terms),
                "matched_tcm_name": " || ".join([t.split('(')[-1].rstrip(')') for t in matched_terms]),
                "drug_name": drug_name,
                "path_with_drug": path_with_drug,
                "classification_reason": reason,
                "_score_value": score_value,
            })

    rows.sort(
        key=lambda item: (
            -(item["_score_value"] if item["_score_value"] is not None else float("-inf")),
            item["path_length"],
            item["target_disease"],
        )
    )

    for row in rows:
        row.pop("_score_value", None)
    
    print(f"Screening Complete: Total={len(rows)}, TCM_Monomer={tcm_count}, Excluded_Western={excluded_count}")
    return rows

def parse_args():
    parser = argparse.ArgumentParser(description="精确筛选潜在中药单体药物候选。")
    parser.add_argument(
        "--input-file",
        default="top1_protein_results_with_drug_paths_formatted.csv",
        type=Path,
        help="包含 drug_name 和 path_with_drug 的输入 CSV。",
    )
    parser.add_argument(
        "--mondo-node",
        default=None,
        type=Path,
        help="MONDO 疾病节点文件路径。（可选）",
    )
    parser.add_argument(
        "--output-file",
        default="tcm_monomer_candidate_drugs.csv",
        type=Path,
        help="筛选后的输出 CSV。",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    mondo_diseases = {}
    if args.mondo_node:
        mondo_diseases = load_mondo_diseases(args.mondo_node)
        print(f"Loaded MONDO diseases: {len(mondo_diseases)}")
    
    rows = screen_candidates(args.input_file, mondo_diseases)

    # 使用 utf-8-sig 编码，确保 Excel 能正确显示中文
    fieldnames = ["target_disease", "Score", "path_length", "mondo_id", "tcm_label", "matched_keyword", "matched_tcm_name", "drug_name", "path_with_drug", "classification_reason"]
    
    with args.output_file.open("w", encoding="utf-8-sig", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Output saved to: {args.output_file}")

if __name__ == "__main__":
    main()