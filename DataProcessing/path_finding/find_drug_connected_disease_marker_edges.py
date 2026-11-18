#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
给定药物 DrugBank ID 和最大长度，找出满足以下条件的候选结果：
1. 在疾病 M 和目标生物标志物 N 之间寻找最短路径；
2. 路径必须经过给定药物 DrugBank ID；
3. M -> ... -> 药物 -> ... -> N 的路径不超过最大长度；
4. 默认要求路径同时经过至少一个 GO 节点（biological_process/molecular_function/cellular_component），可用开关关闭；
5. 疾病 M 必须是 MONDO leaf term，不是疾病分类节点；
6. M-N 来自联网查证候选，且该 M-N 不已经存在于 MarkerDB。

默认边文件：disease_connect/merged_edge.csv

示例：
python3 find_drug_connected_disease_marker_edges.py \
  --target-type protein \
  --no-require-go-node \
  --max-nodes 6

python3 find_drug_connected_disease_marker_edges.py \
  --drug-id DB01229 \
  --target-type metabolite \
    --max-nodes 4 \
  --output-prefix paclitaxel_metabolite_h3

说明：
- DNA 在当前图中按 gene 节点处理。
- 默认使用本脚本内置的 5 个候选药物 ID；也可以用 --drug-id 覆盖。
- 默认按无向图处理；加 --directed 后按 Edge Direction 处理。
- 默认要求路径包含 GO 节点；加 --no-require-go-node 后不再限制。
- MarkerDB 只作为排除库和禁用边来源，不作为候选 M-N 的正例来源。
- 默认禁止路径经过药物后再次进入 disease 节点，避免 disease->drug->disease->marker 的拓扑循环。
"""

import argparse
import csv
import os
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path


FIELDNAMES_EDGE = [
    "Node 1 ID",
    "Node 1 Data Type",
    "Node 1 Source",
    "Edge Type",
    "Edge Direction",
    "Edge Weight",
    "Node 2 ID",
    "Node 2 Data Type",
    "Node 2 Source",
]

DEFAULT_EDGE_FILE = Path(__file__).resolve().parent / "disease_connect" / "merged_edge.csv"
DEFAULT_MARKER_EDGE_FILE = Path(__file__).resolve().parent / "disease_connect" / "marker_edge.csv"
DEFAULT_MONDO_FILE = Path(__file__).resolve().parent / "mondo.obo"

GO_NODE_TYPES = {"biological_process", "molecular_function", "cellular_component"}

DEFAULT_DRUGS = {
    "青蒿素/artemisinin": ["DB13132"],
    "紫杉醇/paclitaxel": ["DB01229"],
    "长春碱/vinblastine": ["DB00570"],
    "吗啡/morphine": ["DB00295"],
    "大麻碱素/cannabinoid": ["DB09061", "DB00470"],
}

TARGET_TYPE_ALIASES = {
    "dna": {"dna", "gene"},
    "gene": {"dna", "gene"},
    "rna": {"rna"},
    "protein": {"protein"},
    "metabolite": {"metabolite"},
}

WEB_CURATED_CANDIDATE_PAIRS = [
    {
        "drug_id": "DB01229",
        "drug_name": "paclitaxel",
        "disease_id": "MONDO:0008170",
        "disease_name": "ovarian cancer",
        "marker_id": "Uniprot:Q8WXI7",
        "marker_type": "protein",
        "marker_name": "MUC16/CA125",
        "evidence": "PMID:36233339; PMID:30917847; PubMed query: ovarian cancer CA125 HE4 biomarker",
    },
    {
        "drug_id": "DB01229",
        "drug_name": "paclitaxel",
        "disease_id": "MONDO:0008170",
        "disease_name": "ovarian cancer",
        "marker_id": "Uniprot:Q14508",
        "marker_type": "protein",
        "marker_name": "WFDC2/HE4",
        "evidence": "PMID:30917847; PMID:36189508; PubMed query: ovarian cancer CA125 HE4 biomarker",
    },
    {
        "drug_id": "DB01229",
        "drug_name": "paclitaxel",
        "disease_id": "MONDO:0003874",
        "disease_name": "ovarian serous surface papillary adenocarcinoma",
        "marker_id": "NCBIGene:94025",
        "marker_type": "gene",
        "marker_name": "MUC16/CA125",
        "evidence": "Leaf MONDO subtype of ovarian serous adenocarcinoma; PMID:36233339; PMID:30917847; PubMed query: ovarian cancer CA125 HE4 biomarker",
    },
    {
        "drug_id": "DB01229",
        "drug_name": "paclitaxel",
        "disease_id": "MONDO:0003874",
        "disease_name": "ovarian serous surface papillary adenocarcinoma",
        "marker_id": "NCBIGene:10406",
        "marker_type": "gene",
        "marker_name": "WFDC2/HE4",
        "evidence": "Leaf MONDO subtype of ovarian serous adenocarcinoma; PMID:30917847; PMID:36189508; PubMed query: ovarian cancer CA125 HE4 biomarker",
    },
    {
        "drug_id": "DB01229",
        "drug_name": "paclitaxel",
        "disease_id": "MONDO:0003874",
        "disease_name": "ovarian serous surface papillary adenocarcinoma",
        "marker_id": "NCBIGene:2348",
        "marker_type": "gene",
        "marker_name": "FOLR1/folate receptor alpha",
        "evidence": "Leaf MONDO subtype of ovarian serous adenocarcinoma; PMID:40446758; PMID:40121972; PMID:30093227; PMID:29098867; PubMed query: ovarian cancer FOLR1 folate receptor alpha biomarker",
    },
    {
        "drug_id": "DB01229",
        "drug_name": "paclitaxel",
        "disease_id": "MONDO:0005233",
        "disease_name": "non-small cell lung carcinoma",
        "marker_id": "Uniprot:P00533",
        "marker_type": "protein",
        "marker_name": "EGFR",
        "evidence": "PMID:34985916; PMID:38199926; PMID:35534623; PubMed query: non-small cell lung cancer EGFR ALK biomarker",
    },
    {
        "drug_id": "DB01229",
        "drug_name": "paclitaxel",
        "disease_id": "MONDO:0005233",
        "disease_name": "non-small cell lung carcinoma",
        "marker_id": "Uniprot:Q9UM73",
        "marker_type": "protein",
        "marker_name": "ALK",
        "evidence": "PMID:34985916; PMID:38199926; PMID:35534623; PubMed query: non-small cell lung cancer EGFR ALK biomarker",
    },
    {
        "drug_id": "DB01229",
        "drug_name": "paclitaxel",
        "disease_id": "MONDO:0008170",
        "disease_name": "ovarian cancer",
        "marker_id": "Uniprot:P15328",
        "marker_type": "protein",
        "marker_name": "FOLR1/folate receptor alpha",
        "evidence": "PMID:40446758; PMID:40121972; PMID:30093227; PMID:29098867; PubMed query: ovarian cancer FOLR1 folate receptor alpha biomarker",
    },
    {
        "drug_id": "DB01229",
        "drug_name": "paclitaxel",
        "disease_id": "MONDO:0003874",
        "disease_name": "ovarian serous surface papillary adenocarcinoma",
        "marker_id": "Uniprot:Q8WXI7",
        "marker_type": "protein",
        "marker_name": "MUC16/CA125",
        "evidence": "Leaf MONDO subtype of ovarian serous adenocarcinoma; PMID:36233339; PMID:30917847; PubMed query: ovarian cancer CA125 HE4 biomarker",
    },
    {
        "drug_id": "DB01229",
        "drug_name": "paclitaxel",
        "disease_id": "MONDO:0003874",
        "disease_name": "ovarian serous surface papillary adenocarcinoma",
        "marker_id": "Uniprot:Q14508",
        "marker_type": "protein",
        "marker_name": "WFDC2/HE4",
        "evidence": "Leaf MONDO subtype of ovarian serous adenocarcinoma; PMID:30917847; PMID:36189508; PubMed query: ovarian cancer CA125 HE4 biomarker",
    },
    {
        "drug_id": "DB01229",
        "drug_name": "paclitaxel",
        "disease_id": "MONDO:0003874",
        "disease_name": "ovarian serous surface papillary adenocarcinoma",
        "marker_id": "Uniprot:P15328",
        "marker_type": "protein",
        "marker_name": "FOLR1/folate receptor alpha",
        "evidence": "Leaf MONDO subtype of ovarian serous adenocarcinoma; PMID:40446758; PMID:40121972; PMID:30093227; PMID:29098867; PubMed query: ovarian cancer FOLR1 folate receptor alpha biomarker",
    },
    {
        "drug_id": "DB01229",
        "drug_name": "paclitaxel",
        "disease_id": "MONDO:0005233",
        "disease_name": "non-small cell lung carcinoma",
        "marker_id": "Uniprot:P01116",
        "marker_type": "protein",
        "marker_name": "KRAS",
        "evidence": "PMID:33316418; PMID:38199926; PMID:32730814; PubMed query: non-small cell lung cancer KRAS biomarker",
    },
    {
        "drug_id": "DB01229",
        "drug_name": "paclitaxel",
        "disease_id": "MONDO:0005233",
        "disease_name": "non-small cell lung carcinoma",
        "marker_id": "Uniprot:Q9NZQ7",
        "marker_type": "protein",
        "marker_name": "CD274/PD-L1",
        "evidence": "PMID:34376553; PMID:37930659; PubMed query: non-small cell lung cancer PD-L1 biomarker",
    },
    {
        "drug_id": "DB01229",
        "drug_name": "paclitaxel",
        "disease_id": "MONDO:0005233",
        "disease_name": "non-small cell lung carcinoma",
        "marker_id": "Uniprot:P15056",
        "marker_type": "protein",
        "marker_name": "BRAF",
        "evidence": "PMID:33316418; PMID:38199926; PMID:32730814; PubMed query: non-small cell lung cancer BRAF biomarker",
    },
    {
        "drug_id": "DB01229",
        "drug_name": "paclitaxel",
        "disease_id": "MONDO:0005233",
        "disease_name": "non-small cell lung carcinoma",
        "marker_id": "Uniprot:P08581",
        "marker_type": "protein",
        "marker_name": "MET",
        "evidence": "PMID:33316418; PMID:38199926; PMID:31125062; PubMed query: non-small cell lung cancer MET biomarker",
    },
    {
        "drug_id": "DB01229",
        "drug_name": "paclitaxel",
        "disease_id": "MONDO:0005233",
        "disease_name": "non-small cell lung carcinoma",
        "marker_id": "Uniprot:P08922",
        "marker_type": "protein",
        "marker_name": "ROS1",
        "evidence": "PMID:33316418; PMID:38199926; PubMed query: non-small cell lung cancer ROS1 biomarker",
    },
    {
        "drug_id": "DB01229",
        "drug_name": "paclitaxel",
        "disease_id": "MONDO:0005233",
        "disease_name": "non-small cell lung carcinoma",
        "marker_id": "Uniprot:P07949",
        "marker_type": "protein",
        "marker_name": "RET",
        "evidence": "PMID:33316418; PMID:38199926; PubMed query: non-small cell lung cancer RET biomarker",
    },
    {
        "drug_id": "DB13132",
        "drug_name": "artemisinin",
        "disease_id": "MONDO:0005920",
        "disease_name": "Plasmodium falciparum malaria",
        "marker_id": "Uniprot:P02741",
        "marker_type": "protein",
        "marker_name": "CRP",
        "evidence": "PMID:34290484; PubMed query: malaria C-reactive protein biomarker",
    },
    {
        "drug_id": "DB13132",
        "drug_name": "artemisinin",
        "disease_id": "MONDO:0005920",
        "disease_name": "Plasmodium falciparum malaria",
        "marker_id": "Uniprot:P00338",
        "marker_type": "protein",
        "marker_name": "LDHA/lactate dehydrogenase",
        "evidence": "PubMed query: malaria lactate dehydrogenase biomarker",
    },
    {
        "drug_id": "DB13132",
        "drug_name": "artemisinin",
        "disease_id": "MONDO:0005920",
        "disease_name": "Plasmodium falciparum malaria",
        "marker_id": "Uniprot:P05362",
        "marker_type": "protein",
        "marker_name": "ICAM1/CD54",
        "evidence": "PMID:7533138; PMID:1711552; PMID:35715862; PubMed query: malaria ICAM-1 biomarker",
    },
    {
        "drug_id": "DB01229",
        "drug_name": "paclitaxel",
        "disease_id": "MONDO:0007254",
        "disease_name": "breast cancer",
        "marker_id": "Uniprot:P04626",
        "marker_type": "protein",
        "marker_name": "ERBB2/HER2",
        "evidence": "PMID:11248153; PMID:25564897; PubMed query: paclitaxel breast cancer HER2 biomarker",
    },
    {
        "drug_id": "DB01229",
        "drug_name": "paclitaxel",
        "disease_id": "MONDO:0021116",
        "disease_name": "luminal A breast carcinoma",
        "marker_id": "NCBIGene:2099",
        "marker_type": "gene",
        "marker_name": "ESR1/estrogen receptor alpha",
        "evidence": "Leaf MONDO luminal breast carcinoma subtype; PMID:28490348; PMID:34408238; PMID:27389414; PMID:34371382; PubMed query: breast cancer ESR1 PGR biomarker",
    },
    {
        "drug_id": "DB01229",
        "drug_name": "paclitaxel",
        "disease_id": "MONDO:0021116",
        "disease_name": "luminal A breast carcinoma",
        "marker_id": "NCBIGene:5241",
        "marker_type": "gene",
        "marker_name": "PGR/progesterone receptor",
        "evidence": "Leaf MONDO luminal breast carcinoma subtype; PMID:28490348; PMID:34408238; PMID:27389414; PMID:34371382; PubMed query: breast cancer ESR1 PGR biomarker",
    },
    {
        "drug_id": "DB01229",
        "drug_name": "paclitaxel",
        "disease_id": "MONDO:0021115",
        "disease_name": "luminal B breast carcinoma",
        "marker_id": "NCBIGene:2099",
        "marker_type": "gene",
        "marker_name": "ESR1/estrogen receptor alpha",
        "evidence": "Leaf MONDO luminal breast carcinoma subtype; PMID:28490348; PMID:34408238; PMID:27389414; PMID:34371382; PubMed query: breast cancer ESR1 PGR biomarker",
    },
    {
        "drug_id": "DB01229",
        "drug_name": "paclitaxel",
        "disease_id": "MONDO:0021115",
        "disease_name": "luminal B breast carcinoma",
        "marker_id": "NCBIGene:5241",
        "marker_type": "gene",
        "marker_name": "PGR/progesterone receptor",
        "evidence": "Leaf MONDO luminal breast carcinoma subtype; PMID:28490348; PMID:34408238; PMID:27389414; PMID:34371382; PubMed query: breast cancer ESR1 PGR biomarker",
    },
    {
        "drug_id": "DB01229",
        "drug_name": "paclitaxel",
        "disease_id": "MONDO:0007254",
        "disease_name": "breast cancer",
        "marker_id": "Uniprot:P03372",
        "marker_type": "protein",
        "marker_name": "ESR1/estrogen receptor alpha",
        "evidence": "PMID:28490348; PMID:34408238; PMID:27389414; PMID:34371382; PubMed query: breast cancer ESR1 PGR biomarker",
    },
    {
        "drug_id": "DB01229",
        "drug_name": "paclitaxel",
        "disease_id": "MONDO:0007254",
        "disease_name": "breast cancer",
        "marker_id": "Uniprot:P06401",
        "marker_type": "protein",
        "marker_name": "PGR/progesterone receptor",
        "evidence": "PMID:28490348; PMID:34408238; PMID:27389414; PMID:34371382; PubMed query: breast cancer ESR1 PGR biomarker",
    },
    {
        "drug_id": "DB01229",
        "drug_name": "paclitaxel",
        "disease_id": "MONDO:0021116",
        "disease_name": "luminal A breast carcinoma",
        "marker_id": "Uniprot:P03372",
        "marker_type": "protein",
        "marker_name": "ESR1/estrogen receptor alpha",
        "evidence": "Leaf MONDO luminal breast carcinoma subtype; PMID:28490348; PMID:34408238; PMID:27389414; PMID:34371382; PubMed query: breast cancer ESR1 PGR biomarker",
    },
    {
        "drug_id": "DB01229",
        "drug_name": "paclitaxel",
        "disease_id": "MONDO:0021116",
        "disease_name": "luminal A breast carcinoma",
        "marker_id": "Uniprot:P06401",
        "marker_type": "protein",
        "marker_name": "PGR/progesterone receptor",
        "evidence": "Leaf MONDO luminal breast carcinoma subtype; PMID:28490348; PMID:34408238; PMID:27389414; PMID:34371382; PubMed query: breast cancer ESR1 PGR biomarker",
    },
    {
        "drug_id": "DB01229",
        "drug_name": "paclitaxel",
        "disease_id": "MONDO:0021115",
        "disease_name": "luminal B breast carcinoma",
        "marker_id": "Uniprot:P03372",
        "marker_type": "protein",
        "marker_name": "ESR1/estrogen receptor alpha",
        "evidence": "Leaf MONDO luminal breast carcinoma subtype; PMID:28490348; PMID:34408238; PMID:27389414; PMID:34371382; PubMed query: breast cancer ESR1 PGR biomarker",
    },
    {
        "drug_id": "DB01229",
        "drug_name": "paclitaxel",
        "disease_id": "MONDO:0021115",
        "disease_name": "luminal B breast carcinoma",
        "marker_id": "Uniprot:P06401",
        "marker_type": "protein",
        "marker_name": "PGR/progesterone receptor",
        "evidence": "Leaf MONDO luminal breast carcinoma subtype; PMID:28490348; PMID:34408238; PMID:27389414; PMID:34371382; PubMed query: breast cancer ESR1 PGR biomarker",
    },
    {
        "drug_id": "DB00570",
        "drug_name": "vinblastine",
        "disease_id": "MONDO:0009691",
        "disease_name": "mycosis fungoides",
        "marker_id": "Uniprot:P01730",
        "marker_type": "protein",
        "marker_name": "CD4",
        "evidence": "PMID:39093388; PMID:19895171; PMID:17013537; PMID:16844512; PubMed query: mycosis fungoides CD4 CD3 diagnostic biomarker",
    },
    {
        "drug_id": "DB00570",
        "drug_name": "vinblastine",
        "disease_id": "MONDO:0009691",
        "disease_name": "mycosis fungoides",
        "marker_id": "Uniprot:P07766",
        "marker_type": "protein",
        "marker_name": "CD3E",
        "evidence": "PMID:39093388; PMID:17496190; PMID:19895171; PMID:17013537; PubMed query: mycosis fungoides CD4 CD3 diagnostic biomarker",
    },
    {
        "drug_id": "DB00570",
        "drug_name": "vinblastine",
        "disease_id": "MONDO:0023067",
        "disease_name": "endemic Kaposi sarcoma",
        "marker_id": "Uniprot:P15692",
        "marker_type": "protein",
        "marker_name": "VEGFA",
        "evidence": "Leaf MONDO subtype of Kaposi sarcoma; PMID:28506734; PMID:30979736; PMID:31252633; PMID:37809059; PubMed query: Kaposi sarcoma VEGF biomarker",
    },
    {
        "drug_id": "DB00570",
        "drug_name": "vinblastine",
        "disease_id": "MONDO:0023067",
        "disease_name": "endemic Kaposi sarcoma",
        "marker_id": "Uniprot:P05231",
        "marker_type": "protein",
        "marker_name": "IL6",
        "evidence": "Leaf MONDO subtype of Kaposi sarcoma; PMID:9171222; PMID:11698289; PMID:26646010; PMID:30265712; PubMed query: Kaposi sarcoma interleukin-6 biomarker",
    },
    {
        "drug_id": "DB00295",
        "drug_name": "morphine",
        "disease_id": "MONDO:0041656",
        "disease_name": "ST-elevation myocardial infarction",
        "marker_id": "Uniprot:P19429",
        "marker_type": "protein",
        "marker_name": "TNNI3/cardiac troponin I",
        "evidence": "PMID:25943557; PMID:36848390; PMID:38344871; PMID:20639847; PubMed query: ST-elevation myocardial infarction troponin I troponin T biomarker",
    },
    {
        "drug_id": "DB00295",
        "drug_name": "morphine",
        "disease_id": "MONDO:0041656",
        "disease_name": "ST-elevation myocardial infarction",
        "marker_id": "Uniprot:P45379",
        "marker_type": "protein",
        "marker_name": "TNNT2/cardiac troponin T",
        "evidence": "PMID:25943557; PMID:36848390; PMID:31269858; PMID:20639847; PubMed query: ST-elevation myocardial infarction troponin I troponin T biomarker",
    },
    {
        "drug_id": "DB00295",
        "drug_name": "morphine",
        "disease_id": "MONDO:0041656",
        "disease_name": "ST-elevation myocardial infarction",
        "marker_id": "Uniprot:P02144",
        "marker_type": "protein",
        "marker_name": "MB/myoglobin",
        "evidence": "PMID:20639847; PubMed query: myocardial infarction myoglobin cardiac biomarker",
    },
    {
        "drug_id": "DB00295",
        "drug_name": "morphine",
        "disease_id": "MONDO:0016669",
        "disease_name": "sickle cell-hemoglobin c disease syndrome",
        "marker_id": "Uniprot:P68871",
        "marker_type": "protein",
        "marker_name": "HBB/hemoglobin beta",
        "evidence": "PMID:27802331; PMID:26928604; PMID:8756079; PubMed query: sickle cell hemoglobin C disease HBB biomarker",
    },
    {
        "drug_id": "DB00295",
        "drug_name": "morphine",
        "disease_id": "MONDO:0016669",
        "disease_name": "sickle cell-hemoglobin c disease syndrome",
        "marker_id": "Uniprot:P69905",
        "marker_type": "protein",
        "marker_name": "HBA1/HBA2 hemoglobin alpha",
        "evidence": "PMID:27802331; PMID:26928604; PMID:8756079; PubMed query: sickle cell hemoglobin C disease hemoglobin biomarker",
    },
    {
        "drug_id": "DB09061",
        "drug_name": "cannabidiol",
        "disease_id": "MONDO:0100135",
        "disease_name": "Dravet syndrome",
        "marker_id": "Uniprot:P35498",
        "marker_type": "protein",
        "marker_name": "SCN1A/Nav1.1",
        "evidence": "PMID:35074891; PMID:38229878; PMID:30146492; PMID:27045673; PubMed query: Dravet syndrome SCN1A biomarker",
    },
    {
        "drug_id": "DB00470",
        "drug_name": "dronabinol",
        "disease_id": "MONDO:0005797",
        "disease_name": "HIV wasting syndrome",
        "marker_id": "Uniprot:P01730",
        "marker_type": "protein",
        "marker_name": "CD4",
        "evidence": "PMID:9792383; PMID:26264263; PMID:9424041; PMID:26760233; PubMed query: HIV wasting syndrome CD4 count biomarker",
    },
    {
        "drug_id": "DB00470",
        "drug_name": "dronabinol",
        "disease_id": "MONDO:0012268",
        "disease_name": "AIDS",
        "marker_id": "Uniprot:P01730",
        "marker_type": "protein",
        "marker_name": "CD4",
        "evidence": "PMID:9792383; PMID:26264263; PMID:9424041; PubMed query: HIV AIDS CD4 count biomarker",
    },
    {
        "drug_id": "DB00470",
        "drug_name": "dronabinol",
        "disease_id": "MONDO:0005314",
        "disease_name": "relapsing-remitting multiple sclerosis",
        "marker_id": "Uniprot:P07196",
        "marker_type": "protein",
        "marker_name": "NEFL/neurofilament light chain",
        "evidence": "PMID:38354532; PMID:28512753; PMID:35182510; PMID:37930670; PubMed query: multiple sclerosis neurofilament light biomarker",
    },
    {
        "drug_id": "DB00470",
        "drug_name": "dronabinol",
        "disease_id": "MONDO:0005314",
        "disease_name": "relapsing-remitting multiple sclerosis",
        "marker_id": "Uniprot:P02686",
        "marker_type": "protein",
        "marker_name": "MBP/myelin basic protein",
        "evidence": "PMID:34917290; PMID:36614334; PMID:32028099; PMID:34889995; PubMed query: multiple sclerosis myelin basic protein biomarker",
    },
    {
        "drug_id": "DB01229",
        "drug_name": "paclitaxel",
        "disease_id": "MONDO:0008170",
        "disease_name": "ovarian cancer",
        "marker_id": "NCBIGene:94025",
        "marker_type": "gene",
        "marker_name": "MUC16/CA125",
        "evidence": "PMID:36233339; PMID:30917847; PubMed query: ovarian cancer CA125 HE4 biomarker",
    },
    {
        "drug_id": "DB01229",
        "drug_name": "paclitaxel",
        "disease_id": "MONDO:0008170",
        "disease_name": "ovarian cancer",
        "marker_id": "NCBIGene:10406",
        "marker_type": "gene",
        "marker_name": "WFDC2/HE4",
        "evidence": "PMID:30917847; PMID:36189508; PubMed query: ovarian cancer CA125 HE4 biomarker",
    },
    {
        "drug_id": "DB01229",
        "drug_name": "paclitaxel",
        "disease_id": "MONDO:0005233",
        "disease_name": "non-small cell lung carcinoma",
        "marker_id": "NCBIGene:1956",
        "marker_type": "gene",
        "marker_name": "EGFR",
        "evidence": "PMID:34985916; PMID:38199926; PMID:35534623; PubMed query: non-small cell lung cancer EGFR ALK biomarker",
    },
    {
        "drug_id": "DB01229",
        "drug_name": "paclitaxel",
        "disease_id": "MONDO:0005233",
        "disease_name": "non-small cell lung carcinoma",
        "marker_id": "NCBIGene:238",
        "marker_type": "gene",
        "marker_name": "ALK",
        "evidence": "PMID:34985916; PMID:38199926; PMID:35534623; PubMed query: non-small cell lung cancer EGFR ALK biomarker",
    },
    {
        "drug_id": "DB13132",
        "drug_name": "artemisinin",
        "disease_id": "MONDO:0005920",
        "disease_name": "Plasmodium falciparum malaria",
        "marker_id": "NCBIGene:1401",
        "marker_type": "gene",
        "marker_name": "CRP",
        "evidence": "PMID:34290484; PubMed query: malaria C-reactive protein biomarker",
    },
    {
        "drug_id": "DB13132",
        "drug_name": "artemisinin",
        "disease_id": "MONDO:0005920",
        "disease_name": "Plasmodium falciparum malaria",
        "marker_id": "NCBIGene:3939",
        "marker_type": "gene",
        "marker_name": "LDHA/lactate dehydrogenase",
        "evidence": "PubMed query: malaria lactate dehydrogenase biomarker",
    },
    {
        "drug_id": "DB01229",
        "drug_name": "paclitaxel",
        "disease_id": "MONDO:0007254",
        "disease_name": "breast cancer",
        "marker_id": "NCBIGene:2064",
        "marker_type": "gene",
        "marker_name": "ERBB2/HER2",
        "evidence": "PMID:11248153; PMID:25564897; PubMed query: paclitaxel breast cancer HER2 biomarker",
    },
    {
        "drug_id": "DB00570",
        "drug_name": "vinblastine",
        "disease_id": "MONDO:0009691",
        "disease_name": "mycosis fungoides",
        "marker_id": "NCBIGene:920",
        "marker_type": "gene",
        "marker_name": "CD4",
        "evidence": "PMID:39093388; PMID:19895171; PMID:17013537; PMID:16844512; PubMed query: mycosis fungoides CD4 CD3 diagnostic biomarker",
    },
    {
        "drug_id": "DB00570",
        "drug_name": "vinblastine",
        "disease_id": "MONDO:0009691",
        "disease_name": "mycosis fungoides",
        "marker_id": "NCBIGene:916",
        "marker_type": "gene",
        "marker_name": "CD3E",
        "evidence": "PMID:39093388; PMID:17496190; PMID:19895171; PMID:17013537; PubMed query: mycosis fungoides CD4 CD3 diagnostic biomarker",
    },
    {
        "drug_id": "DB00570",
        "drug_name": "vinblastine",
        "disease_id": "MONDO:0023067",
        "disease_name": "endemic Kaposi sarcoma",
        "marker_id": "NCBIGene:7422",
        "marker_type": "gene",
        "marker_name": "VEGFA",
        "evidence": "Leaf MONDO subtype of Kaposi sarcoma; PMID:28506734; PMID:30979736; PMID:31252633; PMID:37809059; PubMed query: Kaposi sarcoma VEGF biomarker",
    },
    {
        "drug_id": "DB00570",
        "drug_name": "vinblastine",
        "disease_id": "MONDO:0023067",
        "disease_name": "endemic Kaposi sarcoma",
        "marker_id": "NCBIGene:3569",
        "marker_type": "gene",
        "marker_name": "IL6",
        "evidence": "Leaf MONDO subtype of Kaposi sarcoma; PMID:9171222; PMID:11698289; PMID:26646010; PMID:30265712; PubMed query: Kaposi sarcoma interleukin-6 biomarker",
    },
    {
        "drug_id": "DB00295",
        "drug_name": "morphine",
        "disease_id": "MONDO:0041656",
        "disease_name": "ST-elevation myocardial infarction",
        "marker_id": "NCBIGene:7137",
        "marker_type": "gene",
        "marker_name": "TNNI3/cardiac troponin I",
        "evidence": "PMID:25943557; PMID:36848390; PMID:38344871; PMID:20639847; PubMed query: ST-elevation myocardial infarction troponin I troponin T biomarker",
    },
    {
        "drug_id": "DB00295",
        "drug_name": "morphine",
        "disease_id": "MONDO:0041656",
        "disease_name": "ST-elevation myocardial infarction",
        "marker_id": "NCBIGene:7139",
        "marker_type": "gene",
        "marker_name": "TNNT2/cardiac troponin T",
        "evidence": "PMID:25943557; PMID:36848390; PMID:31269858; PMID:20639847; PubMed query: ST-elevation myocardial infarction troponin I troponin T biomarker",
    },
    {
        "drug_id": "DB00295",
        "drug_name": "morphine",
        "disease_id": "MONDO:0016669",
        "disease_name": "sickle cell-hemoglobin c disease syndrome",
        "marker_id": "NCBIGene:3043",
        "marker_type": "gene",
        "marker_name": "HBB/hemoglobin beta",
        "evidence": "PMID:27802331; PMID:26928604; PMID:8756079; PubMed query: sickle cell hemoglobin C disease HBB biomarker",
    },
    {
        "drug_id": "DB00295",
        "drug_name": "morphine",
        "disease_id": "MONDO:0016669",
        "disease_name": "sickle cell-hemoglobin c disease syndrome",
        "marker_id": "NCBIGene:3040",
        "marker_type": "gene",
        "marker_name": "HBA2/hemoglobin alpha 2",
        "evidence": "PMID:27802331; PMID:26928604; PMID:8756079; PubMed query: sickle cell hemoglobin C disease hemoglobin biomarker",
    },
    {
        "drug_id": "DB09061",
        "drug_name": "cannabidiol",
        "disease_id": "MONDO:0100135",
        "disease_name": "Dravet syndrome",
        "marker_id": "NCBIGene:6323",
        "marker_type": "gene",
        "marker_name": "SCN1A/Nav1.1",
        "evidence": "PMID:35074891; PMID:38229878; PMID:30146492; PMID:27045673; PubMed query: Dravet syndrome SCN1A biomarker",
    },
    {
        "drug_id": "DB00470",
        "drug_name": "dronabinol",
        "disease_id": "MONDO:0005797",
        "disease_name": "HIV wasting syndrome",
        "marker_id": "NCBIGene:920",
        "marker_type": "gene",
        "marker_name": "CD4",
        "evidence": "PMID:9792383; PMID:26264263; PMID:9424041; PMID:26760233; PubMed query: HIV wasting syndrome CD4 count biomarker",
    },
    {
        "drug_id": "DB00470",
        "drug_name": "dronabinol",
        "disease_id": "MONDO:0005314",
        "disease_name": "relapsing-remitting multiple sclerosis",
        "marker_id": "NCBIGene:4747",
        "marker_type": "gene",
        "marker_name": "NEFL/neurofilament light chain",
        "evidence": "PMID:38354532; PMID:28512753; PMID:35182510; PMID:37930670; PubMed query: multiple sclerosis neurofilament light biomarker",
    },
    {
        "drug_id": "DB00470",
        "drug_name": "dronabinol",
        "disease_id": "MONDO:0005314",
        "disease_name": "relapsing-remitting multiple sclerosis",
        "marker_id": "NCBIGene:4155",
        "marker_type": "gene",
        "marker_name": "MBP/myelin basic protein",
        "evidence": "PMID:34917290; PMID:36614334; PMID:32028099; PMID:34889995; PubMed query: multiple sclerosis myelin basic protein biomarker",
    },
]

OFF_LABEL_CURATED_CANDIDATE_PAIRS = [
    {
        "drug_id": "DB13132",
        "drug_name": "artemisinin",
        "disease_id": "MONDO:0016216",
        "disease_name": "adult hepatocellular carcinoma",
        "marker_id": "Uniprot:P02771",
        "marker_type": "protein",
        "marker_name": "AFP/alpha-fetoprotein",
        "drug_disease_evidence": "Off-label/repurposing evidence for artemisinin derivative artesunate in hepatocellular carcinoma: PMID:37733919; PMID:32699265; PMID:32300243; PMID:24987823; PubMed query: artesunate hepatocellular carcinoma",
        "biomarker_evidence": "AFP is an established hepatocellular carcinoma biomarker: PMID:38619448; PMID:39293548; PMID:30665869; PubMed query: alpha-fetoprotein hepatocellular carcinoma biomarker",
    },
    {
        "drug_id": "DB01229",
        "drug_name": "paclitaxel",
        "disease_id": "MONDO:0016682",
        "disease_name": "giant cell glioblastoma",
        "marker_id": "Uniprot:P00533",
        "marker_type": "protein",
        "marker_name": "EGFR",
        "drug_disease_evidence": "Off-label/recurrent glioblastoma evidence for paclitaxel or albumin-bound paclitaxel delivery: PMID:37142373; PMID:38295144; PMID:39713041; PMID:15735921; PMID:10761264; PubMed query: paclitaxel glioblastoma trial",
        "biomarker_evidence": "EGFR is a molecular biomarker/driver in glioblastoma: PMID:30796037; PMID:36012105; PMID:33664496; PMID:33910930; PubMed query: EGFR glioblastoma biomarker",
    },
    {
        "drug_id": "DB01229",
        "drug_name": "paclitaxel",
        "disease_id": "MONDO:0016682",
        "disease_name": "giant cell glioblastoma",
        "marker_id": "Uniprot:P14136",
        "marker_type": "protein",
        "marker_name": "GFAP",
        "drug_disease_evidence": "Off-label/recurrent glioblastoma evidence for paclitaxel or albumin-bound paclitaxel delivery: PMID:37142373; PMID:38295144; PMID:39713041; PMID:15735921; PMID:10761264; PubMed query: paclitaxel glioblastoma trial",
        "biomarker_evidence": "GFAP is a glioblastoma diagnostic/monitoring biomarker: PMID:17998256; PMID:35372061; PMID:38263015; PMID:29053069; PubMed query: GFAP glioblastoma biomarker",
    },
    {
        "drug_id": "DB00570",
        "drug_name": "vinblastine",
        "disease_id": "MONDO:0016682",
        "disease_name": "giant cell glioblastoma",
        "marker_id": "Uniprot:P16949",
        "marker_type": "protein",
        "marker_name": "STMN1/stathmin",
        "drug_disease_evidence": "Off-label/preclinical and case evidence for vinblastine in glioblastoma: PMID:34309720; PMID:30082792; PMID:28852916; PubMed query: vinblastine glioblastoma",
        "biomarker_evidence": "STMN1 is mechanistically linked to vinblastine sensitivity in PDGFRalpha-driven glioblastoma models: PMID:30082792; PubMed query: stathmin glioblastoma biomarker",
    },
    {
        "drug_id": "DB00570",
        "drug_name": "vinblastine",
        "disease_id": "MONDO:0016682",
        "disease_name": "giant cell glioblastoma",
        "marker_id": "Uniprot:P00533",
        "marker_type": "protein",
        "marker_name": "EGFR",
        "drug_disease_evidence": "Off-label/preclinical and case evidence for vinblastine in glioblastoma: PMID:34309720; PMID:30082792; PMID:28852916; PubMed query: vinblastine glioblastoma",
        "biomarker_evidence": "EGFR is a molecular biomarker/driver in glioblastoma: PMID:30796037; PMID:36012105; PMID:33664496; PMID:33910930; PubMed query: EGFR glioblastoma biomarker",
    },
    {
        "drug_id": "DB00570",
        "drug_name": "vinblastine",
        "disease_id": "MONDO:0016682",
        "disease_name": "giant cell glioblastoma",
        "marker_id": "Uniprot:P14136",
        "marker_type": "protein",
        "marker_name": "GFAP",
        "drug_disease_evidence": "Off-label/preclinical and case evidence for vinblastine in glioblastoma: PMID:34309720; PMID:30082792; PMID:28852916; PubMed query: vinblastine glioblastoma",
        "biomarker_evidence": "GFAP is a glioblastoma diagnostic/monitoring biomarker: PMID:17998256; PMID:35372061; PMID:38263015; PMID:29053069; PubMed query: GFAP glioblastoma biomarker",
    },
    {
        "drug_id": "DB00295",
        "drug_name": "morphine",
        "disease_id": "MONDO:0011751",
        "disease_name": "COPD, severe early onset",
        "marker_id": "Uniprot:P02741",
        "marker_type": "protein",
        "marker_name": "CRP/C-reactive protein",
        "drug_disease_evidence": "Off-label/palliative evidence for low-dose morphine in refractory dyspnea in COPD: PMID:32804188; PMID:34507574; PMID:33611087; PMID:26825021; PubMed query: morphine refractory dyspnea COPD",
        "biomarker_evidence": "CRP is used as an inflammatory/exacerbation biomarker in COPD: PMID:30880285; PMID:26451097; PMID:35549921; PubMed query: CRP COPD biomarker",
    },
    {
        "drug_id": "DB00295",
        "drug_name": "morphine",
        "disease_id": "MONDO:0011751",
        "disease_name": "COPD, severe early onset",
        "marker_id": "Uniprot:P05231",
        "marker_type": "protein",
        "marker_name": "IL6/interleukin-6",
        "drug_disease_evidence": "Off-label/palliative evidence for low-dose morphine in refractory dyspnea in COPD: PMID:32804188; PMID:34507574; PMID:33611087; PMID:26825021; PubMed query: morphine refractory dyspnea COPD",
        "biomarker_evidence": "IL6 is an inflammatory biomarker studied in COPD phenotyping/exacerbation contexts; PubMed query: interleukin-6 COPD biomarker",
    },
    {
        "drug_id": "DB09061",
        "drug_name": "cannabidiol",
        "disease_id": "MONDO:0016682",
        "disease_name": "giant cell glioblastoma",
        "marker_id": "Uniprot:P00533",
        "marker_type": "protein",
        "marker_name": "EGFR",
        "drug_disease_evidence": "Off-label/preclinical and adjunctive evidence for cannabidiol in glioblastoma: PMID:33629929; PMID:34918964; PMID:35403130; PMID:35356807; PMID:38583854; PubMed query: cannabidiol glioblastoma",
        "biomarker_evidence": "EGFR is a molecular biomarker/driver in glioblastoma: PMID:30796037; PMID:36012105; PMID:33664496; PMID:33910930; PubMed query: EGFR glioblastoma biomarker",
    },
    {
        "drug_id": "DB09061",
        "drug_name": "cannabidiol",
        "disease_id": "MONDO:0016682",
        "disease_name": "giant cell glioblastoma",
        "marker_id": "Uniprot:P14136",
        "marker_type": "protein",
        "marker_name": "GFAP",
        "drug_disease_evidence": "Off-label/preclinical and adjunctive evidence for cannabidiol in glioblastoma: PMID:33629929; PMID:34918964; PMID:35403130; PMID:35356807; PMID:38583854; PubMed query: cannabidiol glioblastoma",
        "biomarker_evidence": "GFAP is a glioblastoma diagnostic/monitoring biomarker: PMID:17998256; PMID:35372061; PMID:38263015; PMID:29053069; PubMed query: GFAP glioblastoma biomarker",
    },
    {
        "drug_id": "DB00470",
        "drug_name": "dronabinol",
        "disease_id": "MONDO:0005314",
        "disease_name": "relapsing-remitting multiple sclerosis",
        "marker_id": "Uniprot:P07196",
        "marker_type": "protein",
        "marker_name": "NEFL/neurofilament light chain",
        "drug_disease_evidence": "Off-label/symptomatic cannabinoid evidence in multiple sclerosis: PMID:15258006; PMID:22791906; PMID:23856559; PMID:35510826; PMID:12182963; PubMed query: dronabinol multiple sclerosis trial",
        "biomarker_evidence": "Neurofilament light chain is a biomarker of neuroaxonal injury and progression in relapsing MS: PMID:37354600; PMID:37930670; PMID:36032114; PubMed query: neurofilament light relapsing-remitting multiple sclerosis biomarker",
    },
    {
        "drug_id": "DB00470",
        "drug_name": "dronabinol",
        "disease_id": "MONDO:0005314",
        "disease_name": "relapsing-remitting multiple sclerosis",
        "marker_id": "Uniprot:P02686",
        "marker_type": "protein",
        "marker_name": "MBP/myelin basic protein",
        "drug_disease_evidence": "Off-label/symptomatic cannabinoid evidence in multiple sclerosis: PMID:15258006; PMID:22791906; PMID:23856559; PMID:35510826; PMID:12182963; PubMed query: dronabinol multiple sclerosis trial",
        "biomarker_evidence": "MBP/myelin basic protein is a myelin-related biomarker studied in multiple sclerosis: PMID:34917290; PMID:36614334; PMID:32028099; PMID:34889995; PubMed query: multiple sclerosis myelin basic protein biomarker",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find edges on Disease->Drug->marker paths within max hops."
    )
    parser.add_argument(
        "--drug-id",
        action="append",
        default=None,
        help="DrugBank ID，可重复传入；不传则使用脚本内置的药物候选 ID。",
    )
    parser.add_argument(
        "--target-type",
        required=True,
        choices=sorted(TARGET_TYPE_ALIASES.keys()),
        help="路径另一端的节点类型；DNA 会按 gene 节点处理。",
    )
    parser.add_argument(
        "--candidate-mode",
        choices=("curated", "offlabel"),
        default="curated",
        help="候选来源模式：curated 使用原有文献候选；offlabel 要求外部证据支持 D 可治疗 M，且图中不存在 M-D/M-N 直接边。",
    )
    parser.add_argument("--max-hops", default=None, type=int, help="路径最大跳数/边数。")
    parser.add_argument(
        "--max-nodes",
        default=None,
        type=int,
        help="路径最多经过的节点数；传入后内部换算为 max_nodes - 1 跳。",
    )
    parser.add_argument(
        "--edge-file",
        default=str(DEFAULT_EDGE_FILE),
        help="边文件路径，默认 disease_connect/merged_edge.csv。",
    )
    parser.add_argument(
        "--marker-edge-file",
        default=str(DEFAULT_MARKER_EDGE_FILE),
        help="已知 disease-marker 边文件路径，默认 disease_connect/marker_edge.csv。",
    )
    parser.add_argument(
        "--mondo-file",
        default=str(DEFAULT_MONDO_FILE),
        help="MONDO OBO 文件路径，用于过滤 leaf disease，默认 mondo.obo。",
    )
    parser.add_argument(
        "--directed",
        action="store_true",
        help="按 Edge Direction 作为有向图处理；默认按无向图处理。",
    )
    parser.add_argument(
        "--require-go-node",
        dest="require_go_node",
        action="store_true",
        default=True,
        help="要求输出路径至少包含一个 GO BP/MF/CC 节点；默认开启。",
    )
    parser.add_argument(
        "--no-require-go-node",
        dest="require_go_node",
        action="store_false",
        help="不要求输出路径包含 GO BP/MF/CC 节点。",
    )
    parser.add_argument(
        "--allow-disease-after-drug",
        action="store_true",
        help="允许路径经过给定药物后再次进入 disease 节点；默认禁止以避免拓扑循环。",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="输出文件前缀；不传则自动生成。",
    )
    parser.add_argument(
        "--max-example-nodes",
        default=20,
        type=int,
        help="兼容旧参数；summary 中 example_disease_ids/example_target_ids 现在总是输出全部节点。",
    )
    parser.add_argument(
        "--max-output-paths",
        default=0,
        type=int,
        help="最多输出多少条完整路径；0 表示不限制。",
    )
    parser.add_argument(
        "--max-output-edges",
        default=0,
        type=int,
        help="最多输出多少条边；0 表示不限制。",
    )
    return parser.parse_args()


def resolve_max_hops(args: argparse.Namespace) -> int:
    if args.max_nodes is None and args.max_hops is None:
        raise ValueError("必须提供 --max-nodes 或 --max-hops")
    if args.max_nodes is not None:
        if args.max_nodes < 2:
            raise ValueError("--max-nodes 必须 >= 2")
        return args.max_nodes - 1
    if args.max_hops < 1:
        raise ValueError("--max-hops 必须 >= 1")
    return args.max_hops


def normalize_type(node_type: str) -> str:
    return (node_type or "").strip().lower()


def canonical_node_id(node_id: str, node_type: str = "", source: str = "") -> str:
    node_id = (node_id or "").strip()
    node_type = normalize_type(node_type)
    source = (source or "").strip().lower()

    if node_type == "protein" and source == "uniprot" and node_id and not node_id.startswith("Uniprot:"):
        return f"Uniprot:{node_id}"
    if node_type in {"gene", "dna"} and source == "ncbi" and node_id and not node_id.startswith("NCBIGene:"):
        return f"NCBIGene:{node_id}"
    if node_type == "disease" and source == "mondo" and node_id and not node_id.startswith("MONDO:"):
        return f"MONDO:{node_id}"
    return node_id


def markerdb_pair_key(disease_id: str, marker_id: str):
    return disease_id, marker_id


def markerdb_edge_key(node1: str, node2: str):
    return frozenset((node1, node2))


def get_oriented_arcs(node1: str, node2: str, edge_direction: str, directed: bool):
    if not directed:
        return [(node1, node2), (node2, node1)]

    direction = (edge_direction or "").strip().lower()
    if "double" in direction or "bi" in direction:
        return [(node1, node2), (node2, node1)]
    if "reverse" in direction:
        return [(node2, node1)]
    if "single" in direction or "direct" in direction or "forward" in direction:
        return [(node1, node2)]

    return [(node1, node2), (node2, node1)]


def expand_drug_queries(drug_ids):
    if drug_ids:
        expanded = []
        for value in drug_ids:
            for drug_id in value.split(","):
                drug_id = drug_id.strip()
                if drug_id:
                    expanded.append((drug_id, drug_id))
        return expanded

    expanded = []
    for drug_name, ids in DEFAULT_DRUGS.items():
        for drug_id in ids:
            expanded.append((drug_name, drug_id))
    return expanded


def build_graph_and_edges(edge_file: str, directed: bool, blocked_edge_keys=None):
    edges = []
    graph = defaultdict(list)
    reverse_graph = defaultdict(list)
    node_type_map = {}
    direct_edge_sources = defaultdict(set)
    blocked_edge_keys = blocked_edge_keys or set()
    blocked_edge_count = 0

    csv.field_size_limit(1024 * 1024 * 1024)
    with open(edge_file, "r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        missing = [col for col in FIELDNAMES_EDGE if col not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"边文件缺少必要列: {missing}")

        for row in reader:
            node1_type = row.get("Node 1 Data Type", "")
            node2_type = row.get("Node 2 Data Type", "")
            node1 = canonical_node_id(row["Node 1 ID"], node1_type, row.get("Node 1 Source", ""))
            node2 = canonical_node_id(row["Node 2 ID"], node2_type, row.get("Node 2 Source", ""))
            node_type_map[node1] = normalize_type(node1_type)
            node_type_map[node2] = normalize_type(node2_type)

            edge_idx = len(edges)
            edges.append(tuple(row.get(col, "") for col in FIELDNAMES_EDGE))
            direct_edge_sources[markerdb_edge_key(node1, node2)].add(row.get("Edge Type", ""))

            if markerdb_edge_key(node1, node2) in blocked_edge_keys:
                blocked_edge_count += 1
                continue

            for src, dst in get_oriented_arcs(node1, node2, row.get("Edge Direction", ""), directed):
                graph[src].append((dst, edge_idx))
                reverse_graph[dst].append((src, edge_idx))

    return edges, graph, reverse_graph, node_type_map, blocked_edge_count, direct_edge_sources


def load_markerdb_pairs(marker_edge_file: str, target_type: str = None):
    target_types = TARGET_TYPE_ALIASES[target_type] if target_type else None
    disease_to_markers = defaultdict(set)
    marker_to_diseases = defaultdict(set)
    pair_sources = defaultdict(set)
    blocked_edge_keys = set()

    csv.field_size_limit(1024 * 1024 * 1024)
    with open(marker_edge_file, "r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        missing = [col for col in FIELDNAMES_EDGE if col not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"已知 marker 边文件缺少必要列: {missing}")

        for row in reader:
            type1 = normalize_type(row.get("Node 1 Data Type", ""))
            type2 = normalize_type(row.get("Node 2 Data Type", ""))
            node1 = canonical_node_id(row["Node 1 ID"], type1, row.get("Node 1 Source", ""))
            node2 = canonical_node_id(row["Node 2 ID"], type2, row.get("Node 2 Source", ""))

            disease_id = None
            marker_id = None
            marker_type = None
            if type1 == "disease" and (target_types is None or type2 in target_types):
                disease_id = node1
                marker_id = node2
                marker_type = type2
            elif type2 == "disease" and (target_types is None or type1 in target_types):
                disease_id = node2
                marker_id = node1
                marker_type = type1

            if disease_id and marker_id:
                disease_to_markers[disease_id].add(marker_id)
                marker_to_diseases[marker_id].add(disease_id)
                pair_sources[markerdb_pair_key(disease_id, marker_id)].add(
                    row.get("Edge Type", "known_marker") + f":{marker_type}"
                )
                blocked_edge_keys.add(markerdb_edge_key(disease_id, marker_id))

    return disease_to_markers, marker_to_diseases, pair_sources, blocked_edge_keys


def load_mondo_leaf_disease_ids(mondo_file: str):
    terms = {}
    current = None

    def flush_current():
        if current and current.get("id"):
            terms[current["id"]] = {
                "parents": set(current["parents"]),
                "obsolete": current["obsolete"],
            }

    with open(mondo_file, "r", encoding="utf-8") as infile:
        for raw_line in infile:
            line = raw_line.strip()
            if line == "[Term]":
                flush_current()
                current = {"id": None, "parents": [], "obsolete": False}
                continue
            if line.startswith("["):
                flush_current()
                current = None
                continue
            if current is None:
                continue
            if line.startswith("id: MONDO:"):
                current["id"] = line.split()[1]
            elif line.startswith("is_a: MONDO:"):
                current["parents"].append(line.split()[1])
            elif line == "is_obsolete: true":
                current["obsolete"] = True

    flush_current()

    active_ids = {term_id for term_id, term in terms.items() if not term["obsolete"]}
    child_counts = defaultdict(int)
    for term_id, term in terms.items():
        if term_id not in active_ids:
            continue
        for parent_id in term["parents"]:
            if parent_id in active_ids:
                child_counts[parent_id] += 1

    leaf_ids = {term_id for term_id in active_ids if child_counts[term_id] == 0}
    return leaf_ids, active_ids, child_counts


def select_web_curated_candidate_pairs(
    drug_id: str,
    target_type: str,
    markerdb_pair_sources,
    mondo_leaf_disease_ids=None,
    candidate_mode: str = "curated",
    direct_edge_sources=None,
):
    target_types = TARGET_TYPE_ALIASES[target_type]
    source_records = OFF_LABEL_CURATED_CANDIDATE_PAIRS if candidate_mode == "offlabel" else WEB_CURATED_CANDIDATE_PAIRS
    candidate_pairs = {}
    raw_count = 0
    markerdb_excluded = []
    non_leaf_disease_excluded = []
    direct_drug_disease_excluded = []
    direct_disease_marker_excluded = []
    direct_edge_sources = direct_edge_sources or {}

    for record in source_records:
        if record["drug_id"] != drug_id:
            continue
        if normalize_type(record["marker_type"]) not in target_types:
            continue

        raw_count += 1
        disease_id = canonical_node_id(record["disease_id"], "disease", "MONDO")
        marker_id = canonical_node_id(record["marker_id"], record["marker_type"], "")
        pair_key = markerdb_pair_key(disease_id, marker_id)
        normalized_record = dict(record)
        normalized_record["disease_id"] = disease_id
        normalized_record["marker_id"] = marker_id
        if "evidence" not in normalized_record:
            normalized_record["evidence"] = " | ".join(
                value
                for value in (
                    normalized_record.get("drug_disease_evidence", ""),
                    normalized_record.get("biomarker_evidence", ""),
                )
                if value
            )

        if mondo_leaf_disease_ids is not None and disease_id not in mondo_leaf_disease_ids:
            non_leaf_disease_excluded.append(normalized_record)
            continue
        direct_drug_disease_edges = sorted(direct_edge_sources.get(markerdb_edge_key(drug_id, disease_id), []))
        direct_disease_marker_edges = sorted(direct_edge_sources.get(markerdb_edge_key(disease_id, marker_id), []))
        if candidate_mode == "offlabel" and direct_drug_disease_edges:
            excluded_record = dict(normalized_record)
            excluded_record["direct_drug_disease_edge_types"] = "|".join(direct_drug_disease_edges)
            direct_drug_disease_excluded.append(excluded_record)
            continue
        if candidate_mode == "offlabel" and direct_disease_marker_edges:
            excluded_record = dict(normalized_record)
            excluded_record["direct_disease_marker_edge_types"] = "|".join(direct_disease_marker_edges)
            direct_disease_marker_excluded.append(excluded_record)
            continue
        if pair_key in markerdb_pair_sources:
            markerdb_excluded.append(normalized_record)
            continue
        normalized_record["direct_drug_disease_edge_types"] = "|".join(direct_drug_disease_edges)
        normalized_record["direct_disease_marker_edge_types"] = "|".join(direct_disease_marker_edges)
        candidate_pairs[pair_key] = normalized_record

    disease_to_markers = defaultdict(set)
    for disease_id, marker_id in candidate_pairs:
        disease_to_markers[disease_id].add(marker_id)

    return (
        disease_to_markers,
        candidate_pairs,
        raw_count,
        markerdb_excluded,
        non_leaf_disease_excluded,
        direct_drug_disease_excluded,
        direct_disease_marker_excluded,
    )


def bfs_dist(graph, starts, max_hops: int):
    dist = {}
    queue = deque()

    for start in starts:
        if start in dist:
            continue
        dist[start] = 0
        queue.append(start)

    while queue:
        node = queue.popleft()
        if dist[node] >= max_hops:
            continue
        for nxt, _ in graph.get(node, []):
            if nxt not in dist:
                dist[nxt] = dist[node] + 1
                queue.append(nxt)

    return dist


def is_forward_progress(dist_from_source, dist_to_target, src: str, dst: str) -> bool:
    src_from = dist_from_source.get(src)
    dst_from = dist_from_source.get(dst)
    src_to = dist_to_target.get(src)
    dst_to = dist_to_target.get(dst)

    if src_from is None or dst_from is None or src_to is None or dst_to is None:
        return False
    if dst_from < src_from:
        return False
    if dst_from == src_from and src_to <= dst_to:
        return False
    return True


def record_selected(selected, edge_idx: int, side: str, src: str, dst: str):
    entry = selected.setdefault(edge_idx, {"sides": set(), "arcs": set()})
    entry["sides"].add(side)
    entry["arcs"].add(f"{src}->{dst}")


def enumerate_paths_to_node(start_nodes, target_node: str, graph, dist_to_target, max_depth: int):
    paths = []

    for start_node in sorted(start_nodes):
        if dist_to_target.get(start_node) is None or dist_to_target[start_node] > max_depth:
            continue

        stack = [(start_node, [start_node], [])]
        while stack:
            node, path_nodes, path_edges = stack.pop()
            depth = len(path_edges)

            if node == target_node:
                paths.append((path_nodes, path_edges))
                continue
            if depth >= max_depth:
                continue

            remaining_after_next = max_depth - depth - 1
            for next_node, edge_idx in graph.get(node, []):
                if next_node in path_nodes:
                    continue
                next_distance = dist_to_target.get(next_node)
                if next_distance is None or next_distance > remaining_after_next:
                    continue
                stack.append((next_node, path_nodes + [next_node], path_edges + [edge_idx]))

    return paths


def enumerate_paths_to_any_target(start_node: str, target_nodes, graph, dist_to_targets, max_depth: int):
    paths = []
    target_node_set = set(target_nodes)
    stack = [(start_node, [start_node], [])]

    while stack:
        node, path_nodes, path_edges = stack.pop()
        depth = len(path_edges)

        if depth > 0 and node in target_node_set:
            paths.append((path_nodes, path_edges))
        if depth >= max_depth:
            continue

        remaining_after_next = max_depth - depth - 1
        for next_node, edge_idx in graph.get(node, []):
            if next_node in path_nodes:
                continue
            next_distance = dist_to_targets.get(next_node)
            if next_distance is None or next_distance > remaining_after_next:
                continue
            stack.append((next_node, path_nodes + [next_node], path_edges + [edge_idx]))

    return paths


def combine_complete_paths(left_paths, right_paths, max_hops: int, max_output_paths: int):
    combined_paths = []
    truncated = False

    for left_nodes, left_edges in left_paths:
        left_internal_nodes = set(left_nodes[:-1])
        for right_nodes, right_edges in right_paths:
            path_hops = len(left_edges) + len(right_edges)
            if path_hops > max_hops:
                continue
            if left_internal_nodes.intersection(right_nodes[1:]):
                continue

            combined_paths.append(
                {
                    "nodes": left_nodes + right_nodes[1:],
                    "edges": left_edges + right_edges,
                    "hops": path_hops,
                    "disease_id": left_nodes[0],
                    "target_id": right_nodes[-1],
                }
            )

            if max_output_paths > 0 and len(combined_paths) >= max_output_paths:
                truncated = True
                return combined_paths, truncated

    return combined_paths, truncated


def trace_state_to_root(state, parent):
    nodes = [state[0]]
    edges = []
    current_state = state
    while current_state in parent:
        parent_state, edge_idx = parent[current_state]
        edges.append(edge_idx)
        current_state = parent_state
        nodes.append(current_state[0])
    return nodes, edges


def shortest_path_through_required_nodes(
    graph,
    start_node: str,
    target_node: str,
    max_hops: int,
    node_type_map,
    required_node_ids=None,
    require_go_node: bool = False,
    block_disease_after_node_ids=None,
    dist_to_target=None,
    anchor_node_id: str = None,
    dist_to_anchor=None,
):
    required_node_ids = frozenset(required_node_ids or [])
    block_disease_after_node_ids = frozenset(block_disease_after_node_ids or [])

    start_seen_required = frozenset(node_id for node_id in required_node_ids if node_id == start_node)
    start_has_go = node_type_map.get(start_node) in GO_NODE_TYPES
    start_state = (start_node, start_seen_required, start_has_go)
    dist = {start_state: 0}
    parent = {}
    queue = deque([start_state])

    def is_goal_state(node, seen_required, has_go) -> bool:
        return node == target_node and required_node_ids.issubset(seen_required) and (
            has_go or not require_go_node
        )

    def can_still_finish(next_node, next_depth: int, next_seen_required) -> bool:
        if anchor_node_id and anchor_node_id in required_node_ids and anchor_node_id not in next_seen_required:
            if dist_to_anchor is None or dist_to_target is None:
                return True
            next_to_anchor = dist_to_anchor.get(next_node)
            anchor_to_target = dist_to_target.get(anchor_node_id)
            return (
                next_to_anchor is not None
                and anchor_to_target is not None
                and next_depth + next_to_anchor + anchor_to_target <= max_hops
            )

        if dist_to_target is None:
            return True
        next_to_target = dist_to_target.get(next_node)
        return next_to_target is not None and next_depth + next_to_target <= max_hops

    while queue:
        state = queue.popleft()
        node, seen_required, has_go = state
        current_dist = dist[state]

        if is_goal_state(node, seen_required, has_go):
            nodes, edges = trace_state_to_root(state, parent)
            return list(reversed(nodes)), list(reversed(edges))
        if node == target_node or current_dist >= max_hops:
            continue

        seen_blocking_node = bool(seen_required.intersection(block_disease_after_node_ids))
        for next_node, edge_idx in graph.get(node, []):
            if next_node == start_node:
                continue
            if seen_blocking_node and next_node != target_node and node_type_map.get(next_node) == "disease":
                continue

            next_seen_required = seen_required
            if next_node in required_node_ids:
                next_seen_required = frozenset(set(seen_required) | {next_node})
            next_has_go = has_go or node_type_map.get(next_node) in GO_NODE_TYPES
            if next_node == target_node and not is_goal_state(next_node, next_seen_required, next_has_go):
                continue
            next_depth = current_dist + 1
            if not can_still_finish(next_node, next_depth, next_seen_required):
                continue

            next_state = (next_node, next_seen_required, next_has_go)
            if next_state in dist:
                continue
            dist[next_state] = next_depth
            parent[next_state] = (state, edge_idx)
            queue.append(next_state)

    return None


def collect_selected_edges_from_paths(paths):
    selected = {}
    for path_record in paths:
        path_nodes = path_record["nodes"]
        for position, edge_idx in enumerate(path_record["edges"]):
            src = path_nodes[position]
            dst = path_nodes[position + 1]
            record_selected(selected, edge_idx, "witness_path", src, dst)
    return selected


def find_edges_for_drug(
    drug_id: str,
    target_type: str,
    max_hops: int,
    max_output_paths: int,
    graph,
    reverse_graph,
    node_type_map,
    markerdb_pair_sources,
    mondo_leaf_disease_ids,
    require_go_node,
    allow_disease_after_drug,
    candidate_mode,
    direct_edge_sources,
):
    (
        disease_to_markers,
        candidate_pairs,
        raw_candidate_count,
        markerdb_excluded,
        non_leaf_disease_excluded,
        direct_drug_disease_excluded,
        direct_disease_marker_excluded,
    ) = select_web_curated_candidate_pairs(
        drug_id,
        target_type,
        markerdb_pair_sources,
        mondo_leaf_disease_ids,
        candidate_mode,
        direct_edge_sources,
    )
    drug_present = drug_id in node_type_map or drug_id in graph
    if not drug_present or not disease_to_markers:
        return {
            "selected": {},
            "valid_diseases": set(),
            "valid_targets": set(),
            "min_disease_distance": None,
            "min_target_distance": None,
            "paths": [],
            "paths_truncated": False,
            "raw_candidate_pair_count": raw_candidate_count,
            "markerdb_excluded_pair_count": len(markerdb_excluded),
            "non_leaf_disease_excluded_pair_count": len(non_leaf_disease_excluded),
            "direct_drug_disease_excluded_pair_count": len(direct_drug_disease_excluded),
            "direct_disease_marker_excluded_pair_count": len(direct_disease_marker_excluded),
            "candidate_pair_count": len(candidate_pairs),
            "excluded_markerdb_pairs": markerdb_excluded,
            "excluded_non_leaf_disease_pairs": non_leaf_disease_excluded,
            "excluded_direct_drug_disease_pairs": direct_drug_disease_excluded,
            "excluded_direct_disease_marker_pairs": direct_disease_marker_excluded,
            "missing_candidate_pair_count": 0,
            "drug_present": drug_present,
        }

    dist_to_drug = bfs_dist(reverse_graph, [drug_id], max_hops)
    dist_to_target_cache = {}
    paths = []
    paths_truncated = False
    min_disease_distance = None
    min_target_distance = None
    missing_candidate_pair_count = 0

    for disease_id in sorted(disease_to_markers):
        for marker_id in sorted(disease_to_markers[disease_id]):
            if disease_id not in node_type_map or marker_id not in node_type_map:
                missing_candidate_pair_count += 1
                continue
            if dist_to_drug.get(disease_id) is None:
                continue

            if marker_id not in dist_to_target_cache:
                dist_to_target_cache[marker_id] = bfs_dist(reverse_graph, [marker_id], max_hops)
            dist_to_target = dist_to_target_cache[marker_id]
            if dist_to_target.get(drug_id) is None:
                continue

            shortest_path = shortest_path_through_required_nodes(
                graph,
                disease_id,
                marker_id,
                max_hops,
                node_type_map,
                required_node_ids={drug_id},
                require_go_node=require_go_node,
                block_disease_after_node_ids=set() if allow_disease_after_drug else {drug_id},
                dist_to_target=dist_to_target,
                anchor_node_id=drug_id,
                dist_to_anchor=dist_to_drug,
            )
            if shortest_path is None:
                continue

            path_nodes, path_edges = shortest_path
            total_hops = len(path_edges)
            drug_position = path_nodes.index(drug_id)
            left_distance = drug_position
            right_distance = total_hops - drug_position
            go_node_ids = [node for node in path_nodes if node_type_map.get(node) in GO_NODE_TYPES]

            paths.append(
                {
                    "nodes": path_nodes,
                    "edges": path_edges,
                    "hops": total_hops,
                    "disease_id": disease_id,
                    "target_id": marker_id,
                    "candidate": candidate_pairs.get((disease_id, marker_id), {}),
                    "disease_to_drug_hops": left_distance,
                    "drug_to_target_hops": right_distance,
                    "go_node_ids": go_node_ids,
                }
            )
            min_disease_distance = (
                left_distance if min_disease_distance is None else min(min_disease_distance, left_distance)
            )
            min_target_distance = (
                right_distance if min_target_distance is None else min(min_target_distance, right_distance)
            )

            if max_output_paths > 0 and len(paths) >= max_output_paths:
                paths_truncated = True
                break
        if paths_truncated:
            break

    valid_diseases = {path_record["disease_id"] for path_record in paths}
    valid_targets = {path_record["target_id"] for path_record in paths}
    selected = collect_selected_edges_from_paths(paths)

    return {
        "selected": selected,
        "valid_diseases": valid_diseases,
        "valid_targets": valid_targets,
        "min_disease_distance": min_disease_distance,
        "min_target_distance": min_target_distance,
        "paths": paths,
        "paths_truncated": paths_truncated,
        "raw_candidate_pair_count": raw_candidate_count,
        "markerdb_excluded_pair_count": len(markerdb_excluded),
        "non_leaf_disease_excluded_pair_count": len(non_leaf_disease_excluded),
        "direct_drug_disease_excluded_pair_count": len(direct_drug_disease_excluded),
        "direct_disease_marker_excluded_pair_count": len(direct_disease_marker_excluded),
        "candidate_pair_count": len(candidate_pairs),
        "excluded_markerdb_pairs": markerdb_excluded,
        "excluded_non_leaf_disease_pairs": non_leaf_disease_excluded,
        "excluded_direct_drug_disease_pairs": direct_drug_disease_excluded,
        "excluded_direct_disease_marker_pairs": direct_disease_marker_excluded,
        "missing_candidate_pair_count": missing_candidate_pair_count,
        "drug_present": True,
    }


def make_output_paths(args: argparse.Namespace):
    if args.output_prefix:
        prefix = Path(args.output_prefix)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = Path(f"drug_connected_{args.target_type}_h{args.max_hops}_{timestamp}")

    return (
        prefix.with_name(prefix.name + "_paths.csv"),
        prefix.with_name(prefix.name + "_edges.csv"),
        prefix.with_name(prefix.name + "_summary.csv"),
    )


def join_all_ids(nodes) -> str:
    return "|".join(sorted(nodes))


def format_path(path_nodes, node_type_map) -> str:
    return " -> ".join(f"{node_id}({node_type_map.get(node_id, 'unknown')})" for node_id in path_nodes)


def write_outputs(path_output, edge_output, summary_output, args, drug_results, edges, node_type_map):
    path_fieldnames = [
        "query_drug_name",
        "drug_id",
        "candidate_mode",
        "target_type",
        "path_index",
        "path_hops",
        "disease_to_drug_hops",
        "drug_to_target_hops",
        "disease_id",
        "disease_name",
        "target_id",
        "target_name",
        "candidate_evidence",
        "drug_disease_evidence",
        "biomarker_evidence",
        "direct_drug_disease_edge_types",
        "direct_disease_marker_edge_types",
        "go_node_ids",
        "go_node_types",
        "path_node_ids",
        "path_node_types",
        "path_edge_indices",
        "path",
    ]
    edge_fieldnames = [
        "query_drug_name",
        "drug_id",
        "candidate_mode",
        "target_type",
        "path_side",
        "usable_arcs",
        "edge_index",
    ] + FIELDNAMES_EDGE
    summary_fieldnames = [
        "query_drug_name",
        "drug_id",
        "candidate_mode",
        "target_type",
        "marker_edge_file",
        "mondo_file",
        "max_hops",
        "max_nodes",
        "directed",
        "require_go_node",
        "allow_disease_after_drug",
        "drug_present_in_graph",
        "raw_web_candidate_pair_count",
        "markerdb_excluded_pair_count",
        "non_leaf_disease_excluded_pair_count",
        "direct_drug_disease_excluded_pair_count",
        "direct_disease_marker_excluded_pair_count",
        "candidate_pair_count_after_markerdb_exclusion",
        "candidate_pair_count_after_all_exclusions",
        "missing_candidate_pair_count",
        "min_disease_distance_to_drug",
        "min_drug_to_target_distance",
        "candidate_disease_count",
        "candidate_target_count",
        "complete_path_count",
        "paths_truncated",
        "selected_edge_count",
        "selected_arc_count",
        "example_disease_ids",
        "example_target_ids",
    ]

    with open(path_output, "w", encoding="utf-8", newline="") as path_file:
        writer = csv.DictWriter(path_file, fieldnames=path_fieldnames)
        writer.writeheader()

        for query_name, drug_id, result in drug_results:
            for path_index, path_record in enumerate(result["paths"], start=1):
                path_nodes = path_record["nodes"]
                path_node_types = [node_type_map.get(node_id, "unknown") for node_id in path_nodes]
                go_node_ids = path_record.get("go_node_ids", [])
                go_node_types = [node_type_map.get(node_id, "unknown") for node_id in go_node_ids]
                candidate = path_record.get("candidate", {})
                writer.writerow(
                    {
                        "query_drug_name": query_name,
                        "drug_id": drug_id,
                        "candidate_mode": args.candidate_mode,
                        "target_type": args.target_type,
                        "path_index": path_index,
                        "path_hops": path_record["hops"],
                        "disease_to_drug_hops": path_record.get("disease_to_drug_hops"),
                        "drug_to_target_hops": path_record.get("drug_to_target_hops"),
                        "disease_id": path_record["disease_id"],
                        "disease_name": candidate.get("disease_name", ""),
                        "target_id": path_record["target_id"],
                        "target_name": candidate.get("marker_name", ""),
                        "candidate_evidence": candidate.get("evidence", ""),
                        "drug_disease_evidence": candidate.get("drug_disease_evidence", ""),
                        "biomarker_evidence": candidate.get("biomarker_evidence", ""),
                        "direct_drug_disease_edge_types": candidate.get("direct_drug_disease_edge_types", ""),
                        "direct_disease_marker_edge_types": candidate.get("direct_disease_marker_edge_types", ""),
                        "go_node_ids": "|".join(go_node_ids),
                        "go_node_types": "|".join(go_node_types),
                        "path_node_ids": "|".join(path_nodes),
                        "path_node_types": "|".join(path_node_types),
                        "path_edge_indices": "|".join(str(edge_idx) for edge_idx in path_record["edges"]),
                        "path": format_path(path_nodes, node_type_map),
                    }
                )

    with open(edge_output, "w", encoding="utf-8", newline="") as edge_file:
        writer = csv.DictWriter(edge_file, fieldnames=edge_fieldnames)
        writer.writeheader()

        for query_name, drug_id, result in drug_results:
            selected_items = sorted(result["selected"].items())
            if args.max_output_edges > 0:
                selected_items = selected_items[: args.max_output_edges]

            for edge_idx, info in selected_items:
                edge_values = dict(zip(FIELDNAMES_EDGE, edges[edge_idx]))
                writer.writerow(
                    {
                        "query_drug_name": query_name,
                        "drug_id": drug_id,
                        "candidate_mode": args.candidate_mode,
                        "target_type": args.target_type,
                        "path_side": "|".join(sorted(info["sides"])),
                        "usable_arcs": "|".join(sorted(info["arcs"])),
                        "edge_index": edge_idx,
                        **edge_values,
                    }
                )

    with open(summary_output, "w", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=summary_fieldnames)
        writer.writeheader()

        for query_name, drug_id, result in drug_results:
            selected_arc_count = sum(len(info["arcs"]) for info in result["selected"].values())
            writer.writerow(
                {
                    "query_drug_name": query_name,
                    "drug_id": drug_id,
                    "candidate_mode": args.candidate_mode,
                    "target_type": args.target_type,
                    "marker_edge_file": args.marker_edge_file,
                    "mondo_file": args.mondo_file,
                    "max_hops": args.max_hops,
                    "max_nodes": args.max_hops + 1,
                    "directed": args.directed,
                    "require_go_node": args.require_go_node,
                    "allow_disease_after_drug": args.allow_disease_after_drug,
                    "drug_present_in_graph": result["drug_present"],
                    "raw_web_candidate_pair_count": result["raw_candidate_pair_count"],
                    "markerdb_excluded_pair_count": result["markerdb_excluded_pair_count"],
                    "non_leaf_disease_excluded_pair_count": result["non_leaf_disease_excluded_pair_count"],
                    "direct_drug_disease_excluded_pair_count": result["direct_drug_disease_excluded_pair_count"],
                    "direct_disease_marker_excluded_pair_count": result["direct_disease_marker_excluded_pair_count"],
                    "candidate_pair_count_after_markerdb_exclusion": result["candidate_pair_count"],
                    "candidate_pair_count_after_all_exclusions": result["candidate_pair_count"],
                    "missing_candidate_pair_count": result["missing_candidate_pair_count"],
                    "min_disease_distance_to_drug": result["min_disease_distance"],
                    "min_drug_to_target_distance": result["min_target_distance"],
                    "candidate_disease_count": len(result["valid_diseases"]),
                    "candidate_target_count": len(result["valid_targets"]),
                    "complete_path_count": len(result["paths"]),
                    "paths_truncated": result["paths_truncated"],
                    "selected_edge_count": len(result["selected"]),
                    "selected_arc_count": selected_arc_count,
                    "example_disease_ids": join_all_ids(result["valid_diseases"]),
                    "example_target_ids": join_all_ids(result["valid_targets"]),
                }
            )


def main():
    args = parse_args()
    args.max_hops = resolve_max_hops(args)
    if args.candidate_mode == "offlabel" and args.target_type != "protein":
        raise ValueError("--candidate-mode offlabel 目前只支持 --target-type protein")
    if args.max_output_paths < 0:
        raise ValueError("--max-output-paths 必须 >= 0")
    if args.max_output_edges < 0:
        raise ValueError("--max-output-edges 必须 >= 0")
    if not os.path.exists(args.edge_file):
        raise FileNotFoundError(f"找不到边文件: {args.edge_file}")
    if not os.path.exists(args.marker_edge_file):
        raise FileNotFoundError(f"找不到已知 marker 边文件: {args.marker_edge_file}")
    if not os.path.exists(args.mondo_file):
        raise FileNotFoundError(f"找不到 MONDO OBO 文件: {args.mondo_file}")

    path_output, edge_output, summary_output = make_output_paths(args)
    drug_queries = expand_drug_queries(args.drug_id)

    print("[1/5] 加载 MarkerDB 排除库...")
    print(f"      marker_edge_file: {os.path.abspath(args.marker_edge_file)}")
    markerdb_disease_to_markers, markerdb_marker_to_diseases, markerdb_pair_sources, markerdb_blocked_edge_keys = load_markerdb_pairs(
        args.marker_edge_file,
        None,
    )
    markerdb_pair_count = sum(len(markers) for markers in markerdb_disease_to_markers.values())
    print(f"      MarkerDB disease-marker 对: {markerdb_pair_count:,}")
    print(f"      涉及疾病: {len(markerdb_disease_to_markers):,}")
    print(f"      涉及 marker: {len(markerdb_marker_to_diseases):,}")

    print("[2/5] 加载 MONDO leaf disease 约束...")
    print(f"      mondo_file: {os.path.abspath(args.mondo_file)}")
    mondo_leaf_disease_ids, mondo_active_disease_ids, mondo_child_counts = load_mondo_leaf_disease_ids(
        args.mondo_file
    )
    print(f"      MONDO active terms: {len(mondo_active_disease_ids):,}")
    print(f"      MONDO leaf terms: {len(mondo_leaf_disease_ids):,}")
    print(f"      MONDO non-leaf terms: {len(mondo_active_disease_ids) - len(mondo_leaf_disease_ids):,}")

    print("[3/5] 加载边并构建图结构...")
    print(f"      edge_file: {os.path.abspath(args.edge_file)}")
    edges, graph, reverse_graph, node_type_map, blocked_edge_count, direct_edge_sources = build_graph_and_edges(
        args.edge_file,
        args.directed,
        markerdb_blocked_edge_keys,
    )
    print(f"      边总数: {len(edges):,}")
    print(f"      节点总数: {len(node_type_map):,}")
    print(f"      已禁用 MarkerDB 直接边: {blocked_edge_count:,}")

    go_constraint_label = "GO + leaf" if args.require_go_node else "leaf"
    candidate_mode_label = "off-label candidate" if args.candidate_mode == "offlabel" else "web-curated candidate"
    print(f"[4/5] 计算每个药物的 {candidate_mode_label} + {go_constraint_label} 约束路径...")
    drug_results = []
    for query_name, drug_id in drug_queries:
        result = find_edges_for_drug(
            drug_id,
            args.target_type,
            args.max_hops,
            args.max_output_paths,
            graph,
            reverse_graph,
            node_type_map,
            markerdb_pair_sources,
            mondo_leaf_disease_ids,
            args.require_go_node,
            args.allow_disease_after_drug,
            args.candidate_mode,
            direct_edge_sources,
        )
        drug_results.append((query_name, drug_id, result))
        print(
            "      {name} ({drug_id}): raw_candidates={raw_count:,}, markerdb_excluded={excluded_count:,}, non_leaf_excluded={non_leaf_count:,}, direct_drug_disease_excluded={direct_dd_count:,}, direct_disease_marker_excluded={direct_dm_count:,}, usable_candidates={candidate_count:,}, diseases={disease_count:,}, targets={target_count:,}, paths={path_count:,}, edges={edge_count:,}".format(
                name=query_name,
                drug_id=drug_id,
                raw_count=result["raw_candidate_pair_count"],
                excluded_count=result["markerdb_excluded_pair_count"],
                non_leaf_count=result["non_leaf_disease_excluded_pair_count"],
                direct_dd_count=result["direct_drug_disease_excluded_pair_count"],
                direct_dm_count=result["direct_disease_marker_excluded_pair_count"],
                candidate_count=result["candidate_pair_count"],
                disease_count=len(result["valid_diseases"]),
                target_count=len(result["valid_targets"]),
                path_count=len(result["paths"]),
                edge_count=len(result["selected"]),
            )
        )

    print("[5/5] 写出 CSV...")
    write_outputs(path_output, edge_output, summary_output, args, drug_results, edges, node_type_map)
    print(f"      paths: {path_output}")
    print(f"      edges: {edge_output}")
    print(f"      summary: {summary_output}")
    if args.max_output_paths > 0:
        print(f"      注意：每个药物最多输出 {args.max_output_paths:,} 条完整路径。")
    if args.max_output_edges > 0:
        print(f"      注意：每个药物最多输出 {args.max_output_edges:,} 条边。")
    print("完成。")


if __name__ == "__main__":
    main()