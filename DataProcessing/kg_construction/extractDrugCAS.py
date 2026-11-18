import csv
import collections
import os
import re
import xml.etree.ElementTree as ET
import sys
import argparse
import csv
import itertools
from DataProcessing.kg_construction.commonSettings import *
filedNameForDrugATC= ['Drug ID', 'Drug Name', 'CAS', 'ATC Code', 'InChi']

def process_xml(xml_path, writer_edge):
    """
    Process a DrugBank XML file to extract drug-ATC and drug-InChI relationships.
    :param xml_path: The path to the DrugBank XML file
    :param writer_edge: CSV writer for edges
    :return: None
    """
    with open(xml_path) as xml_file:
        tree = ET.parse(xml_file)
    root = tree.getroot()

    ns = '{http://www.drugbank.ca}'
    inchikey_template = "{ns}calculated-properties/{ns}property[{ns}kind='InChIKey']/{ns}value"
    inchi_template = "{ns}calculated-properties/{ns}property[{ns}kind='InChI']/{ns}value"

    written_nodes = set()
    drug_count = 0

    for drug in root:
        drug_count += 1
        row = collections.OrderedDict()
        assert drug.tag == ns + 'drug'
        row['Type'] = drug.get('type')
        drug_id = drug.findtext(ns + "drugbank-id[@primary='true']")
        row['ID'] = drug_id
        row['name'] = drug.findtext(ns + "name")
        row['Description'] = drug.findtext(ns + "description")
        row['Groups'] = [group.text for group in drug.findall("{ns}groups/{ns}group".format(ns=ns))]
        row['Atc_codes'] = [code.get('code') for code in drug.findall("{ns}atc-codes/{ns}atc-code".format(ns=ns))]
        row['Categories'] = [x.findtext(ns + 'category') for x in drug.findall("{ns}categories/{ns}category".format(ns=ns))]
        row['Inchi'] = drug.findtext(inchi_template.format(ns=ns))
        row['Inchikey'] = drug.findtext(inchikey_template.format(ns=ns))
        row['CAS'] = drug.findtext(ns + "cas-number")

        aliases = {
            elem.text for elem in
            drug.findall("{ns}international-brands/{ns}international-brand".format(ns=ns)) +
            drug.findall("{ns}synonyms/{ns}synonym[@language='English']".format(ns=ns)) +
            drug.findall("{ns}international-brands/{ns}international-brand".format(ns=ns)) +
            drug.findall("{ns}products/{ns}product/{ns}name".format(ns=ns))
        }
        aliases.add(row['name'])
        row['Names'] = '|'.join(sorted(aliases))

        # Process ATC codes
        atc_codes = ""
        for atc_code in row['Atc_codes']:
            if not atc_code:
                continue
            atc_codes += (atc_code + "|")


        # Process InChI
        inchi = row['Inchi']
        if inchi:
            inchi = inchi.strip()
        else:
            inchi = ''
        
        cas = row['CAS']
        if cas:
            cas = cas.strip()
        else:
            cas = ''

        # Write drug-ATC relationships
        write_data = {
            'Drug ID': drug_id,
            'Drug Name': row['name'],
            'CAS': cas,
            'ATC Code': atc_codes.strip('|'),
            'InChi': inchi
        }
        writer_edge.writerow(write_data)

    print(f"Processed {drug_count} drugs.")

def main():
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Process a DrugBank XML file to extract drug-ATC and drug-inchi relationships.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Input DrugBank XML file")
    parser.add_argument('-o', '--output', type=str, nargs=1, required=True, help="Output CSV files (edge)")

    args = parser.parse_args()
    # 获取输入和输出文件路径
    xml_file = args.input
    output_edge_csv = args.output[0]

    with open(output_edge_csv, "w", newline="") as file_edge:
        writer_edge = csv.DictWriter(file_edge, fieldnames=filedNameForDrugATC)
        writer_edge.writeheader()
        # Process the XML file
        process_xml(xml_file, writer_edge)
        print(f"数据已成功导出到边记录：{output_edge_csv}")

if __name__ == "__main__":
    main()
