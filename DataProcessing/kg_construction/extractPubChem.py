import os
import time
import pandas as pd
import csv
import math
import argparse
import threading
import multiprocessing
import concurrent.futures
from rdkit.Chem import PandasTools
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdMolDescriptors
import numpy as np
from DataProcessing.kg_construction.commonSettings import *

# 添加线程锁
file_lock = threading.Lock()
property_dict = {
    'PUBCHEM_COMPOUND_CID': 'Compound ID', #
    'PUBCHEM_IUPAC_INCHIKEY': 'InChI Key',
    'PUBCHEM_OPENEYE_ISO_SMILES': 'SMILES', #
    'PUBCHEM_MONOISOTOPIC_WEIGHT': 'Monoisotopic Weight', #
    'PUBCHEM_IUPAC_CAS_NAME': 'CAS Name', # 
    'PUBCHEM_XLOGP3': 'LogP',
    'PUBCHEM_COMPOUND_CANONICALIZED': 'Canonicalized Structure',
    'PUBCHEM_IUPAC_TRADITIONAL_NAME': 'Traditional IUPAC Name',
    'PUBCHEM_ATOM_UDEF_STEREO_COUNT': 'Undefined Stereo Atoms Count',
    'PUBCHEM_CACTVS_COMPLEXITY': 'Molecular Complexity',
    'PUBCHEM_CACTVS_ROTATABLE_BOND': 'Rotatable Bonds Count',
    'PUBCHEM_XLOGP3_AA': 'LogP (AA)',
    'PUBCHEM_COORDINATE_TYPE': 'Coordinate Type',
    'PUBCHEM_NONSTANDARDBOND': 'Non-standard Bond Count',
    'PUBCHEM_BONDANNOTATIONS': 'Bond Annotations',
    'PUBCHEM_HEAVY_ATOM_COUNT': 'Heavy Atom Count',
    'PUBCHEM_IUPAC_SYSTEMATIC_NAME': 'Systematic IUPAC Name',
    'PUBCHEM_COMPONENT_COUNT': 'Component Count',
    'PUBCHEM_OPENEYE_CAN_SMILES': 'Canonical SMILES',
    'PUBCHEM_CACTVS_TPSA': 'Topological Polar Surface Area (TPSA)',
    'PUBCHEM_CACTVS_TAUTO_COUNT': 'Tautomer Count',
    'PUBCHEM_CACTVS_SUBSKEYS': 'Substructure Keys',
    'PUBCHEM_IUPAC_OPENEYE_NAME': 'OpenEye IUPAC Name',
    'PUBCHEM_MOLECULAR_WEIGHT': 'Molecular Weight',
    'PUBCHEM_ATOM_DEF_STEREO_COUNT': 'Defined Stereo Atoms Count',
    'PUBCHEM_BOND_DEF_STEREO_COUNT': 'Defined Stereo Bonds Count',
    'PUBCHEM_TOTAL_CHARGE': 'Total Charge',
    'PUBCHEM_IUPAC_INCHI': 'InChI',
    'PUBCHEM_CACTVS_HBOND_DONOR': 'Hydrogen Bond Donors',
    'PUBCHEM_MOLECULAR_FORMULA': 'Molecular Formula',
    'PUBCHEM_SMILES': 'SMILES',
    'PUBCHEM_REFERENCE_STANDARDIZATION': 'Reference Standardization',
    'PUBCHEM_EXACT_MASS': 'Exact Mass',
    'PUBCHEM_IUPAC_NAME_MARKUP': 'IUPAC Name Markup',
    'PUBCHEM_CACTVS_HBOND_ACCEPTOR': 'Hydrogen Bond Acceptors',
    'PUBCHEM_BOND_UDEF_STEREO_COUNT': 'Undefined Stereo Bonds Count',
    'PUBCHEM_IUPAC_NAME': 'IUPAC Name',
    'ROMol': 'Molecular Structure (RDKit)',
    'PUBCHEM_ISOTOPIC_ATOM_COUNT': 'Isotopic Atom Count',
}

def extract_properties(mol):
    # 创建一个字典来存储每个分子的属性
    properties = {
        'PUBCHEM_COMPOUND_CID': mol.GetProp('PUBCHEM_COMPOUND_CID') if mol.HasProp('PUBCHEM_COMPOUND_CID') else np.nan,
        'PUBCHEM_IUPAC_INCHIKEY': mol.GetProp('PUBCHEM_IUPAC_INCHIKEY') if mol.HasProp('PUBCHEM_IUPAC_INCHIKEY') else np.nan,
        'PUBCHEM_OPENEYE_ISO_SMILES': mol.GetProp('PUBCHEM_OPENEYE_ISO_SMILES') if mol.HasProp('PUBCHEM_OPENEYE_ISO_SMILES') else np.nan,
        'PUBCHEM_MONOISOTOPIC_WEIGHT': Descriptors.MolWt(mol) if mol is not None else np.nan,
        'PUBCHEM_IUPAC_CAS_NAME': mol.GetProp('PUBCHEM_IUPAC_CAS_NAME') if mol.HasProp('PUBCHEM_IUPAC_CAS_NAME') else np.nan,
        'PUBCHEM_XLOGP3': Descriptors.MolLogP(mol) if mol is not None else np.nan,
        'PUBCHEM_COMPOUND_CANONICALIZED': mol.GetProp('PUBCHEM_COMPOUND_CANONICALIZED') if mol.HasProp('PUBCHEM_COMPOUND_CANONICALIZED') else np.nan,
        'PUBCHEM_IUPAC_TRADITIONAL_NAME': mol.GetProp('PUBCHEM_IUPAC_TRADITIONAL_NAME') if mol.HasProp('PUBCHEM_IUPAC_TRADITIONAL_NAME') else np.nan,
        'PUBCHEM_ATOM_UDEF_STEREO_COUNT': mol.GetProp('PUBCHEM_ATOM_UDEF_STEREO_COUNT') if mol.HasProp('PUBCHEM_ATOM_UDEF_STEREO_COUNT') else np.nan,
        'PUBCHEM_CACTVS_COMPLEXITY': Descriptors.MolLogP(mol) if mol is not None else np.nan,  # Placeholder, adjust as needed
        'PUBCHEM_CACTVS_ROTATABLE_BOND': Descriptors.NumRotatableBonds(mol) if mol is not None else np.nan,
        'PUBCHEM_XLOGP3_AA': mol.GetProp('PUBCHEM_XLOGP3_AA') if mol.HasProp('PUBCHEM_XLOGP3_AA') else np.nan,
        'PUBCHEM_COORDINATE_TYPE': mol.GetProp('PUBCHEM_COORDINATE_TYPE') if mol.HasProp('PUBCHEM_COORDINATE_TYPE') else np.nan,
        'PUBCHEM_NONSTANDARDBOND': mol.GetProp('PUBCHEM_NONSTANDARDBOND') if mol.HasProp('PUBCHEM_NONSTANDARDBOND') else np.nan,
        'PUBCHEM_BONDANNOTATIONS': mol.GetProp('PUBCHEM_BONDANNOTATIONS') if mol.HasProp('PUBCHEM_BONDANNOTATIONS') else np.nan,
        'PUBCHEM_HEAVY_ATOM_COUNT': Descriptors.HeavyAtomCount(mol) if mol is not None else np.nan,
        'PUBCHEM_IUPAC_SYSTEMATIC_NAME': mol.GetProp('PUBCHEM_IUPAC_SYSTEMATIC_NAME') if mol.HasProp('PUBCHEM_IUPAC_SYSTEMATIC_NAME') else np.nan,
        'PUBCHEM_COMPONENT_COUNT': mol.GetProp('PUBCHEM_COMPONENT_COUNT') if mol.HasProp('PUBCHEM_COMPONENT_COUNT') else np.nan,
        'PUBCHEM_OPENEYE_CAN_SMILES': mol.GetProp('PUBCHEM_OPENEYE_CAN_SMILES') if mol.HasProp('PUBCHEM_OPENEYE_CAN_SMILES') else np.nan,
        'PUBCHEM_CACTVS_TPSA': Descriptors.TPSA(mol) if mol is not None else np.nan,
        'PUBCHEM_CACTVS_TAUTO_COUNT': mol.GetProp('PUBCHEM_CACTVS_TAUTO_COUNT') if mol.HasProp('PUBCHEM_CACTVS_TAUTO_COUNT') else np.nan,
        'PUBCHEM_CACTVS_SUBSKEYS': mol.GetProp('PUBCHEM_CACTVS_SUBSKEYS') if mol.HasProp('PUBCHEM_CACTVS_SUBSKEYS') else np.nan,
        'PUBCHEM_IUPAC_OPENEYE_NAME': mol.GetProp('PUBCHEM_IUPAC_OPENEYE_NAME') if mol.HasProp('PUBCHEM_IUPAC_OPENEYE_NAME') else np.nan,
        'PUBCHEM_MOLECULAR_WEIGHT': Descriptors.MolWt(mol) if mol is not None else np.nan,
        'PUBCHEM_ATOM_DEF_STEREO_COUNT': mol.GetProp('PUBCHEM_ATOM_DEF_STEREO_COUNT') if mol.HasProp('PUBCHEM_ATOM_DEF_STEREO_COUNT') else np.nan,
        'PUBCHEM_BOND_DEF_STEREO_COUNT': mol.GetProp('PUBCHEM_BOND_DEF_STEREO_COUNT') if mol.HasProp('PUBCHEM_BOND_DEF_STEREO_COUNT') else np.nan,
        'PUBCHEM_TOTAL_CHARGE': mol.GetProp('PUBCHEM_TOTAL_CHARGE') if mol.HasProp('PUBCHEM_TOTAL_CHARGE') else np.nan,
        'PUBCHEM_IUPAC_INCHI': mol.GetProp('PUBCHEM_IUPAC_INCHI') if mol.HasProp('PUBCHEM_IUPAC_INCHI') else np.nan,
        'PUBCHEM_CACTVS_HBOND_DONOR': mol.GetProp('PUBCHEM_CACTVS_HBOND_DONOR') if mol.HasProp('PUBCHEM_CACTVS_HBOND_DONOR') else np.nan,
        'PUBCHEM_MOLECULAR_FORMULA': mol.GetProp('PUBCHEM_MOLECULAR_FORMULA') if mol.HasProp('PUBCHEM_MOLECULAR_FORMULA') else np.nan,
        'PUBCHEM_SMILES': mol.GetProp('PUBCHEM_SMILES') if mol.HasProp('PUBCHEM_SMILES') else np.nan,
        'PUBCHEM_REFERENCE_STANDARDIZATION': mol.GetProp('PUBCHEM_REFERENCE_STANDARDIZATION') if mol.HasProp('PUBCHEM_REFERENCE_STANDARDIZATION') else np.nan,
        'PUBCHEM_EXACT_MASS': Descriptors.ExactMolWt(mol) if mol is not None else np.nan,
        'PUBCHEM_IUPAC_NAME_MARKUP': mol.GetProp('PUBCHEM_IUPAC_NAME_MARKUP') if mol.HasProp('PUBCHEM_IUPAC_NAME_MARKUP') else np.nan,
        'PUBCHEM_CACTVS_HBOND_ACCEPTOR': mol.GetProp('PUBCHEM_CACTVS_HBOND_ACCEPTOR') if mol.HasProp('PUBCHEM_CACTVS_HBOND_ACCEPTOR') else np.nan,
        'PUBCHEM_BOND_UDEF_STEREO_COUNT': mol.GetProp('PUBCHEM_BOND_UDEF_STEREO_COUNT') if mol.HasProp('PUBCHEM_BOND_UDEF_STEREO_COUNT') else np.nan,
        'PUBCHEM_IUPAC_NAME': mol.GetProp('PUBCHEM_IUPAC_NAME') if mol.HasProp('PUBCHEM_IUPAC_NAME') else np.nan,
        'ROMol': mol if mol is not None else np.nan,
        'PUBCHEM_ISOTOPIC_ATOM_COUNT': mol.GetProp('PUBCHEM_ISOTOPIC_ATOM_COUNT') if mol.HasProp('PUBCHEM_ISOTOPIC_ATOM_COUNT') else np.nan
    }
    return properties

# 读取SDF文件
def extract_from_sdf(sdf_file):
    suppl = Chem.SDMolSupplier(sdf_file)
    data = []
    
    for mol in suppl:
        if mol is not None:
            properties = extract_properties(mol)
            data.append(properties)
    
    # 将数据转换为 DataFrame
    df = pd.DataFrame(data)
    return df

# 处理SDF data并写入临时CSV
def process_sdf_data_to_temp_csv(sdf_data, temp_file_path):
    countNaN = 0
    try:
        with open(temp_file_path, "w", newline="") as temp_file:
            writer = csv.DictWriter(temp_file, fieldnames=fieldnames_node)
            writer.writeheader()
            
            # Iterate over the dataframe to create nodes
            for index, row in sdf_data.iterrows():
                CID = row['PUBCHEM_COMPOUND_CID']
                TAGED_CID = 'PubChem:' + str(CID)
                InChI = row['PUBCHEM_IUPAC_INCHI']
                InChIKey = row['PUBCHEM_IUPAC_INCHIKEY']
                SMILES = row['PUBCHEM_OPENEYE_ISO_SMILES']
                MW = row['PUBCHEM_MONOISOTOPIC_WEIGHT']
                LogP = row['PUBCHEM_XLOGP3']
                HAC = row['PUBCHEM_HEAVY_ATOM_COUNT']
                MF = row['PUBCHEM_MOLECULAR_FORMULA']
                TPSA = row['PUBCHEM_CACTVS_TPSA']
                IUPAC = row['PUBCHEM_IUPAC_CAS_NAME']
                ExactMass = row['PUBCHEM_EXACT_MASS']
                RB = row['PUBCHEM_CACTVS_ROTATABLE_BOND']
                SysName = row['PUBCHEM_IUPAC_SYSTEMATIC_NAME']
                
                # Check if any of the required fields are empty
                if any([pd.isna(CID), pd.isna(InChIKey), pd.isna(SMILES), pd.isna(MW), pd.isna(LogP), pd.isna(HAC), pd.isna(MF), pd.isna(TPSA), pd.isna(IUPAC), pd.isna(ExactMass), pd.isna(RB), pd.isna(SysName)]):
                    countNaN += 1
                    continue
                    
                comments = f"InChIKey: {InChIKey if InChIKey else ''}, InChI: {InChI if InChI else ''}, MW: {MW if MW else ''}, LogP: {LogP if LogP else ''}, HAC: {HAC if HAC else ''}, MF: {MF if MF else ''}, TPSA: {TPSA if TPSA else ''}, ExactMass: {ExactMass if ExactMass else ''}, RB: {RB if RB else ''}, SysName: {SysName if SysName else ''}"
                node_data = {
                    "ID": TAGED_CID,
                    "Type": "Compound",
                    "Name": IUPAC,
                    "Sequence": SMILES,
                    "Comments": comments
                }
                
                # Write nodes to CSV
                writer.writerow(node_data)
                
        return countNaN, True
    except Exception as e:
        print(f"写入临时文件 {temp_file_path} 时出错: {str(e)}")
        return countNaN, False

# 处理单个SDF文件
def process_file(file_idx, file_path, output_dir):
    start_time = time.time()
    temp_file = os.path.join(output_dir, f"temp_{file_idx}.csv")
    
    try:
        print(f"[{file_idx}] 开始处理文件: {os.path.basename(file_path)}")
        sdf_data = extract_from_sdf(file_path)
        print(f"[{file_idx}] 文件 {os.path.basename(file_path)} 中共有 {len(sdf_data)} 个分子")
        
        countNaN, success = process_sdf_data_to_temp_csv(sdf_data, temp_file)
        
        elapsed_time = time.time() - start_time
        print(f"[{file_idx}] 处理完成 {os.path.basename(file_path)}, 用时: {elapsed_time:.2f}秒, 共有 {countNaN} 个分子缺少必要字段")
        
        if success:
            return temp_file, elapsed_time
        else:
            return None, elapsed_time
            
    except Exception as e:
        print(f"[{file_idx}] 处理文件 {os.path.basename(file_path)} 时出错: {str(e)}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return None, 0

# 合并所有临时文件
def merge_temp_files(temp_files, output_csv):
    try:
        with open(output_csv, 'w', newline='') as outfile:
            writer = None
            
            for temp_file in temp_files:
                if not os.path.exists(temp_file) or os.path.getsize(temp_file) == 0:
                    continue
                    
                with open(temp_file, 'r', newline='') as infile:
                    reader = csv.DictReader(infile)
                    
                    if writer is None:
                        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
                        writer.writeheader()
                    
                    for row in reader:
                        writer.writerow(row)
                
                # 删除临时文件
                os.remove(temp_file)
                
        return True
    except Exception as e:
        print(f"合并临时文件时出错: {str(e)}")
        return False

# 主程序
def main():
    import time
    
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Process Pubchem SDF files and export to CSV.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Input SDF file path")
    parser.add_argument('-o', '--output', type=str, nargs=1, required=True, help="Output CSV files (node)")
    args = parser.parse_args()
    
    input_path = args.input
    output_node_csv = args.output[0]
    
    # 创建临时目录
    temp_dir = "temp_pubchem"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # 计算要使用的线程数（CPU核心数的一半）
    # max_workers = max(1, multiprocessing.cpu_count() // 2)
    max_workers = 4
    print(f"将使用 {max_workers} 个线程进行并行处理")
    
    overall_start_time = time.time()
    
    # 检查输入是文件还是目录
    if os.path.isfile(input_path):
        # 单文件处理
        files_to_process = [input_path]
        base_dir = os.path.dirname(input_path)
    else:
        # 目录处理，获取所有SDF文件
        base_dir = input_path
        files_to_process = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith('.sdf')]
    
    totalFiles = len(files_to_process)
    print(f"待处理SDF文件数量: {totalFiles}")
    
    # 使用线程池并行处理所有文件
    processed_results = []
    total_processing_time = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_file = {
            executor.submit(process_file, idx, file_path, temp_dir): (idx, file_path) 
            for idx, file_path in enumerate(files_to_process)
        }
        
        # 收集任务结果
        for future in concurrent.futures.as_completed(future_to_file):
            idx, file_path = future_to_file[future]
            try:
                temp_file, elapsed_time = future.result()
                if temp_file:  # 如果处理成功
                    processed_results.append(temp_file)
                    total_processing_time += elapsed_time
                
                print(f"进度: {len(processed_results)}/{totalFiles}, 文件: {os.path.basename(file_path)}, 处理时间: {elapsed_time:.2f}秒")
            
            except Exception as e:
                print(f"获取文件 {os.path.basename(file_path)} 的处理结果时出错: {str(e)}")
    
    # 合并所有临时文件到最终输出
    print(f"处理完成，正在合并 {len(processed_results)} 个临时文件到最终输出...")
    if merge_temp_files(processed_results, output_node_csv):
        print(f"合并完成，输出文件: {output_node_csv}")
    else:
        print("合并失败，请检查错误信息")
    
    # 清理临时目录
    if os.path.exists(temp_dir) and not os.listdir(temp_dir):  # 确保目录为空
        os.rmdir(temp_dir)
    
    # 计算总时间和加速比
    overall_elapsed_time = time.time() - overall_start_time
    print(f"所有处理完成，总运行时间: {overall_elapsed_time:.2f}秒，处理时间总和: {total_processing_time:.2f}秒")
    if overall_elapsed_time > 0:
        print(f"多线程加速比: {total_processing_time/overall_elapsed_time:.2f}x")

if __name__ == "__main__":
    main()
