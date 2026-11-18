import argparse
import csv
import math
import re
import requests
import json
from DataProcessing.kg_construction.commonSettings import *
import os
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

def fetchStatusViaAPI(marker_id, type):
    formatted_marker_id = f"MDB{int(marker_id):08d}"
    API_key = "cc69fb7c1a4809bbae7c0701d5188c17"
    type = type.lower()
    url = ""
    if type == "protein":
        url = f"http://markerdb.ca/api/v1/proteinapi/proteinrequest?api_key={API_key}&markerdb_id={formatted_marker_id}"
    elif type == "chemical":
        url = f"http://markerdb.ca/api/v1/chemicalapi/chemicalrequest?api_key={API_key}&markerdb_id={formatted_marker_id}"
    elif type == "gene":
        url = f"http://markerdb.ca/api/v1/geneapi/generequest?api_key={API_key}&markerdb_id={formatted_marker_id}"
    elif type == "general":
        url = f"http://markerdb.ca/api/v1/generalapi/generalrequest?api_key={API_key}&markerdb_id={formatted_marker_id}"
    else:
        raise ValueError("Unsupported type. Use 'protein', 'chemical', or 'gene'.")

    try:
        print(f"Fetching data for {marker_id} via API: {url}")
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # 检查返回的数据是否表示ID无效
        if isinstance(data, dict) and data.get('failed') == 'markerdb id is not valid':
            print(f"INFO: Marker ID {marker_id} is not valid.")
            return None
        else:
            return data
    except requests.RequestException as e:
        # print(f"Error fetching data for {marker_id} via API: {e}")
        return None

def process_marker(marker_id, data_types, output_files, lock):
    """
    处理单个marker_id的下载和写入任务
    """
    for data_type in data_types:
        data = fetchStatusViaAPI(marker_id, data_type)
        
        if data:
            formatted_marker_id = f"MDB{int(marker_id):08d}"
            with lock:
                output_files[data_type].write(f"{formatted_marker_id}\t{json.dumps(data)}\n")
                output_files[data_type].flush() # 确保立即写入
            print(f"SUCCESS: Found and wrote {data_type} data for marker {marker_id}.")
            return # 找到后不再尝试其他类型
    
    print(f"INFO: No data found for marker {marker_id} across all types.")

# 主函数
def main():
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Download data from MarkerDB and save to separate files by type.")
    parser.add_argument('-o', '--output_dir', type=str, required=True, help="Output directory to save data files")
    parser.add_argument('-w', '--workers', type=int, default=10, help="Number of worker threads for downloading")

    args = parser.parse_args()
    
    output_dir = args.output_dir
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    output_files = {
        "protein": open(os.path.join(output_dir, "protein_data.txt"), 'w', encoding='utf-8'),
        "chemical": open(os.path.join(output_dir, "chemical_data.txt"), 'w', encoding='utf-8'),
        "gene": open(os.path.join(output_dir, "gene_data.txt"), 'w', encoding='utf-8')
    }

    lock = threading.Lock()
    data_types = ["gene", "chemical", "protein"]
    start_id = 1
    end_id = 628968 # 根据需要调整
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_marker, marker_id, data_types, output_files, lock) for marker_id in range(start_id, end_id + 1)]
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred in a thread: {e}")

    # 关闭所有文件
    for f in output_files.values():
        f.close()

    print(f"数据已成功导出到目录: {output_dir}")

if __name__ == "__main__":
    main()