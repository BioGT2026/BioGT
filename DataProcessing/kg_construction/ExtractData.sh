#!/bin/bash

target_dataset=$1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

run_script() {
    local script_name=$1
    shift
    python3 "$SCRIPT_DIR/$script_name" "$@"
}

# Initialize error log
ERROR_LOG="running.log"
echo "Execution started at $(date)" >"$ERROR_LOG"

# Data path prefix
# Check if the hostname is mu01 or gpu01, then set the use_gpu_flag to 1
hostname=$(hostname)
echo "Hostname: $hostname" >>"$ERROR_LOG"
if [ "$hostname" == "mu01" ]; then
    use_gpu_node_flag=1
elif [ "$hostname" == "gpu01" ]; then
    use_gpu_node_flag=1
else
    use_gpu_node_flag=0
fi

output_folder="graphWithNodeType"
merged_folder="merge"

data_path_prefix="/home/$USER/Data"
output_path_prefix="/home/$USER/Data/${output_folder}"
merged_data_path_prefix="${output_path_prefix}/$merged_folder"
if [ "$use_gpu_node_flag" == 1 ]; then
    echo "Using GPU node for data processing."
    data_path_prefix="/home/$USER/dataset"
    output_path_prefix="/home/$USER/dataset/${output_folder}"
    merged_data_path_prefix="${output_path_prefix}/$merged_folder"
fi

if [ ! -d "$output_path_prefix" ]; then
    mkdir -p "$output_path_prefix"
fi

# Function to log errors
log_error() {
    local dataset=$1
    local script=$2
    local error=$3
    echo "[$(date)] ERROR in $dataset processing: $script failed with error: $error" >>"$ERROR_LOG"
}

decompressPubChem() {
    data_path_prefix_local=$1
    original_dir="${data_path_prefix_local}/SDF"
    target_dir="${data_path_prefix_local}/uncompressed_sdfs"

    # 创建目标目录（如果不存在）
    if [ ! -d "${target_dir}" ]; then
        mkdir -p "${target_dir}"
    else
        # Check if the target directory is empty
        if [ "$(ls -A ${target_dir})" ]; then
            echo "目标目录不为空！已完成解压缩：${target_dir}"
            return 0
        fi
    fi

    # 遍历所有 .sdf.gz 文件（自动跳过 .md5 文件）
    find "${original_dir}" -type f -name 'Compound_*.sdf.gz' | while read -r gz_file; do
        # 提取文件名（不含路径）
        filename=$(basename "$gz_file")
        # 生成解压后的文件名（移除 .gz 后缀）
        output_file="${filename%.gz}"
        # 解压到输出目录
        echo "解压: $filename → $target_dir/$output_file"
        gunzip -c "$gz_file" >"$target_dir/$output_file"
    done

    echo "完成！所有文件已解压至：$target_dir"
}

func() {
    target_dataset_name=$1
    if [ "$target_dataset_name" == "uniprot" ]; then
        uniprot_data_set=("uniprot_sprot")
        path_to_intact_ppi="$data_path_prefix/ppi/intact_human_ppi.tsv"
        path_to_biogrid_ppi="$data_path_prefix/ppi/biogrid_ppi.tsv"
        for data_set in "${uniprot_data_set[@]}"; do
            if ! run_script extractUniprot.py -i "$data_path_prefix/$target_dataset_name/$data_set.fasta" -ip "$path_to_intact_ppi" -bp "$path_to_biogrid_ppi" -o "$output_path_prefix/${data_set}_node.csv" "$output_path_prefix/${data_set}_edge.csv" 2>>"$ERROR_LOG"; then
                log_error "$target_dataset_name" "extractUniprot.py" "Failed to process $data_set"
            fi
        done
    elif [ "$target_dataset_name" == "reactome" ]; then
        data_set="reactome"
        if ! run_script extractReactome.py -i "$data_path_prefix/$target_dataset_name" -o "$output_path_prefix/${data_set}_node.csv" "$output_path_prefix/${data_set}_edge.csv" 2>>"$ERROR_LOG"; then
            log_error "$target_dataset_name" "extractReactome.py" "Failed to process $data_set"
        fi
    elif [ "$target_dataset_name" == "drugcentral" ]; then
        data_set="drugcentral"
        # Get the mondo umls mapping file
        if ! run_script extractMondoUMLS.py -i "$data_path_prefix/mondo/mondo.obo" -o "./mondo_umls.csv" 2>>"$ERROR_LOG"; then
            log_error "$target_dataset_name" "extractMondoUMLS.py" "Failed to process mondo umls mapping"
        fi
        # Get the drugbank cas mapping file
        if ! run_script extractDrugCAS.py -i "$data_path_prefix/DrugBank/CompleteDB/full-database.xml" -o "./drug_cas.csv" 2>>"$ERROR_LOG"; then
            log_error "$target_dataset_name" "extractDrugCAS.py" "Failed to process drugbank cas mapping"
        fi
        # Extract drugcentral data
        if ! run_script extractDrugCentral.py -i "$data_path_prefix/DrugCentral/drug_disease.csv" "./drug_cas.csv" "./mondo_umls.csv" -o "$output_path_prefix/${data_set}_edge.csv" 2>>"$ERROR_LOG"; then
            log_error "$target_dataset_name" "extractDrugCentral.py" "Failed to process $data_set"
        fi
    elif [ "$target_dataset_name" == "sider" ]; then
        data_set="sider"
        # Get the mondo umls mapping file
        if ! run_script extractMondoUMLS.py -i "$data_path_prefix/mondo/mondo.obo" -o "./mondo_umls.csv" 2>>"$ERROR_LOG"; then
            log_error "$target_dataset_name" "extractMondoUMLS.py" "Failed to process mondo umls mapping"
        fi
        # Get the drugbank cas mapping file
        if ! run_script extractDrugCAS.py -i "$data_path_prefix/DrugBank/CompleteDB/full-database.xml" -o "./drug_cas.csv" 2>>"$ERROR_LOG"; then
            log_error "$target_dataset_name" "extractDrugCAS.py" "Failed to process drugbank cas mapping"
        fi
        # Extract hpo umls mapping
        if ! run_script extractHPOUMLS.py -i "$data_path_prefix/hpo/hp.obo" -o "./hpo_umls.csv" 2>>"$ERROR_LOG"; then
            log_error "$target_dataset_name" "extractHPOUMLS.py" "Failed to process hpo umls mapping"
        fi
        # Extract sider data
        if ! run_script extractSider.py -i "$data_path_prefix/sider/drug_atc.tsv" "$data_path_prefix/sider/meddra_all_se.tsv" "./hpo_umls.csv" "./mondo_umls.csv" "./drug_cas.csv" -o "$output_path_prefix/${data_set}_edge.csv" 2>>"$ERROR_LOG"; then
            log_error "$target_dataset_name" "extractSider.py" "Failed to process $data_set"
        fi
    elif [ "$target_dataset_name" == "mondo" ]; then
        mondo_data_set=("mondo")
        path_to_go_mapping="$output_path_prefix/go_node.csv"
        for data_set in "${mondo_data_set[@]}"; do
            if ! run_script extractMondo.py -i "$data_path_prefix/$target_dataset_name/$data_set.obo" "$data_path_prefix/$target_dataset_name/$data_set.owl" -g "$path_to_go_mapping" -o "$output_path_prefix/${data_set}_node.csv" "$output_path_prefix/${data_set}_edge.csv" 2>>"$ERROR_LOG"; then
                log_error "$target_dataset_name" "extractMondo.py" "Failed to process $data_set"
            fi
        done
    elif [ "$target_dataset_name" == "marker" ]; then
        path_to_mondo_mapping="$output_path_prefix/mondo_node.csv"
        path_to_refseq_mapping="$output_path_prefix/refseq_node.csv"
        dataset_path="biomarker"
        if ! run_script extractBioMarkerAndMerge.py -i "$data_path_prefix/$dataset_path/MarkerDBGene.csv" "$data_path_prefix/$dataset_path/download_all_sequence_variants.tsv" "$data_path_prefix/$dataset_path/chemical_data.txt" "$data_path_prefix/$dataset_path/download_all_chemicals.tsv" "$data_path_prefix/$dataset_path/protein_data.txt" "$data_path_prefix/$dataset_path/download_all_proteins.tsv" "$data_path_prefix/$dataset_path/micrna.csv" --refseq-nodes "$path_to_refseq_mapping" -m "$path_to_mondo_mapping" -o "$output_path_prefix/marker_edge.csv" 2>>"$ERROR_LOG"; then
            log_error "$target_dataset_name" "extractBioMarkerAndMerge.py" "Failed to process $data_set"
        fi
    elif [ "$target_dataset_name" == "uberon" ]; then
        uberon_data_set=("uberon")
        path_to_go_mapping="$output_path_prefix/go_node.csv"
        for data_set in "${uberon_data_set[@]}"; do
            if ! run_script extractUberon.py -i "$data_path_prefix/Anatomical/$data_set.obo" -g "$path_to_go_mapping" -o "$output_path_prefix/${data_set}_node.csv" "$output_path_prefix/${data_set}_edge.csv" 2>>"$ERROR_LOG"; then
                log_error "$target_dataset_name" "extractUberon.py" "Failed to process $data_set"
            fi
        done
    elif [ "$target_dataset_name" == "hporelated" ]; then
        # Get the mondo omim mapping file
        if ! run_script extractMondoOMIM.py -i "$data_path_prefix/mondo/mondo.obo" -o "./mondo_omim.csv" 2>>"$ERROR_LOG"; then
            log_error "$target_dataset_name" "extractMondoOMIM.py" "Failed to process mondo OMIM mapping"
        fi
        # Get the hpo relations
        if ! run_script extractHPORelations.py -i "$data_path_prefix/hpo/maxo-annotations.tsv" "$data_path_prefix/hpo/genes_to_disease.txt" "$data_path_prefix/hpo/genes_to_phenotype.txt" "$data_path_prefix/hpo/phenotype_to_genes.txt" "$data_path_prefix/hpo/phenotype.hpoa" "./mondo_omim.csv" -o "$output_path_prefix/hpo_relations_edge.csv" 2>>"$ERROR_LOG"; then
            log_error "$target_dataset_name" "extractHPORelations.py" "Failed to process hpo relations"
        fi
    elif [ "$target_dataset_name" == "hpo" ]; then
        if ! run_script extractHPO.py -i "$data_path_prefix/$target_dataset_name/hp.obo" -o "$output_path_prefix/hp_node.csv" "$output_path_prefix/hp_edge.csv" 2>>"$ERROR_LOG"; then
            log_error "$target_dataset_name" "extractHPO.py" "Failed to process hp.obo"
        fi
    elif [ "$target_dataset_name" == "go" ]; then
        if ! run_script extractGO.py --obo "$data_path_prefix/$target_dataset_name/go-basic.obo" --gaf "$data_path_prefix/$target_dataset_name/goa_human.gaf" --nodes_out "$output_path_prefix/go_node.csv" --edges_out "$output_path_prefix/go_edge.csv" 2>>"$ERROR_LOG"; then
            log_error "$target_dataset_name" "extractGO.py" "Failed to process $data_set"
        fi
    elif [ "$target_dataset_name" == "pubchem" ]; then
        # call decompressPubChem func
        decompressPubChem "$data_path_prefix/$target_dataset_name"
        local uncompressed_dir="$data_path_prefix/$target_dataset_name/uncompressed_sdfs"

        # Check if the uncompressed_sdfs directory exists
        if [ ! -d "$uncompressed_dir" ]; then
            echo "Error: Directory $uncompressed_dir does not exist." >>"$ERROR_LOG"
            exit 1
        fi
        if ! run_script extractPubChem.py -i "$data_path_prefix/$target_dataset_name/uncompressed_sdfs" -o "$output_path_prefix/${target_dataset_name}_node.csv" 2>>"$ERROR_LOG"; then
            log_error "$target_dataset_name" "extractPubChem.py" "Failed to process $target_dataset_name"
        fi
    elif [ "$target_dataset_name" == "hmdb" ]; then
        data_set="hmdb"
        # Get the mondo omim mapping file
        if ! run_script extractMondoOMIM.py -i "$data_path_prefix/mondo/mondo.obo" -o "./mondo_omim.csv" 2>>"$ERROR_LOG"; then
            log_error "$target_dataset_name" "extractMondoOMIM.py" "Failed to process mondo OMIM mapping"
        fi
        if ! run_script extractHMDB.py -i "$data_path_prefix/hmdb/hmdb_metabolites.xml" "$data_path_prefix/hmdb/hmdb_proteins.xml" "./mondo_omim.csv" "$output_path_prefix/reactome_node.csv" -o "$output_path_prefix/${data_set}_node.csv" "$output_path_prefix/${data_set}_edge.csv" 2>>"$ERROR_LOG"; then
            log_error "$target_dataset_name" "extractHMDB.py" "Failed to process $data_set"
        fi
    elif [ "$target_dataset_name" == "drugbank" ]; then
        drugs_record="DrugBank/CompleteDB/full-database.xml"
        drugs_sdf="DrugBank/Structure/structures.sdf"
        gene_ids="DrugBank/GeneID.tsv"
        if ! run_script extractDrugs.py -i "$data_path_prefix/$drugs_record" "$data_path_prefix/$drugs_sdf" "$data_path_prefix/$gene_ids" -o "$output_path_prefix/drugbank_node.csv" "$output_path_prefix/drugbank_edge.csv" 2>>"$ERROR_LOG"; then
            log_error "$target_dataset_name" "extractDrugs.py" "Failed to process DrugBank data"
        fi
    elif [ "$target_dataset_name" == "gene" ]; then
        path_to_protein_mapping="$output_path_prefix/../refseq_uniprot_edge.csv"
        path_to_go_mapping="$output_path_prefix/go_node.csv"
        data_set="selected_gene_files"
        if ! run_script extractGene.py -i "$data_path_prefix/NCBI/gene/ftp.ncbi.nih.gov/gene/DATA/$data_set" -p "$path_to_protein_mapping" -g "$path_to_go_mapping" -o "$output_path_prefix/${data_set}_node.csv" "$output_path_prefix/${data_set}_edge.csv" 2>>"$ERROR_LOG"; then
            log_error "$target_dataset_name" "extractGene.py" "Failed to process $data_set"
        fi
    elif [ "$target_dataset_name" == "clinvar" ]; then
        path="NCBI/clinvar/ftp.ncbi.nlm.nih.gov/pub/clinvar/xml/ClinVarVCVRelease_2025-02.xml"
        data_set="ClinVarVCVRelease_2025-02"
        if ! run_script extractClinvar.py -i "$data_path_prefix/$path" -o "$output_path_prefix/${data_set}_node.csv" "$output_path_prefix/${data_set}_edge.csv" 2>>"$ERROR_LOG"; then
            log_error "$target_dataset_name" "extractClinvar.py" "Failed to process $data_set"
        fi
    elif [ "$target_dataset_name" == "refseq" ]; then
        path="NCBI/refseq/ftp.ncbi.nlm.nih.gov/refseq/release/complete"
        data_set="refseq"
        path_to_protein_mapping="$output_path_prefix/../refseq_uniprot_edge.csv"
        path_to_go_mapping="$output_path_prefix/go_node.csv"
        if ! run_script extractRefSeq.py -i "$data_path_prefix/$path" -p "$path_to_protein_mapping" -g "$path_to_go_mapping" -o "$output_path_prefix/${data_set}_node.csv" "$output_path_prefix/${data_set}_edge.csv" 2>>"$ERROR_LOG"; then
            log_error "$target_dataset_name" "extractRefSeq.py" "Failed to process $data_set"
        fi
    elif [ "$target_dataset_name" == "protein-id" ]; then
        path_to_protein_mapping="uniprot/idmapping.dat"
        data_set="refseq_uniprot"
        if ! run_script extractUniprotMapping.py -i "$data_path_prefix/$path_to_protein_mapping" -o "$output_path_prefix/${data_set}_edge.csv" 2>>"$ERROR_LOG"; then
            log_error "$target_dataset_name" "extractUniprotMapping.py" "Failed to process $path_to_protein_mapping"
        else
            mv "$output_path_prefix/${data_set}_edge.csv" ../
        fi
    fi
}

if [ "$target_dataset" == "all" ]; then
    # target_data_sets=("protein-id" "uniprot" "reactome" "mondo" "hpo" "go" "hmdb" "gene" "clinvar" "uberon" "drugbank" "refseq")
    target_data_sets=("reactome" "mondo" "hpo" "drugbank" "hporelated" "hmdb" "drugcentral" "sider" "marker") # Maintaing the order of execution since some datasets depend on others.
    # target_data_sets=("protein-id" "go" "gene" "uniprot" "reactome" "refseq" "hmdb" "mondo" "hpo" "uberon" "drugbank" "clinvar") # Maintaing the order of execution since some datasets depend on others.
    for data_set in "${target_data_sets[@]}"; do
        echo "Processing dataset: $data_set"
        func "$data_set"
    done
elif [ "$target_dataset" == "merge" ]; then
    # Merge all node and edge files
    if [ ! -d $merged_data_path_prefix ]; then
        mkdir -p $merged_data_path_prefix
    fi
    # Find the files in the directory but exclude the subdirectories
    nodeFileList=$(find "$output_path_prefix" -maxdepth 1 -type f -name "*_node.csv")
    edgeFileList=$(find "$output_path_prefix" -maxdepth 1 -type f -name "*_edge.csv")
    nodeFileList=$(echo "$nodeFileList" | tr '\n' ' ')
    edgeFileList=$(echo "$edgeFileList" | tr '\n' ' ')
    # Remove marker_edge.csv from edgeFileList if it exists
    marker_edge_path="$output_path_prefix/marker_edge.csv"
    edgeFileList=$(echo "$edgeFileList" | sed "s|$marker_edge_path||g")
    echo "Number of node files: $(echo "$nodeFileList" | wc -w)"
    echo "$nodeFileList"
    echo "Number of edge files: $(echo "$edgeFileList" | wc -w)"
    echo "$edgeFileList"
    echo "Locate the marker edge file at: $marker_edge_path"
    # Call mergeData.py - Remove quotes around the file list variables
    run_script mergeData.py -n "$nodeFileList" -e "$edgeFileList" --marker-edge-file "$marker_edge_path" --out-nodes "$merged_data_path_prefix/merged_node.csv" --out-edges "$merged_data_path_prefix/merged_edge.csv" 2>>"$ERROR_LOG"
else
    func "$target_dataset"
fi

echo "Execution completed at $(date)" >>$ERROR_LOG
