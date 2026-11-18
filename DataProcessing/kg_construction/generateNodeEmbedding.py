import csv
import os
import argparse
from collections import defaultdict
from DataProcessing.kg_construction.commonSettings import fieldnames_node, fieldnames_edge
import logging
import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import gc
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np
import multiprocessing
from functools import partial

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
csv.field_size_limit(1000 * 1024 * 1024)

FIELDNAMES_NODE = fieldnames_node
FIELDNAMES_NODE_EMBED = ["ID", "Type", "Name", "Feature"]
FIELDNAMES_EDGE = fieldnames_edge
MERGE_DELIMITER = '; '
DEFAULT_EMBED_SIZE = 128
DEFAULT_BATCH_SIZE_FOR_EMBEDDING = 128
DEFAULT_LOCAL_MODEL_PATH = "~/.cache/huggingface/hub"
RDKIT_FP_BITS = DEFAULT_EMBED_SIZE * 8

# --- Helper Functions ---

def log_memory_usage(stage=""):
    if torch.cuda.is_available():
        free_mem, total_mem = torch.cuda.mem_get_info()
        logging.debug(f"Memory Usage ({stage}): CUDA Free: {free_mem/1024**3:.2f}GB / Total: {total_mem/1024**3:.2f}GB")

def _process_single_smiles_for_rdkit(smiles_n_bits_tuple):
    smiles, n_bits = smiles_n_bits_tuple
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
            arr = np.zeros((n_bits,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        except Exception as e:
            logging.error(f"Error generating fingerprint for SMILES (worker): {smiles} - {e}")
            return np.zeros((n_bits,), dtype=np.float32)
    else:
        return np.zeros((n_bits,), dtype=np.float32)

def _generate_transformer_embeddings(model, tokenizer, sequences, batch_size, device):
    log_memory_usage(f"Before generating transformer embeddings with {model.__class__.__name__}")
    model.to(device)
    model.eval()
    logging.info(f"Using device to generate sequence embeddings: {device}")

    vocab_size = getattr(model.config, 'vocab_size', None)
    if vocab_size is None:
        try:
            vocab_size = model.embeddings.word_embeddings.num_embeddings
        except AttributeError:
            logging.warning("Could not determine model vocabulary size automatically.")

    all_embeddings = []
    target_num = len(sequences)
    for i in range(0, target_num, batch_size):
        batch_sequences = sequences[i:i + batch_size]
        encoded_input = tokenizer(batch_sequences, padding=True, truncation=True, return_tensors='pt')

        if vocab_size is not None:
            max_token_id = encoded_input['input_ids'].max().item()
            if max_token_id >= vocab_size:
                logging.error(f"Error in batch starting at index {i}: Max token ID ({max_token_id}) is out of bounds for vocab size ({vocab_size}).")
                raise ValueError(f"Invalid token ID detected (>= {vocab_size}) in batch starting at index {i}.")

        encoded_input = encoded_input.to(device)
        try:
            with torch.no_grad():
                model_output = model(**encoded_input)
                batch_embeddings_tensor = model_output.last_hidden_state.mean(dim=1)
            all_embeddings.append(batch_embeddings_tensor.cpu())
        except RuntimeError as e:
            logging.error(f"RuntimeError during model forward pass for batch {i//batch_size}: {e}")
            raise e
        finally:
            del encoded_input, model_output, batch_embeddings_tensor
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        log_memory_usage(f"After transformer batch {i//batch_size + 1}")
    
    embeddings = torch.cat(all_embeddings, dim=0)
    model.to('cpu')
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    log_memory_usage("After generating transformer embeddings")
    return embeddings

def _generate_rdkit_embeddings(smiles_list, n_bits, device):
    log_memory_usage("Before generating RDKit fingerprints (parallel)")
    if not smiles_list:
        return torch.empty((0, n_bits), dtype=torch.float32)

    overall_batch_size = 100000
    all_fingerprints_np_overall = []

    for i in range(0, len(smiles_list), overall_batch_size):
        current_smiles_batch = smiles_list[i:i + overall_batch_size]
        logging.info(f"Processing overall RDKit batch {i//overall_batch_size + 1}/{(len(smiles_list) -1) // overall_batch_size + 1} (size: {len(current_smiles_batch)})")
        
        num_processes = max(1, multiprocessing.cpu_count() // 2)
        maxtasks = 500
        
        tasks = [(s, n_bits) for s in current_smiles_batch]

        current_batch_fingerprints_np = []
        with multiprocessing.Pool(processes=num_processes, maxtasksperchild=maxtasks) as pool:
            try:
                chunksize_val = max(1, len(tasks) // (num_processes * 8))
                current_batch_fingerprints_np = pool.map(_process_single_smiles_for_rdkit, tasks, chunksize=chunksize_val)
            except Exception as e:
                logging.error(f"Error during parallel RDKit fingerprint generation: {e}", exc_info=True)
                logging.info("Falling back to serial RDKit fingerprint generation for this batch.")
                current_batch_fingerprints_np = [_process_single_smiles_for_rdkit(task_tuple) for task_tuple in tasks]
        all_fingerprints_np_overall.extend(current_batch_fingerprints_np)
        log_memory_usage(f"After processing overall RDKit batch {i//overall_batch_size + 1}")
        gc.collect()

    processed_fingerprints = [fp_array if fp_array is not None and fp_array.shape == (n_bits,) else np.zeros((n_bits,), dtype=np.float32) for fp_array in all_fingerprints_np_overall]
    
    if not processed_fingerprints:
        return torch.empty((0, n_bits), dtype=torch.float32)
    
    embeddings = torch.tensor(np.array(processed_fingerprints), dtype=torch.float32)
    log_memory_usage("After generating RDKit fingerprints (parallel)")
    return embeddings

def _generate_sentence_embeddings(model, sentences, batch_size, device):
    log_memory_usage("Before generating sentence embeddings")
    model.to(device)
    model.eval()
    logging.info(f"Using device to generate comment embeddings: {device}")
    all_embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]
        try:
            with torch.no_grad():
                batch_embeddings_tensor = model.encode(batch_sentences, convert_to_tensor=True, device=device)
            all_embeddings.append(batch_embeddings_tensor.cpu())
        except Exception as e:
            logging.error(f"Error encoding sentence batch {i//batch_size + 1}: {e}")
            zero_embeddings = torch.zeros((len(batch_sentences), model.get_sentence_embedding_dimension()), dtype=torch.float32)
            all_embeddings.append(zero_embeddings)
        finally:
            del batch_embeddings_tensor
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        log_memory_usage(f"After sentence batch {i//batch_size + 1}")

    embeddings = torch.cat(all_embeddings, dim=0)
    model.to('cpu')
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    log_memory_usage("After generating sentence embeddings")
    return embeddings

def _load_model_from_path(model_type, local_model_base_path, device_for_load):
    log_memory_usage(f"Before loading {model_type} model")
    model, tokenizer = None, None
    hf_model_identifier = None
    is_sentence_transformer_model = False

    if model_type == 'protein': hf_model_identifier = "Rostlab/prot_bert_bfd"
    elif model_type == 'nucleotide': hf_model_identifier = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
    elif model_type == 'sentence':
        hf_model_identifier = 'sentence-transformers/all-MiniLM-L6-v2'
        is_sentence_transformer_model = True
    
    load_from_path_str = hf_model_identifier

    if local_model_base_path and hf_model_identifier:
        expanded_local_base = os.path.expanduser(local_model_base_path)
        cache_dir_name = f"models--{hf_model_identifier.replace('/', '--')}"
        potential_local_model_dir = os.path.join(expanded_local_base, cache_dir_name)
        
        actual_model_path_to_check = potential_local_model_dir
        if os.path.isdir(potential_local_model_dir):
            snapshots_dir = os.path.join(potential_local_model_dir, "snapshots")
            if os.path.isdir(snapshots_dir):
                snapshot_subdirs = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d)) and not d.startswith('.')]
                if snapshot_subdirs:
                    snapshot_paths = sorted([os.path.join(snapshots_dir, d) for d in snapshot_subdirs], key=lambda p: os.path.getmtime(p), reverse=True)
                    actual_model_path_to_check = snapshot_paths[0]
            
            if os.path.isdir(actual_model_path_to_check) and \
               (os.path.exists(os.path.join(actual_model_path_to_check, "config.json")) or \
                (is_sentence_transformer_model and (os.path.exists(os.path.join(actual_model_path_to_check, "modules.json")) or os.path.exists(os.path.join(actual_model_path_to_check, "sentence_bert_config.json"))))):
                logging.info(f"Found valid local model for '{hf_model_identifier}' at: {actual_model_path_to_check}")
                load_from_path_str = actual_model_path_to_check
            else:
                logging.info(f"Local path '{actual_model_path_to_check}' not a valid model. Will try Hugging Face Hub.")
        else:
            logging.info(f"Local base model directory '{potential_local_model_dir}' not found. Will try Hugging Face Hub.")

    if not load_from_path_str:
        logging.error(f"No model identifier or path for model_type '{model_type}'.")
        return None, None

    try:
        logging.info(f"Attempting to load {model_type} model from: {load_from_path_str}")
        if is_sentence_transformer_model:
            model = SentenceTransformer(load_from_path_str, device=str(device_for_load))
            tokenizer = None
        else:
            tokenizer = AutoTokenizer.from_pretrained(load_from_path_str)
            model = AutoModel.from_pretrained(load_from_path_str)
        log_memory_usage(f"After loading {model_type} model from {load_from_path_str}")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Failed to load {model_type} model from '{load_from_path_str}': {e}", exc_info=True)
        return None, None

def _unload_model_from_mem(model, tokenizer=None):
    model_name = model.__class__.__name__ if model else "N/A"
    log_memory_usage(f"Before unloading {model_name}")
    del model
    if tokenizer: del tokenizer
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    logging.info(f"Unloaded model: {model_name} and cleared cache.")
    log_memory_usage(f"After unloading {model_name}")

def _get_texts_for_node_list(node_list_for_type, field_name):
    texts = []
    for node_data in node_list_for_type:
        raw_value = node_data.get(field_name)
        text_to_append = ''
        if raw_value is not None and not pd.isna(raw_value):
            text_to_append = str(raw_value) if not isinstance(raw_value, str) else raw_value
        texts.append(text_to_append)
    return texts

def generate_features_for_merged_nodes(merged_nodes_dict, embed_size, local_model_base_path, device_str, batch_size_embed):
    logging.info("Starting feature generation for merged nodes...")
    log_memory_usage("Start generate_features_for_merged_nodes")

    used_device = torch.device(device_str if torch.cuda.is_available() and device_str.startswith("cuda") else "cpu")
    logging.info(f"Feature generation will use device: {used_device}")

    nodes_by_type_map = defaultdict(list)
    for node_id, node_data_item in merged_nodes_dict.items():
        node_data_with_id = node_data_item.copy()
        node_data_with_id['_original_id_for_feature_gen_'] = node_id
        nodes_by_type_map[node_data_item['Type']].append(node_data_with_id)

    all_node_features_map = {}

    for ntype, nodes_in_type_list in nodes_by_type_map.items():
        num_nodes_for_type = len(nodes_in_type_list)
        if num_nodes_for_type == 0:
            logging.debug(f"Skipping feature generation for type '{ntype}' as it has 0 nodes.")
            continue

        log_memory_usage(f"Start processing ntype '{ntype}' for feature generation")
        logging.info(f"Generating features for node type: '{ntype}' ({num_nodes_for_type} nodes)")
        
        primary_embeddings_tensor = None
        comment_embeddings_tensor = None
        raw_embed_size = 0
        
        if ntype == 'protein':
            model, tokenizer = _load_model_from_path('protein', local_model_base_path, used_device)
            if model:
                sequences = _get_texts_for_node_list(nodes_in_type_list, 'Sequence')
                if sequences and any(s.strip() for s in sequences):
                    primary_embeddings_tensor = _generate_transformer_embeddings(model, tokenizer, sequences, batch_size_embed, used_device)
                    raw_embed_size = primary_embeddings_tensor.shape[1]
                _unload_model_from_mem(model, tokenizer)
        elif ntype in ['gene', 'dna', 'rna']:
            model, tokenizer = _load_model_from_path('nucleotide', local_model_base_path, used_device)
            if model:
                sequences = _get_texts_for_node_list(nodes_in_type_list, 'Sequence')
                if sequences and any(s.strip() for s in sequences):
                    primary_embeddings_tensor = _generate_transformer_embeddings(model, tokenizer, sequences, batch_size_embed, used_device)
                    raw_embed_size = primary_embeddings_tensor.shape[1]
                _unload_model_from_mem(model, tokenizer)
        elif ntype in ['compound', 'metabolite']:
            smiles_list = _get_texts_for_node_list(nodes_in_type_list, 'Sequence')
            if smiles_list and any(s.strip() for s in smiles_list):
                primary_embeddings_tensor = _generate_rdkit_embeddings(smiles_list, RDKIT_FP_BITS, used_device)
                raw_embed_size = primary_embeddings_tensor.shape[1]

        if primary_embeddings_tensor is not None and raw_embed_size != embed_size:
            logging.info(f"Projecting primary {ntype} embeddings from {raw_embed_size} to {embed_size}...")
            projection_primary = nn.Linear(raw_embed_size, embed_size).to(used_device)
            nn.init.xavier_uniform_(projection_primary.weight)
            primary_embeddings_tensor = projection_primary(primary_embeddings_tensor.to(used_device)).cpu().detach()
            del projection_primary
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        elif primary_embeddings_tensor is not None:
             primary_embeddings_tensor = primary_embeddings_tensor.cpu().detach()

        comments = _get_texts_for_node_list(nodes_in_type_list, 'Comments')
        if comments and any(c.strip() for c in comments):
            sentence_model, _ = _load_model_from_path('sentence', local_model_base_path, used_device)
            if sentence_model:
                comment_embeddings_tensor = _generate_sentence_embeddings(sentence_model, comments, batch_size_embed, used_device)
                comment_embeddings_tensor = comment_embeddings_tensor.cpu().detach()
                _unload_model_from_mem(sentence_model)
        
        combined_input_list = []
        if primary_embeddings_tensor is not None:
            combined_input_list.append(primary_embeddings_tensor)
        else:
            logging.info(f"No primary sequence/fingerprint for '{ntype}'. Using default random tensor as base.")
            default_base_tensor = torch.randn(num_nodes_for_type, embed_size)
            combined_input_list.append(default_base_tensor)

        if comment_embeddings_tensor is not None:
            if comment_embeddings_tensor.shape[0] == num_nodes_for_type:
                 combined_input_list.append(comment_embeddings_tensor)
            else:
                logging.warning(f"Comment embedding shape mismatch for {ntype}. Skipping comments.")
        
        if not combined_input_list:
            logging.warning(f"No embeddings generated for {ntype}. Creating zero tensor.")
            final_ntype_tensor = torch.zeros((num_nodes_for_type, embed_size), dtype=torch.float32)
        else:
            combined_tensor_cpu = torch.cat([t.cpu() for t in combined_input_list], dim=1)
            combined_dim = combined_tensor_cpu.shape[1]

            if combined_dim != embed_size:
                logging.info(f"Projecting combined {ntype} embeddings from {combined_dim} to {embed_size}...")
                projection_final = nn.Linear(combined_dim, embed_size)
                nn.init.xavier_uniform_(projection_final.weight)
                final_ntype_tensor = projection_final(combined_tensor_cpu).detach()
                del projection_final
            else:
                final_ntype_tensor = combined_tensor_cpu.detach()
        
        for i, node_data_with_id in enumerate(nodes_in_type_list):
            original_node_id = node_data_with_id['_original_id_for_feature_gen_']
            all_node_features_map[original_node_id] = final_ntype_tensor[i]
        
        del primary_embeddings_tensor, comment_embeddings_tensor, final_ntype_tensor
        gc.collect()
        log_memory_usage(f"End processing ntype '{ntype}' for feature generation")

    log_memory_usage("End generate_features_for_merged_nodes")
    return all_node_features_map

def replace_data_type(existing_type):
    if existing_type is None or existing_type == "":
        return ""

    etype_lower = existing_type.lower()

    if etype_lower in [
        "environment", "ro", "species", "pmid", "doi", "isbn", "wikipedia",
        "ncit", "emapa", "envo", "oxford", "rnordv", "maxo", "ontology", "mirbase",
        "ogms", "treatment", "gocl", "gov", "goa", "hea", "issn", "jstor", "medline",
        "mitre", "msh", "pimd", "pmc", "poc", "po_ref", "sabio-rk", "vz", "wb_ref",
        "wiki", "http", "https", "misc", "unknown"
    ]:
        return "misc"
    if etype_lower in [
        "refseq_protein", "pfam", "uniprot", "interpro", "prosite", "merops_fam",
        "corum", "intact", "iuphar_gpcr", "iuphar_receptor", "uniprotkb-kw", "protein"
    ]:
        return "protein"
    if etype_lower in ["rna", "refseq_rna", "mrna", "rrna", "rnamods"]:
        return "rna"
    if etype_lower in [
        "hgnc", "ensembl", "clinvar", "sequence", "chromosome", "gene", "refseq_genome",
        "fb", "zfin", "geneontology"
    ]:
        return "gene"
    if etype_lower in ["omim", "mim", "mesh"]:
        return "disease"
    if etype_lower in [
        "chebi", "pubchem_compound", "chemical", "foodon",
        "ligand", "resid", "um-bbd_reactionid", "compound"
    ]:
        return "compound"
    if etype_lower in ["cl", "cell_line", "cell", "clm", "go_cellularcomponent", "iao", "nif_subcellular", "po", "so"]:
        return "cell_component"
    if etype_lower in ["go_biologicalprocess", "go_ref", "goc", "ma", "organism"]:
        return "biological_process"
    if etype_lower in ["nbo", "mfomd", "go_molecularfunction", "ec", "rhea", "um-bbd_enzymeid", "tc"]:
        return "molecular_function"
    if etype_lower in ["phenotype", "hsapdv", "mp", "pato"]:
        return "phenotype"
    if etype_lower in [
        "kegg", "pathway", "biocyc", "kegg_pathway", "metacyc", "reactome",
        "um-bbd_pathwayid", "unipathway", "kegg.module"
    ]:
        return "pathway"
    if etype_lower in ["fma", "anatomy", "ddanat", "fbbt", "uberon", "wbbt", "xao", "zfa"]:
        return "anatomy"
    logging.debug(f"No specific mapping for type '{existing_type}', keeping original.")
    return "Unknown"

def merge_field(existing_value, new_value, delimiter=MERGE_DELIMITER):
    existing_str = existing_value if existing_value else ""
    new_str = new_value if new_value else ""

    existing_parts = set(p.strip() for p in existing_str.split(delimiter) if p.strip())
    new_parts = set(p.strip() for p in new_str.split(delimiter) if p.strip())

    all_parts = sorted(list(existing_parts.union(new_parts)))
    return delimiter.join(all_parts)

def process_node_files(node_files, generate_features, embed_size, local_model_base_path, device_str, batch_size_embed):
    merged_nodes = {}
    embeded_nodes = {}
    nodes_skipped_missing_data = 0
    nodes_deduplicated = 0

    for file_path in node_files:
        logging.info(f"Processing node file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8', newline='') as infile:
                delimiter = ','

                reader = csv.DictReader(infile, delimiter=delimiter)

                if not reader.fieldnames:
                    logging.warning(f"File {file_path} appears to be empty or could not read header. Skipping.")
                    continue

                actual_header = reader.fieldnames
                if not all(field in actual_header for field in FIELDNAMES_NODE):
                    logging.warning(f"File {file_path} header {actual_header} doesn't contain all expected fields {FIELDNAMES_NODE}. Processing based on available fields.")

                line_num = 1
                for row in reader:
                    line_num += 1
                    try:
                        node_id = row.get("ID")
                        node_name = row.get("Name")
                        node_type = row.get("Type")
                        node_seq = row.get("Sequence", "")
                        node_comments = row.get("Comments", "")

                        if not node_id:
                            logging.warning(f"Skipping row {line_num} in {file_path}: Missing 'ID'. Row: {row}")
                            nodes_skipped_missing_data += 1
                            continue
                        if not node_name:
                            logging.warning(f"Row {line_num} in {file_path} (ID: {node_id}): Missing 'Name'. Using empty string.")
                            node_name = ""

                        current_node_data = {
                            "ID": node_id,
                            "Name": node_name,
                            "Type": replace_data_type(node_type),
                            "Sequence": node_seq if node_seq else "",
                            "Comments": node_comments if node_comments else ""
                        }

                        if node_id not in merged_nodes:
                            merged_nodes[node_id] = current_node_data
                        else:
                            nodes_deduplicated += 1
                            existing_node = merged_nodes[node_id]
                            is_identical = (existing_node["Name"] == current_node_data["Name"] and
                                            (existing_node["Sequence"] or "") == (current_node_data["Sequence"] or "") and
                                            (existing_node["Comments"] or "") == (current_node_data["Comments"] or ""))

                            if not is_identical:
                                if existing_node["Name"] == current_node_data["Name"]:
                                    merged_seq = merge_field(existing_node["Sequence"], current_node_data["Sequence"])
                                    merged_comments = merge_field(existing_node["Comments"], current_node_data["Comments"])
                                    merged_nodes[node_id]["Sequence"] = merged_seq
                                    merged_nodes[node_id]["Comments"] = merged_comments
                                    logging.debug(f"Merged node ID {node_id} from {file_path}.")
                                else:
                                    logging.warning(f"Node ID {node_id} has conflicting names: '{existing_node['Name']}' vs '{current_node_data['Name']}' in {file_path}. Keeping the first encountered name.")
                    except Exception as e:
                        logging.error(f"Error processing row {line_num} in {file_path}: {e} - Row data: {row}")

        except FileNotFoundError:
            logging.error(f"Node file not found: {file_path}")
        except Exception as e:
            logging.error(f"Failed to read or process node file {file_path}: {e}")

    if generate_features and merged_nodes:
        logging.info("Starting feature generation for all merged nodes...")
        node_id_to_feature_tensor = generate_features_for_merged_nodes(
            merged_nodes, 
            embed_size, 
            local_model_base_path, 
            device_str,
            batch_size_embed
        )
        logging.info(f"Generated features for {len(node_id_to_feature_tensor)} nodes.")
        
        updated_count = 0
        for node_id, feature_tensor in node_id_to_feature_tensor.items():
            if node_id in merged_nodes:
                feature_string = ";".join(map(lambda x: f"{x:.4f}", feature_tensor.tolist()))
                current_node_data = {
                    "ID": node_id,
                    "Name": node_name,
                    "Type": replace_data_type(node_type),
                    "Feature": feature_string,
                }
                embeded_nodes[node_id] = current_node_data
                updated_count +=1
            else:
                logging.warning(f"Node ID {node_id} from feature generation not found in merged_nodes. This should not happen.")
        logging.info(f"Updated 'Sequence' and 'Comments' for {updated_count} nodes with generated features.")
    elif generate_features:
        logging.warning("Feature generation was requested, but no nodes were merged. Skipping feature generation.")

    logging.info(f"Finished processing node files. Found {len(merged_nodes)} unique node IDs. Skipped {nodes_skipped_missing_data} rows due to missing data. Deduplicated {nodes_deduplicated} nodes. Final generated nodes with features: {len(embeded_nodes)}.")
    return embeded_nodes

def process_edge_files(edge_files, merged_nodes_with_feature_data):
    virtual_nodes = {}
    unique_edges = []
    seen_edge_keys = set()
    skipped_unknown_count = 0
    edges_skipped_missing_data = 0
    edges_deduplicated = 0
    edges_replaced_data_type = 0

    for file_path in edge_files:
        logging.info(f"Processing edge file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8', newline='') as infile:
                delimiter = ','
                reader = csv.DictReader(infile, delimiter=delimiter)

                if not reader.fieldnames:
                    logging.warning(f"File {file_path} appears to be empty or could not read header. Skipping.")
                    continue

                actual_header = reader.fieldnames
                if not all(field in actual_header for field in FIELDNAMES_EDGE):
                    logging.warning(f"File {file_path} header {actual_header} doesn't contain all expected fields {FIELDNAMES_EDGE}. Processing based on available fields.")

                line_num = 1
                for row in reader:
                    line_num += 1
                    try:
                        node1_id_raw = row.get("Node 1 ID")
                        node2_id_raw = row.get("Node 2 ID")
                        direction = row.get("Edge Direction")
                        original_node1_type_from_edge = row.get("Node 1 Data Type")
                        original_node2_type_from_edge = row.get("Node 2 Data Type")

                        if not node1_id_raw or not node2_id_raw or direction is None:
                            logging.warning(f"Skipping row {line_num} in {file_path}: Missing key fields (Node 1 ID, Node 2 ID, or Edge Direction). Row: {row}")
                            edges_skipped_missing_data += 1
                            continue
                        if not original_node1_type_from_edge or not original_node2_type_from_edge:
                            logging.warning(f"Skipping row {line_num} in {file_path} (Nodes: {node1_id_raw}, {node2_id_raw}): Missing 'Node 1 Data Type' or 'Node 2 Data Type'. Row: {row}")
                            edges_skipped_missing_data += 1
                            continue
                        
                        node1_id = node1_id_raw
                        node2_id = node2_id_raw

                        replaced_node1_type_from_edge = replace_data_type(original_node1_type_from_edge)
                        replaced_node2_type_from_edge = replace_data_type(original_node2_type_from_edge)

                        if replaced_node1_type_from_edge.lower() == 'unknown' or replaced_node2_type_from_edge.lower() == 'unknown':
                            logging.debug(f"Skipping edge ({node1_id}, {node2_id}) from {file_path}: Replaced type from edge file is 'unknown' for one or both nodes ('{replaced_node1_type_from_edge}', '{replaced_node2_type_from_edge}').")
                            skipped_unknown_count += 1
                            continue

                        edge_key = (node1_id, node2_id, direction)

                        if edge_key not in seen_edge_keys:
                            seen_edge_keys.add(edge_key)

                            edge_data = {field: row.get(field, "") for field in FIELDNAMES_EDGE}
                            edge_data["Node 1 ID"] = node1_id
                            edge_data["Node 2 ID"] = node2_id
                            edge_data["Node 1 Data Type"] = replaced_node1_type_from_edge
                            edge_data["Node 2 Data Type"] = replaced_node2_type_from_edge

                            authoritative_node1_type = replaced_node1_type_from_edge
                            if node1_id in merged_nodes_with_feature_data:
                                authoritative_node1_type = merged_nodes_with_feature_data[node1_id]['Type']
                                edges_replaced_data_type += 1
                            elif node1_id in virtual_nodes:
                                authoritative_node1_type = virtual_nodes[node1_id]['Type']
                                edges_replaced_data_type += 1
                            edge_data["Node 1 Data Type"] = authoritative_node1_type

                            authoritative_node2_type = replaced_node2_type_from_edge
                            if node2_id in merged_nodes_with_feature_data:
                                authoritative_node2_type = merged_nodes_with_feature_data[node2_id]['Type']
                                edges_replaced_data_type += 1
                            elif node2_id in virtual_nodes:
                                authoritative_node2_type = virtual_nodes[node2_id]['Type']
                                edges_replaced_data_type += 1
                            edge_data["Node 2 Data Type"] = authoritative_node2_type

                            unique_edges.append(edge_data)

                            if node1_id not in merged_nodes_with_feature_data and node1_id not in virtual_nodes:
                                virtual_nodes[node1_id] = {
                                    "ID": node1_id,
                                    "Name": node1_id,
                                    "Type": replaced_node1_type_from_edge,
                                    "Comments": "",
                                    "Sequence": ""
                                }
                            if node2_id not in merged_nodes_with_feature_data and node2_id not in virtual_nodes:
                                virtual_nodes[node2_id] = {
                                    "ID": node2_id,
                                    "Name": node2_id,
                                    "Type": replaced_node2_type_from_edge,
                                    "Comments": "",
                                    "Sequence": ""
                                }
                        else:
                            logging.debug(f"Duplicate edge ({node1_id}, {node2_id}) found in {file_path}, skipping.")
                    except Exception as e:
                        logging.error(f"Error processing row {line_num} in {file_path}: {e} - Row data: {row}")

        except FileNotFoundError:
            logging.error(f"Edge file not found: {file_path}")
        except Exception as e:
            logging.error(f"Failed to read or process edge file {file_path}: {e}")

    if skipped_unknown_count > 0:
        logging.info(f"Skipped {skipped_unknown_count} edges because one or both original node types (from edge file, after replacement) were 'unknown'.")
    logging.info(f"Finished processing edge files. Found {len(unique_edges)} unique, valid edges. Skipped {edges_skipped_missing_data} rows due to missing data. Deduplicated {edges_deduplicated} edges, replaced {edges_replaced_data_type} node types based on merged nodes data.")
    return unique_edges, virtual_nodes

def write_output_csv(data, output_path, fieldnames):
    logging.info(f"Writing {len(data)} rows to {output_path}")
    try:
        with open(output_path, 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL, delimiter=',')
            writer.writeheader()
            writer.writerows(data)
        logging.info(f"Successfully wrote output to {output_path}")
    except Exception as e:
        logging.error(f"Failed to write output file {output_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Merge node and edge data from multiple CSV files.")
    parser.add_argument('-n', '--node-file', nargs='+', required=True, help="Paths to input node CSV file.")
    parser.add_argument('--out-nodes', required=True, help="Path for the merged output node CSV file.")
    parser.add_argument('--verbose', '-v', action='store_true', help="Enable debug logging.")
    parser.add_argument('--generate-features', action='store_true', help="Enable generation of node features, replacing 'Sequence' and 'Comments' fields.")
    parser.add_argument('--embed-size', type=int, default=DEFAULT_EMBED_SIZE, help=f"Target embedding size for generated features (default: {DEFAULT_EMBED_SIZE}).")
    parser.add_argument('--local-model-path', type=str, default=DEFAULT_LOCAL_MODEL_PATH, help=f"Base path for local Hugging Face models (default: {DEFAULT_LOCAL_MODEL_PATH}).")
    parser.add_argument('--device', type=str, default="cpu", help="Device for feature generation (e.g., 'cpu', 'cuda:0') (default: 'cpu').")
    parser.add_argument('--batch-size-embed', type=int, default=DEFAULT_BATCH_SIZE_FOR_EMBEDDING, help=f"Batch size for embedding generation models (default: {DEFAULT_BATCH_SIZE_FOR_EMBEDDING}).")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    merged_nodes_with_feature_data = process_node_files(
        args.node_file,
        args.generate_features,
        args.embed_size,
        args.local_model_path,
        args.device,
        args.batch_size_embed
    )

    nodes_to_write = list(merged_nodes_with_feature_data.values())
    write_output_csv(nodes_to_write, args.out_nodes, FIELDNAMES_NODE_EMBED)
    logging.info("Data embeding process completed.")

if __name__ == "__main__":
    main()
