import pandas as pd
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
import os
import logging
import gc  # Garbage collector interface
from Model.common import *
import multiprocessing
from functools import partial

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    AutoTokenizer = None
    AutoModel = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
except ImportError:
    Chem = None
    DataStructs = None
    AllChem = None

from Model.graph_splitter import (
    RandomEdgeSplitter,
    NodeRandomSplitter,
    NodeKFoldSplitter,
    SpecificIDEdgeSplitter,
    ExternalEdgePairSplitter,
    resolve_typed_node_ids,
    CategoryEdgeSplitter,
    DiseaseCentricSplitter,
    IndependentCohortSplitter,
    TypeExclusionSplitter,
    KFoldEdgeSplitter,
    SplitResult,
)

# This should be consistent with download_models.py
MODEL_MAPPING = {
    'protein': 'facebook/esm2_t6_8M_UR50D',
    'nucleotide': 'InstaDeepAI/nucleotide-transformer-v2-50m-multi-species', # For gene, dna, rna
    'smiles': 'seyonec/PubChem10M_SMILES_BPE_450k', # For compound, metabolite, drug
    'sentence': 'sentence-transformers/all-MiniLM-L6-v2', # For comments
    'default_text': 'albert-base-v2' # For other text-based node types
}

FEATURE_CACHE_SCHEMA_VERSION = 2

NODE_NAME_FIELD_CANDIDATES = (
    'name',
    'disease_name',
    'disease',
    'mondo_name',
    'display_name',
    'preferred_name',
    'label',
    'title',
)


def _torch_load_trusted_cache(path, map_location=None):
    """Load local BioGT cache files that may contain numpy arrays under PyTorch 2.6+."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


# Helper function for multiprocessing RDKit fingerprint generation
# This function should be defined at the top level of the module or be picklable
def _process_single_smiles_for_rdkit(smiles, n_bits):
    """
    Processes a single SMILES string to generate an RDKit Morgan fingerprint.
    Returns a NumPy array of the fingerprint or a zero vector on failure.
    """
    if Chem is None:
        return np.zeros((n_bits,), dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
            arr = np.zeros((n_bits,), dtype=np.float32) # Ensure consistent dtype and shape
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        except Exception as e:
            print(f"Error generating fingerprint for SMILES (worker): {smiles} - {e}")
            return np.zeros((n_bits,), dtype=np.float32)
    else:
        return np.zeros((n_bits,), dtype=np.float32)


def _get_node_record_by_original_id(id_to_node_data, orig_id):
    if not id_to_node_data:
        return {}

    candidate_keys = []
    for raw_key in (orig_id, str(orig_id).strip()):
        if raw_key not in candidate_keys:
            candidate_keys.append(raw_key)

    try:
        numeric_key = int(str(orig_id).strip())
        if numeric_key not in candidate_keys:
            candidate_keys.append(numeric_key)
        numeric_str_key = str(numeric_key)
        if numeric_str_key not in candidate_keys:
            candidate_keys.append(numeric_str_key)
    except (TypeError, ValueError):
        pass

    for candidate_key in candidate_keys:
        if candidate_key in id_to_node_data:
            return id_to_node_data[candidate_key]

    return {}


def _extract_node_display_name(node_record):
    if not isinstance(node_record, dict) or not node_record:
        return None

    normalized_record = {str(key).strip().lower(): value for key, value in node_record.items()}
    for field_name in NODE_NAME_FIELD_CANDIDATES:
        raw_value = normalized_record.get(field_name)
        if raw_value is None:
            continue

        cleaned_value = str(raw_value).strip()
        if cleaned_value and cleaned_value.lower() != 'nan':
            return cleaned_value

    return None


def _build_name_lookup_from_dgl_mapping(dgl_to_orig, id_to_node_data):
    if not dgl_to_orig or not id_to_node_data:
        return {}

    name_lookup = {}
    for dgl_id, orig_id in dgl_to_orig.items():
        node_record = _get_node_record_by_original_id(id_to_node_data, orig_id)
        node_name = _extract_node_display_name(node_record)
        if node_name:
            name_lookup[node_name] = int(dgl_id)

    return name_lookup


class DGLGraphCreator(object):
    """
    Class to create a DGL graph from CSV files containing node and edge information.
    Supports saving and loading pre-computed node features and graph data to/from a cache directory.
    Generates node features by combining sequence embeddings (if applicable),
    comment embeddings, and default learnable embeddings, projecting to a final size.
    Uses RDKit for compound/metabolite fingerprints if available.
    """
    def __init__(self, node_csv_path, edge_csv_path, embed_size=128, val_size=0.075, test_size=0.075, random_state=42,
                 freeze_pretrained=True, graph_cache_path=None, feature_cache_path=None, device=None,
                 local_model_base_path=None,
                 generate_embeddings=True,
                 use_sequence_features=True,
                 merged_sequence_features=False,
                 split_strategy="random",
                 split_category_map=None,
                 split_num_folds=5,
                 split_fold_index=0,
                 split_val_fold_index=None,
                 split_disease_node_type="disease",
                 split_exclude_types=None,
                 split_exclude_action="train",
                 split_specific_ids_path=None,
                 split_external_edges_path=None,
                 split_cohort_ids_path=None,
                 split_val_cohort_ids_path=None,
                 split_holdout_scope="target_edges",
                 split_target_input=None,
                 split_target_output=None,
                 use_evidence_weights=True,
                 source_ablation_mode="none",
                 source_ablation_sources=None,
                 source_ablation_match="any_endpoint",
                 graph_node_types=None,
                 graph_relation_types=None,
                 filter_isolated_nodes=True,
                 regenerate_features_after_filter=False):  # New parameter for isolated-node filtering
        self.node_csv_path = node_csv_path
        self.edge_csv_path = edge_csv_path
        self.default_embed_size = embed_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        self.freeze_pretrained = freeze_pretrained
        self.graph_cache_path = graph_cache_path
        self.feature_cache_path = feature_cache_path
        self.device = device if device else torch.device('cpu')  # Store the device, default to CPU if None
        self.local_model_base_path = os.path.expanduser(local_model_base_path) if local_model_base_path else None
        self.generate_embeddings = generate_embeddings  # Store the new parameter
        self.use_sequence_features = use_sequence_features # Store the new parameter
        self.merged_sequence_features = merged_sequence_features # Store the new parameter for using sentence transformer for sequences
        self._data_is_filtered = False # Initialize filter status flag
        self.split_strategy = split_strategy
        self.split_category_map = split_category_map or {}
        self.split_num_folds = split_num_folds
        self.split_fold_index = split_fold_index
        self.split_val_fold_index = split_val_fold_index
        self.fold_splits = None
        self.split_metadata = None
        self.excluded_eid_dict = {}
        self.split_disease_node_type = split_disease_node_type
        self.split_exclude_types = list(split_exclude_types) if split_exclude_types else []
        self.split_exclude_action = split_exclude_action
        self.split_specific_ids_path = split_specific_ids_path
        self.split_external_edges_path = split_external_edges_path
        self.split_cohort_ids_path = split_cohort_ids_path
        self.split_val_cohort_ids_path = split_val_cohort_ids_path
        self.split_holdout_scope = split_holdout_scope
        self.split_target_input = split_target_input
        self.split_target_output = split_target_output
        self.use_evidence_weights = bool(use_evidence_weights)
        self.source_ablation_mode = (source_ablation_mode or "none").strip().lower()
        self.source_ablation_sources = {
            str(source).strip().lower()
            for source in (source_ablation_sources or [])
            if str(source).strip()
        }
        self.source_ablation_match = (source_ablation_match or "any_endpoint").strip().lower()
        self.graph_node_types = {
            str(node_type).strip().lower()
            for node_type in (graph_node_types or [])
            if str(node_type).strip()
        }
        self.graph_relation_types = {
            str(relation_type).strip().lower()
            for relation_type in (graph_relation_types or [])
            if str(relation_type).strip()
        }
        node_holdout_strategies = {
            "node_random", "node_kfold", "specific_ids", "external_edges"
        }
        if filter_isolated_nodes and (split_strategy or "").lower() in node_holdout_strategies:
            logging.warning(
                "Disabling isolated-node filtering for split strategy '%s' so "
                "held-out typed nodes remain available for evaluation.",
                split_strategy,
            )
            self.filter_isolated_nodes = False
        else:
            self.filter_isolated_nodes = filter_isolated_nodes
        self.regenerate_features_after_filter = regenerate_features_after_filter

        self._cache_graphs_file = 'graphs.bin'
        self._cache_metadata_file = 'metadata.pt'
        self._cache_features_file = 'features.pt'

        self.node_id_map = {}
        self.node_name_map = {}
        self.node_type_alias_map = {}
        self.node_type_map = {}
        self.node_dglid_to_orig = {}
        self.node_original_data = {}
        self.graph_data = {}
        self.edge_list_dict = {}
        self.edge_weights_full = {}
        self.train_weights_data = {}
        self.val_weights_data = {}
        self.test_weights_data = {}
        self.num_nodes_dict = {}
        self.node_features = None
        self._node_features_loaded_from_cache = False
        self.node_info = {}
        self.g_full = None
        self.g_train = None
        self.g_val = None
        self.g_test = None
        self.train_eid_dict = {}
        self.val_eid_dict = {}
        self.test_eid_dict = {}
        self.nodes_df = None
        self.id_to_node_data = {}
        self._loaded_from_cache = False

        if self.graph_cache_path:
            logging.info(f"Checking for cached data in: {self.graph_cache_path}")
            if self._load_from_cache(self.graph_cache_path): # This attempts to load graph, metadata, and features
                self._loaded_from_cache = True # Indicates graph/metadata might be from cache.

                # CRITICAL CHECK: Verify if node_features were actually loaded and are valid.
                # self.node_features is populated by _load_cached_features (called by _load_from_cache)
                valid_features_present = False
                if hasattr(self, 'node_features') and self.node_features and isinstance(self.node_features, nn.ModuleDict) and self.node_features:
                    # Check if it contains at least one valid nn.Embedding with a positive embedding dimension
                    if any(isinstance(emb, nn.Embedding) and emb.weight.numel() > 0 and emb.embedding_dim > 0 for emb in self.node_features.values()):
                        valid_features_present = True
                
                if valid_features_present:
                    logging.info("Successfully loaded all essential data (graph, metadata, features) from cache.")
                    log_memory_usage("Initialization complete (from cache)", self.device)
                    return # All good, return early
                else:
                    # Features were not successfully loaded from cache, even if graph/metadata might have been.
                    logging.warning("Cache load for graph/metadata may have succeeded, but node features are missing or invalid. "
                                    "Proceeding to ensure data and features are fully processed.")
                    # Fall through to the main data loading and feature creation logic.
                    # self._loaded_from_cache might still be True if graph/metadata part was okay,
                    # but we need to ensure the rest of the initialization runs.
            else: # _load_from_cache returned False (e.g., graph or metadata files missing/corrupt)
                logging.info("Cache not found or main cache loading (graph/metadata) failed. Proceeding with full data processing.")
                self._loaded_from_cache = False
        else: # No graph_cache_path
            logging.info("No graph_cache_path specified. Proceeding with full data processing.")
            self._loaded_from_cache = False

        # This part of the code will now be reached if:
        # 1. No cache path was specified.
        # 2. _load_from_cache (for graph/metadata) returned False.
        # 3. _load_from_cache returned True, but the 'valid_features_present' check above failed, and we fell through.

        log_memory_usage("Before data loading", self.device)
        # If primary data (nodes_df etc.) wasn't loaded by a successful cache operation.
        # The self._loaded_from_cache flag helps determine if _load_and_process_data is needed.
        # If _loaded_from_cache is True here, it means graph/metadata was loaded, but features failed the check.
        # If _loaded_from_cache is False, nothing was loaded from cache, or a major part failed.
        if not (self._loaded_from_cache and hasattr(self, 'nodes_df') and self.nodes_df is not None and hasattr(self, 'num_nodes_dict') and self.num_nodes_dict):
            # This condition ensures _load_and_process_data runs if cache didn't fully load graph data,
            # or if there was no cache attempt / major cache failure.
            logging.info("Primary graph data (nodes_df, etc.) not loaded from cache or incomplete. Calling _load_and_process_data.")
            self._load_and_process_data()
        else:
            logging.info("Primary graph data (nodes_df, etc.) seems to be loaded from cache.")
            
        log_memory_usage("After data loading", self.device)

        # Ensure features are created/loaded if not already validly present from a successful cache load
        # (which would have caused an early return).
        final_check_valid_features = False
        if hasattr(self, 'node_features') and self.node_features and isinstance(self.node_features, nn.ModuleDict) and self.node_features:
            if any(isinstance(emb, nn.Embedding) and emb.weight.numel() > 0 and emb.embedding_dim > 0 for emb in self.node_features.values()):
                final_check_valid_features = True

        if not final_check_valid_features:
            logging.info("Node features are missing or invalid after cache attempts or initial load; calling _create_or_load_node_features.")
            self._create_or_load_node_features() # This has the fallback to create default embeddings
        else:
            logging.info("Node features were already successfully loaded/created and validated.")

        log_memory_usage("After feature creation/loading", self.device)
        self._split_edges()
        log_memory_usage("After edge splitting", self.device)
        self._build_graphs()
        log_memory_usage("After graph building", self.device)
        
        # Optional: regenerate node features after filtering to match filtered graph structure.
        # Disabled by default for compatibility with previous training behavior.
        if self._data_is_filtered and self.regenerate_features_after_filter:
            logging.info("Regenerating node features to match filtered graph structure...")
            self._create_or_load_node_features(force_recreate=True)
            log_memory_usage("After node features regeneration", self.device)
        elif self._data_is_filtered:
            logging.info("Skipping node feature regeneration after filtering (compatibility mode).")
        else:
            logging.info("Skipping node feature regeneration because isolated-node filtering is disabled.")
        
        self._prepare_node_info()
        log_memory_usage("Initialization complete (full processing)", self.device)

        if self.graph_cache_path and not self._loaded_from_cache:
            logging.info(f"Saving processed data to cache (with filtered graph structure): {self.graph_cache_path}")
            self._save_to_cache(self.graph_cache_path)

    def _get_text_by_dgl_id(self, ntype, field):
        """
        Retrieves a list of text (sequence or comments) for a given node type,
        indexed by DGL node ID. Ensures all returned text items are strings.
        """
        if ntype not in self.node_dglid_to_orig:
            logging.warning(f"Node type '{ntype}' not found in node_dglid_to_orig.")
            return [''] * int(self.num_nodes_dict.get(ntype, 0))

        text_list = []
        expected_nodes = int(self.num_nodes_dict.get(ntype, 0))
        missing_mapping_count = 0
        missing_data_count = 0
        for dgl_id in range(expected_nodes):
            orig_id = self.node_dglid_to_orig[ntype].get(dgl_id)
            if orig_id is None:
                missing_mapping_count += 1
                text_list.append('')
                continue
            node_data = self.id_to_node_data.get(orig_id, {})
            if not node_data:
                missing_data_count += 1
            raw_value = node_data.get(field)  # Get the raw value for the field

            text_to_append = ''  # Default to empty string
            if raw_value is not None and not pd.isna(raw_value):  # Check for None and pandas NA
                if not isinstance(raw_value, str):
                    logging.debug(f"Node {orig_id} (type {ntype}) has non-string data for field '{field}': {raw_value} (type: {type(raw_value)}). Converting to string.")
                    text_to_append = str(raw_value)
                else:
                    text_to_append = raw_value

            text_list.append(text_to_append)
        if missing_mapping_count or missing_data_count:
            logging.warning(
                "[%s] Field '%s': filled %d missing DGL-id mappings and %d missing node records with empty text.",
                ntype,
                field,
                missing_mapping_count,
                missing_data_count,
            )
        return text_list

    def _get_joint_node_texts(self, ntype):
        """Serialize all available fields for each node into one semantic input."""
        if ntype not in self.node_dglid_to_orig:
            return []

        joint_texts = []
        priority_fields = ("type", "id", "name", "comments")
        for dgl_id in sorted(self.node_dglid_to_orig[ntype].keys()):
            original_id = self.node_dglid_to_orig[ntype][dgl_id]
            node_record = _get_node_record_by_original_id(
                self.id_to_node_data,
                original_id,
            )
            normalized_record = {
                str(key).strip().lower(): value
                for key, value in node_record.items()
            }
            normalized_record.setdefault("id", original_id)
            normalized_record.setdefault("type", ntype)

            ordered_fields = list(priority_fields)
            ordered_fields.extend(
                field
                for field in sorted(normalized_record)
                if field not in priority_fields and field != "sequence"
            )
            if self.use_sequence_features:
                ordered_fields.append("sequence")

            parts = []
            for field in ordered_fields:
                raw_value = normalized_record.get(field)
                if raw_value is None or pd.isna(raw_value):
                    continue
                value = str(raw_value).strip()
                if not value:
                    continue
                if field == "sequence":
                    value = value[:1024]
                parts.append(f"{field}: {value}")
            joint_texts.append(" | ".join(parts))
        return joint_texts

    def _deterministic_feature_projection(self, features, output_dim):
        """
        Reduce pretrained features without an untrained projection module that
        is discarded before optimization.
        """
        input_dim = int(features.shape[1])
        output_dim = int(output_dim)
        if input_dim == output_dim:
            return features.float()
        if input_dim < output_dim:
            return F.pad(features.float(), (0, output_dim - input_dim))

        generator = torch.Generator(device="cpu").manual_seed(
            int(self.random_state) + input_dim * 1009 + output_dim * 9176
        )
        projection = torch.randn(
            input_dim,
            output_dim,
            generator=generator,
            dtype=torch.float32,
        ) / (output_dim ** 0.5)
        projected_chunks = []
        for start in range(0, features.shape[0], 8192):
            projected_chunks.append(
                features[start : start + 8192].float().matmul(projection)
            )
        projected = torch.cat(projected_chunks, dim=0)
        return F.layer_norm(projected, (output_dim,))

    def _clean_sequence(self, sequence, sequence_type):
        """
        Cleans a biological sequence by removing invalid characters based on its type.
        """
        if not isinstance(sequence, str):
            return ""

        sequence = sequence.upper()
        
        if sequence_type == 'protein':
            # Standard amino acids + 'U' (Selenocysteine) and 'O' (Pyrrolysine)
            # and common ambiguity codes. We are permissive here.
            valid_chars = "ACDEFGHIKLMNPQRSTVWYUOBZX"
        elif sequence_type == 'nucleotide':
            # Standard DNA/RNA bases + ambiguity codes
            valid_chars = "GATCRYWSMKHBVDN"
        else: # Includes SMILES and other text, which we don't want to alter
            return sequence

        cleaned = "".join(c for c in sequence if c in valid_chars)
        
        if len(cleaned) != len(sequence):
            logging.debug(f"Cleaned sequence (type: {sequence_type}). Original len: {len(sequence)}, Cleaned len: {len(cleaned)}")
            
        return cleaned

    def _create_or_load_node_features(self, force_recreate=False):
        log_memory_usage("Start create_or_load_node_features", self.device)
        loaded = False
        self._node_features_loaded_from_cache = False
        
        if force_recreate:
            logging.info("Force recreating node features (skipping cache load)...")
        else:
            features_path = self._get_feature_cache_filepath(self.graph_cache_path)
            if features_path:
                logging.info(f"Checking node feature cache: {features_path}")
            if features_path and self._load_node_features(features_path):
                logging.info("Using loaded node features from cache.")
                loaded = True
                self._node_features_loaded_from_cache = True
            elif features_path:
                logging.info("Could not load features from cache path. Will create new features.")

        if not loaded:
            logging.info("Creating new features...")
            self._create_node_features()
            if self.feature_cache_path:
                features_path = self._get_feature_cache_filepath(self.graph_cache_path)
                logging.info("Saving newly created node features to shared cache: %s", features_path)
                self._save_node_features(features_path)

        log_memory_usage("End create_or_load_node_features", self.device)

    def _get_feature_cache_filepath(self, cache_dir=None):
        if self.feature_cache_path:
            expanded = os.path.expanduser(self.feature_cache_path)
            if expanded.endswith(".pt"):
                os.makedirs(os.path.dirname(expanded) or ".", exist_ok=True)
                return expanded
            os.makedirs(expanded, exist_ok=True)
            return os.path.join(expanded, self._cache_features_file)
        if cache_dir:
            _, _, features_path = self._get_cache_filepaths(cache_dir)
            return features_path
        return None

    def _create_node_features(self):
        logging.info("Creating node features...")
        log_memory_usage("Start _create_node_features", self.device)
        features = {}
        
        if not self.generate_embeddings:
            logging.info("Skipping pre-trained/RDKit embedding generation. Creating default learnable embeddings.")
            for ntype, num_nodes in self.num_nodes_dict.items():
                if num_nodes == 0:
                    logging.debug(f"Skipping node type '{ntype}' as it has 0 nodes.")
                    continue
                logging.info(f"Creating default learnable nn.Embedding for node type: '{ntype}' ({num_nodes} nodes) with size {self.default_embed_size}")
                emb_layer = nn.Embedding(num_nodes, self.default_embed_size)
                nn.init.xavier_uniform_(emb_layer.weight)
                emb_layer.weight.requires_grad = True
                features[ntype] = emb_layer
            
            self.node_features = nn.ModuleDict(features)
            logging.info(f"Default node feature creation complete. ModuleDict keys: {list(self.node_features.keys())}")
            log_memory_usage("End _create_node_features (default embeddings)", self.device)
            return

        def load_model(model_type):
            log_memory_usage(f"Before loading {model_type} model", self.device)
            model, tokenizer = None, None
            
            hf_model_identifier = MODEL_MAPPING.get(model_type)
            if not hf_model_identifier:
                logging.error(f"No model identifier found for model type: {model_type}")
                return None, None

            is_sentence_transformer_model = 'sentence-transformers' in hf_model_identifier
            
            # --- Construct potential local paths ---
            # Path format 1: As saved by our custom sentence-transformer logic (e.g., "sentence-transformers_all-MiniLM-L6-v2")
            path_format_1 = hf_model_identifier.replace("/", "_")
            
            # Path format 2: As saved by huggingface_hub's snapshot_download (e.g., "models--facebook--esm2_t6_8M_UR50D")
            path_format_2 = f"models--{hf_model_identifier.replace('/', '--')}"

            load_path = None
            
            if self.local_model_base_path:
                potential_path_1 = os.path.join(self.local_model_base_path, path_format_1)
                potential_path_2 = os.path.join(self.local_model_base_path, path_format_2)

                if os.path.isdir(potential_path_1):
                    logging.info(f"Found local model directory for '{model_type}' at: {potential_path_1}")
                    load_path = potential_path_1
                elif os.path.isdir(potential_path_2):
                    logging.info(f"Found Hugging Face cache-style directory for '{model_type}' at: {potential_path_2}")
                    # This is likely a directory from snapshot_download. The actual model files are often in a subdirectory.
                    # First, check for a 'snapshots' subdirectory, which is the most common structure.
                    snapshots_dir = os.path.join(potential_path_2, 'snapshots')
                    if os.path.isdir(snapshots_dir) and os.listdir(snapshots_dir):
                        try:
                            latest_snapshot = sorted(os.listdir(snapshots_dir))[-1]
                            snapshot_path = os.path.join(snapshots_dir, latest_snapshot)
                            if os.path.isdir(snapshot_path):
                                logging.info(f"Using latest snapshot directory: {snapshot_path}")
                                load_path = snapshot_path
                            else:
                                logging.warning(f"Snapshot path {snapshot_path} is not a directory. Falling back to parent.")
                                load_path = potential_path_2
                        except (IndexError, FileNotFoundError):
                            logging.warning("Could not find a valid snapshot directory. Falling back to parent directory.")
                            load_path = potential_path_2
                    else:
                        # If 'snapshots' dir is missing or empty, it might be an older cache format
                        # where files are in 'blobs' and referenced by 'refs'.
                        # The from_pretrained method can handle the parent directory if it's a valid cache layout.
                        logging.info("No 'snapshots' directory found or it is empty. Using the parent directory as load path.")
                        load_path = potential_path_2

            if load_path:
                logging.info(f"Loading model and tokenizer for '{model_type}' from local path: '{load_path}'...")
            else:
                # Fallback to original identifier if no local path is found
                load_path = hf_model_identifier
                logging.warning(f"No pre-downloaded local directory found for '{hf_model_identifier}'. "
                                f"Attempting to load using identifier and cache_dir='{self.local_model_base_path}'. "
                                "This may still attempt network access if not fully cached by Hugging Face.")

            try:
                if is_sentence_transformer_model:
                    if SentenceTransformer is None:
                        logging.error(
                            "sentence-transformers is not installed; "
                            "cannot generate joint text features."
                        )
                        return None, None
                    # For SentenceTransformer, passing a local path directly is the most robust way for offline loading.
                    model = SentenceTransformer(load_path, device=self.device)
                    tokenizer = None # SentenceTransformer handles its own tokenization
                else:
                    if AutoTokenizer is None or AutoModel is None:
                        logging.error(
                            "transformers is not installed; cannot generate "
                            "specialized sequence features."
                        )
                        return None, None
                    # For standard transformers, passing the local path works for offline.
                    # If load_path is an identifier, cache_dir will be used.
                    tokenizer = AutoTokenizer.from_pretrained(load_path, cache_dir=self.local_model_base_path, trust_remote_code=True)
                    model = AutoModel.from_pretrained(load_path, cache_dir=self.local_model_base_path, trust_remote_code=True)
                
                logging.info(f"Successfully loaded model for '{model_type}'.")
                return model, tokenizer
            except Exception as e:
                logging.error(f"Error loading model for '{model_type}' from '{load_path}': {e}", exc_info=True)
                return None, None

        def unload_model(model, tokenizer=None):
            model_name = model.__class__.__name__ if model else "N/A"
            log_memory_usage(f"Before unloading {model_name}", self.device)
            del model
            if tokenizer:
                del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logging.info(f"Unloaded model: {model_name} and cleared cache.")
            log_memory_usage(f"After unloading {model_name}", self.device)

        # Load the sentence transformer once for all text fields
        sentence_model, _ = load_model('sentence')
        
        if sentence_model is None:
            logging.error("CRITICAL: Failed to load sentence transformer model. Text features will be skipped!")
        else:
            logging.info(f"Successfully loaded sentence transformer model: {sentence_model.__class__.__name__}")

        for ntype, num_nodes in self.num_nodes_dict.items():
            if num_nodes == 0:
                logging.debug(f"Skipping node type '{ntype}' as it has 0 nodes.")
                continue

            log_memory_usage(f"Start processing ntype '{ntype}'", self.device)
            logging.info(f"Processing features for node type: '{ntype}' ({num_nodes} nodes)")
            target_embed_size = self.default_embed_size
            
            # This will hold all feature tensors for the current node type before concatenation
            combined_features = []
            logging.info(f"[{ntype}] Starting with empty combined_features list")

            # --- 1. Sequence-based Embeddings (Primary) ---
            sequence_embeddings_tensor = None
            is_sequence_type = False
            
            model_type_for_sequence = None
            if ntype == 'protein':
                model_type_for_sequence = 'protein'
            elif ntype in ['gene', 'dna', 'rna']:
                model_type_for_sequence = 'nucleotide'
            elif ntype in ['compound', 'metabolite', 'drug']:
                model_type_for_sequence = 'smiles'

            if model_type_for_sequence:
                is_sequence_type = True
                sequences = self._get_text_by_dgl_id(ntype, field='sequence')
                logging.info(f"[{ntype}] Found {len(sequences)} sequence values for '{model_type_for_sequence}' type")
                if sequences and any(s.strip() for s in sequences):
                    logging.info(f"[{ntype}] Processing sequences (non-empty count: {sum(1 for s in sequences if s.strip())})")
                    if not self.use_sequence_features:
                        # Completely skip sequence features when disabled
                        logging.info(f"[{ntype}] Sequence features DISABLED - skipping sequence processing")
                    elif self.merged_sequence_features:
                        logging.info(
                            f"[{ntype}] Sequence will be embedded together with "
                            "all node metadata in merged mode"
                        )
                    else:
                        # Use specialized sequence models (protein/nucleotide/SMILES) - default behavior
                        logging.info(f"[{ntype}] Using SPECIALIZED model for sequences")
                        sequences = [self._clean_sequence(s, model_type_for_sequence) for s in sequences]
                        
                        model, tokenizer = load_model(model_type_for_sequence)
                        if model and tokenizer:
                            sequence_embeddings_tensor = self._generate_transformer_embeddings(model, tokenizer, sequences)
                            unload_model(model, tokenizer)
                            if sequence_embeddings_tensor is not None:
                                logging.info(f"[{ntype}] Generated SPECIALIZED sequence embeddings: shape={sequence_embeddings_tensor.shape}")
                                combined_features.append(sequence_embeddings_tensor)
                else:
                    logging.info(f"[{ntype}] No non-empty sequences found")

            # --- 2. One semantic embedding over every available node field ---
            joint_texts = self._get_joint_node_texts(ntype)
            non_empty_count = sum(1 for text in joint_texts if text.strip())
            logging.info(
                f"[{ntype}] Joint metadata input: {len(joint_texts)} total, "
                f"{non_empty_count} non-empty"
            )
            if joint_texts and non_empty_count > 0 and sentence_model:
                joint_embeddings_tensor = self._generate_sentence_embeddings(
                    sentence_model,
                    joint_texts,
                )
                if joint_embeddings_tensor is not None:
                    logging.info(
                        f"[{ntype}] Generated joint all-information embeddings: "
                        f"shape={joint_embeddings_tensor.shape}"
                    )
                    combined_features.append(joint_embeddings_tensor)
            elif not sentence_model:
                logging.warning(
                    f"[{ntype}] sentence_model is None; joint metadata was not embedded"
                )

            logging.info(f"[{ntype}] Total features collected: {len(combined_features)} tensors")
            for i, feat in enumerate(combined_features):
                logging.info(f"[{ntype}]   Feature {i}: shape={feat.shape}")

            # --- 3. Fallback / Default Embedding ---
            # If after all attempts, there are no features, create a default learnable embedding.
            # This ensures every node type has features.
            if not combined_features:
                logging.warning(f"[{ntype}] No features collected! Creating default learnable embeddings of size {target_embed_size}")
                default_emb = nn.Embedding(num_nodes, target_embed_size)
                nn.init.xavier_uniform_(default_emb.weight)
                final_embeddings_tensor = default_emb.weight.data.clone()
                logging.info(f"[{ntype}] Created default embedding: shape={final_embeddings_tensor.shape}")
                # Skip the rest of the combination logic
                
            else:
                # --- 4. Feature Combination and Projection ---
                # Concatenate all generated features
                logging.info(f"[{ntype}] Concatenating {len(combined_features)} feature tensors")
                combined_embeddings_tensor = torch.cat(combined_features, dim=1)

                combined_dim = combined_embeddings_tensor.shape[1]
                logging.info(f"[{ntype}] Combined dimension after concatenation: {combined_dim}")

                final_embeddings_tensor = None
                if combined_dim != target_embed_size:
                    logging.info(
                        f"[{ntype}] Deterministically projecting joint features "
                        f"{combined_dim} -> {target_embed_size}"
                    )
                    final_embeddings_tensor = self._deterministic_feature_projection(
                        combined_embeddings_tensor,
                        target_embed_size,
                    )
                    logging.info(f"[{ntype}] After projection: shape={final_embeddings_tensor.shape}")
                    log_memory_usage(f"After final projection for '{ntype}'", self.device)
                else:
                    logging.info(f"[{ntype}] No projection needed ({combined_dim} == {target_embed_size})")
                    final_embeddings_tensor = combined_embeddings_tensor

            final_emb_layer = nn.Embedding(num_nodes, target_embed_size)
            final_emb_layer.weight.data.copy_(final_embeddings_tensor.detach())

            # Freeze embeddings if they are pretrained sequence types, otherwise make them trainable
            final_emb_layer.weight.requires_grad = not (self.freeze_pretrained and is_sequence_type)
            logging.info(f"[{ntype}] Final embedding layer created: shape={final_emb_layer.weight.shape}, trainable={final_emb_layer.weight.requires_grad}")

            features[ntype] = final_emb_layer
            log_memory_usage(f"After creating final embedding layer for '{ntype}'", self.device)

            # Clean up intermediate tensors to free memory
            # Only delete variables that were actually created
            if 'sequence_embeddings_tensor' in locals():
                del sequence_embeddings_tensor
            if 'combined_embeddings_tensor' in locals():
                del combined_embeddings_tensor
            if 'final_embeddings_tensor' in locals():
                del final_embeddings_tensor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log_memory_usage(f"End processing ntype '{ntype}'", self.device)

        # Unload the sentence model after processing all node types
        if sentence_model:
            unload_model(sentence_model)

        self.node_features = nn.ModuleDict(features)

        logging.info(f"Node feature creation complete. Final ModuleDict keys: {list(self.node_features.keys())}")
        logging.info(f"All final node embedding features have dimension: {self.default_embed_size}")
        log_memory_usage("End _create_node_features", self.device)

    def _generate_transformer_embeddings(self, model, tokenizer, texts, batch_size=32):
        """
        Generates embeddings for a list of texts using a standard Hugging Face Transformer model.
        Processes in batches to manage memory.
        """
        model.to(self.device)
        model.eval()
        all_embeddings = []
        
        logging.info(f"Generating Transformer embeddings for {len(texts)} texts in batches of {batch_size}...")
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Synchronize before moving new data to GPU to catch errors from previous iterations
            if torch.cuda.is_available() and self.device.type == "cuda":
                torch.cuda.synchronize()

            # Ensure all items in batch are strings
            batch_texts = [str(t) if t is not None else "" for t in batch_texts]

            try:
                inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
                
                # --- Pre-flight check for CUDA assert ---
                # A common cause for "device-side assert" is input_ids being out of range for the model's embedding layer.
                vocab_size = model.config.vocab_size
                if (inputs['input_ids'] >= vocab_size).any():
                    logging.error(f"CUDA Assert Pre-flight Check FAILED: input_ids out of vocabulary range (vocab size: {vocab_size}).")
                    
                    # Find all problematic tokens and their locations
                    problematic_indices = torch.where(inputs['input_ids'] >= vocab_size)
                    
                    # Log details for each problematic text in the batch
                    for i in range(problematic_indices[0].shape[0]):
                        batch_idx = problematic_indices[0][i].item()
                        token_idx = problematic_indices[1][i].item()
                        problematic_token_id = inputs['input_ids'][batch_idx, token_idx].item()
                        problematic_text = batch_texts[batch_idx]
                        logging.error(
                            f"Problem in batch index {batch_idx}, token index {token_idx}: "
                            f"Token ID {problematic_token_id} >= vocab size {vocab_size}. "
                            f"Text: '{problematic_text}'"
                        )

                    # Replace all out-of-bounds IDs with the tokenizer's unknown token ID
                    unk_token_id = tokenizer.unk_token_id
                    if unk_token_id is None:
                        # Fallback to 0 if unk_token_id is not defined (a safe guess for many tokenizers)
                        unk_token_id = 0
                        logging.warning("Tokenizer does not have a designated 'unk_token_id'. Falling back to token ID 0.")
                    
                    logging.warning(f"Replacing all out-of-bounds token IDs with UNK token ID: {unk_token_id}")
                    inputs['input_ids'][inputs['input_ids'] >= vocab_size] = unk_token_id

                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Mean pooling of the last hidden state
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
                # Force CUDA synchronization to catch errors at the source
                if torch.cuda.is_available() and self.device.type == "cuda":
                    torch.cuda.synchronize()

                all_embeddings.append(embeddings.cpu()) # Move to CPU immediately
            except Exception as e:
                logging.error(f"Error processing batch {i//batch_size}. The error was: {e}", exc_info=True)
                logging.error(f"The batch texts that caused the error were: {batch_texts}")
                # Append a zero tensor of the correct shape if an error occurs
                hidden_size = model.config.hidden_size
                all_embeddings.append(torch.zeros(len(batch_texts), hidden_size, device='cpu'))

            log_memory_usage(f"After processing batch {i//batch_size}", self.device)

        embeddings = torch.cat(all_embeddings, dim=0)
        model.to('cpu') # Move model back to CPU
        log_memory_usage("After generating transformer embeddings", self.device)
        return embeddings

    def _generate_sentence_embeddings(self, model, texts, batch_size=64):
        """
        Generates embeddings for a list of texts using a SentenceTransformer model.
        """
        # The SentenceTransformer model is already on the correct device from when it was loaded
        logging.info(f"Generating Sentence-BERT embeddings for {len(texts)} texts...")
        log_memory_usage("Before generating sentence embeddings", self.device)
        
        # Ensure all texts are strings
        texts = [str(t) if t is not None else "" for t in texts]

        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)
        
        # Move embeddings to CPU to free up GPU memory for other models
        embeddings = embeddings.cpu()
        log_memory_usage("After generating sentence embeddings", self.device)
        return embeddings

    def _load_node_features(self, features_path):
        """
        Loads node features from a cache file.
        Reconstructs the self.node_features ModuleDict structure based on
        self.num_nodes_dict and the keys found in the cached state_dict.
        """
        try:
            if not os.path.exists(features_path):
                logging.warning(f"Node features cache file not found: {features_path}")
                return False

            logging.info(f"Loading node features from {features_path}...")
            # state_dict_from_cache is the state_dict of the nn.ModuleDict saved previously
            loaded_data = _torch_load_trusted_cache(features_path, map_location=self.device)

            if not isinstance(loaded_data, dict) or 'node_features_state_dict' not in loaded_data:
                logging.error(f"Cached node features file {features_path} is not in the expected format (missing 'node_features_state_dict' key).")
                return False
            if loaded_data.get('feature_schema_version') != FEATURE_CACHE_SCHEMA_VERSION:
                logging.warning(
                    "Node feature cache schema is %s, expected %s. "
                    "Only node features will be regenerated.",
                    loaded_data.get('feature_schema_version', 'legacy'),
                    FEATURE_CACHE_SCHEMA_VERSION,
                )
                return False
            state_dict_from_cache = loaded_data['node_features_state_dict']

            reconstructed_node_features = nn.ModuleDict()

            # Part 1: Reconstruct nn.Embedding layers for node types
            # These are the main embedding layers for each node type.
            # In _create_node_features, these are initialized to self.default_embed_size.
            for ntype, num_nodes in self.num_nodes_dict.items():
                if num_nodes == 0:
                    continue
                
                embedding_weight_key = f"{ntype}.weight"
                if embedding_weight_key in state_dict_from_cache:
                    # The embedding dimension for the final ntype embedding should be self.default_embed_size
                    cached_embedding_dim = state_dict_from_cache[embedding_weight_key].shape[1]
                    cached_num_nodes = state_dict_from_cache[embedding_weight_key].shape[0]
                    
                    if cached_embedding_dim != self.default_embed_size:
                        logging.error(
                            f"CACHE MISMATCH: Cached embedding dimension for node type '{ntype}' is {cached_embedding_dim}, "
                            f"but current default_embed_size is {self.default_embed_size}. "
                            f"This cache is incompatible and must be regenerated. Aborting cache load."
                        )
                        return False  # Abort cache loading, force regeneration
                    
                    # CRITICAL: Check if the number of nodes has changed (e.g., due to filtering)
                    if cached_num_nodes != num_nodes:
                        logging.error(
                            f"CACHE MISMATCH: Number of nodes for '{ntype}' in cache is {cached_num_nodes}, "
                            f"but current num_nodes_dict has {num_nodes}. "
                            f"This indicates the graph structure has changed (likely due to node filtering or data modifications). "
                            f"The cache is incompatible and must be regenerated. Aborting cache load."
                        )
                        return False  # Abort cache loading, force regeneration
                    
                    reconstructed_node_features[ntype] = nn.Embedding(num_nodes, self.default_embed_size)
                # If not in cache, this ntype might not have had a direct embedding saved,
                # or it's an error in the cache. Validation later will catch missing ntypes.

            # Part 2: Reconstruct nn.Linear projection layers
            # Projection layers are named e.g., "protein_final_projection"
            # Their weights are "protein_final_projection.weight"
            # We identify them by looking for '.weight' keys where the module name isn't an ntype already processed.
            
            module_names_in_cache = set()
            for key in state_dict_from_cache.keys():
                module_names_in_cache.add(key.split('.')[0]) # Get module name, e.g., 'protein' or 'protein_final_projection'

            for module_name in module_names_in_cache:
                if module_name not in self.num_nodes_dict: # It's not a primary ntype embedding
                    weight_key = f"{module_name}.weight"
                    bias_key = f"{module_name}.bias"
                    if weight_key in state_dict_from_cache:
                        in_features = state_dict_from_cache[weight_key].shape[1]
                        out_features = state_dict_from_cache[weight_key].shape[0]
                        reconstructed_node_features[module_name] = nn.Linear(in_features, out_features)
            
            # Now reconstructed_node_features has the necessary module structure. Load the weights.
            try:
                reconstructed_node_features.load_state_dict(state_dict_from_cache)
            except RuntimeError as e:
                logging.error(f"RuntimeError loading state_dict into reconstructed node_features: {e}", exc_info=True)
                logging.warning("This may be due to a mismatch between cached features and current model/data structure.")
                return False

            self.node_features = reconstructed_node_features
            logging.info("Node features loaded successfully from cache using reconstructed structure.")

            # Validation: Ensure all ntypes (from num_nodes_dict) with nodes have a feature module.
            # _create_node_features ensures a final_emb_layer for each ntype with nodes,
            # stored directly under the ntype key in self.node_features.
            for ntype, num_nodes in self.num_nodes_dict.items():
                if num_nodes > 0 and ntype not in self.node_features:
                    logging.error(f"Validation failed: Node type '{ntype}' has nodes but no corresponding feature module was loaded from cache.")
                    self.node_features = None # Invalidate on failure
                    return False
            
            return True

        except FileNotFoundError: # Should be caught by os.path.exists, but as a safeguard.
            logging.warning(f"Node features cache file not found (FileNotFoundError): {features_path}")
            self.node_features = None
            return False
        except Exception as e:
            logging.error(f"General error occurred while loading node features from {features_path}: {e}", exc_info=True)
            self.node_features = None # Ensure it's cleared on any error
            return False

    def _save_node_features(self, features_path):
        """
        Saves node features to a cache file.
        """
        if self.node_features is None:
            logging.warning("No node features to save to cache.")
            return

        try:
            logging.info(f"Saving node features to {features_path}...")
            # We need to save the state_dict of the nn.ModuleDict (or its components)
            # and potentially metadata needed to reconstruct them (e.g., embedding dimensions).
            
            # Saving the state_dict of the nn.ModuleDict itself
            features_state_dict = self.node_features.state_dict()

            # It's good practice to also save info needed for reconstruction if not easily inferred
            # For nn.Embedding, num_embeddings can be inferred from num_nodes_dict,
            # and embedding_dim from default_embed_size.
            # For nn.Linear (projections), in_features/out_features are part of the state_dict's weight shape.
            
            # For simplicity, we'll just save the state_dict.
            # A more robust cache would include parameters to reconstruct each layer.
            torch.save(
                {
                    'feature_schema_version': FEATURE_CACHE_SCHEMA_VERSION,
                    'node_features_state_dict': features_state_dict,
                },
                features_path,
            )
            logging.info("Node features saved successfully to cache.")
            log_memory_usage("After saving node features to cache", self.device)

        except Exception as e:
            logging.error(f"Error saving node features to {features_path}: {e}", exc_info=True)
            if os.path.exists(features_path):
                os.remove(features_path)

    def _get_cache_filepaths(self, cache_dir):
        graphs_path = os.path.join(cache_dir, self._cache_graphs_file)
        metadata_path = os.path.join(cache_dir, self._cache_metadata_file)
        features_path = os.path.join(cache_dir, self._cache_features_file)
        return graphs_path, metadata_path, features_path

    def _get_cache_settings(self):
        return {
            'use_evidence_weights': self.use_evidence_weights,
            'source_ablation_mode': self.source_ablation_mode,
            'source_ablation_sources': sorted(self.source_ablation_sources),
            'source_ablation_match': self.source_ablation_match,
            'graph_node_types': sorted(self.graph_node_types),
            'graph_relation_types': sorted(self.graph_relation_types),
        }

    def _is_default_cache_settings(self):
        return self._get_cache_settings() == {
            'use_evidence_weights': True,
            'source_ablation_mode': 'none',
            'source_ablation_sources': [],
            'source_ablation_match': 'any_endpoint',
            'graph_node_types': [],
            'graph_relation_types': [],
        }

    def _cleanup_cache(self, cache_dir):
        """
        Removes all cache files to force regeneration.
        This is called when cache is detected to be incompatible.
        """
        try:
            graphs_path, metadata_path, features_path = self._get_cache_filepaths(cache_dir)
            
            # Remove graph cache file
            if os.path.exists(graphs_path):
                os.remove(graphs_path)
                logging.info(f"Removed incompatible cache file: {graphs_path}")
            
            # Remove metadata file (single file format)
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
                logging.info(f"Removed incompatible cache file: {metadata_path}")
            
            # Remove metadata directory (new format)
            metadata_dir = metadata_path.replace('.pt', '')
            if os.path.isdir(metadata_dir):
                import shutil
                shutil.rmtree(metadata_dir)
                logging.info(f"Removed incompatible cache directory: {metadata_dir}")
            
            # Remove features file
            if os.path.exists(features_path):
                os.remove(features_path)
                logging.info(f"Removed incompatible cache file: {features_path}")
                
            logging.info("Cache cleanup completed. Cache will be regenerated on next run.")
        except Exception as e:
            logging.error(f"Error during cache cleanup: {e}", exc_info=True)


    def _load_from_cache(self, cache_dir):
        graphs_path, metadata_path, features_path = self._get_cache_filepaths(cache_dir)

        metadata_dir = metadata_path.replace('.pt', '')
        if not os.path.exists(graphs_path) or (not os.path.exists(metadata_path) and not os.path.isdir(metadata_dir)):
            logging.warning("Cache files (graphs.bin or metadata.pt/metadata directory) not found.")
            return False

        try:
            logging.info(f"Loading graphs from {graphs_path}...")
            g_list, _ = dgl.load_graphs(graphs_path)
            if len(g_list) != 4:
                logging.error(f"Expected 4 graphs in cache, found {len(g_list)}. Cache is invalid.")
                return False
            self.g_full, self.g_train, self.g_val, self.g_test = g_list[0], g_list[1], g_list[2], g_list[3]
            logging.info("Graphs loaded successfully.")

            metadata_dir = metadata_path.replace('.pt', '')
            
            # 检查是否使用新的分开存储的元数据格式
            if os.path.isdir(metadata_dir):
                logging.info(f"Loading metadata from directory: {metadata_dir}")
                self.node_info = _torch_load_trusted_cache(os.path.join(metadata_dir, 'node_info.pt'))
                self.train_eid_dict = _torch_load_trusted_cache(os.path.join(metadata_dir, 'train_eid_dict.pt'))
                self.val_eid_dict = _torch_load_trusted_cache(os.path.join(metadata_dir, 'val_eid_dict.pt'))
                self.test_eid_dict = _torch_load_trusted_cache(os.path.join(metadata_dir, 'test_eid_dict.pt'))
                self.num_nodes_dict = _torch_load_trusted_cache(os.path.join(metadata_dir, 'num_nodes_dict.pt'))
                split_metadata_path = os.path.join(metadata_dir, 'split_metadata.pt')
                fold_splits_path = os.path.join(metadata_dir, 'fold_splits.pt')
                excluded_eids_path = os.path.join(metadata_dir, 'excluded_eid_dict.pt')
                self.split_metadata = (
                    _torch_load_trusted_cache(split_metadata_path)
                    if os.path.exists(split_metadata_path)
                    else None
                )
                self.fold_splits = (
                    _torch_load_trusted_cache(fold_splits_path)
                    if os.path.exists(fold_splits_path)
                    else None
                )
                self.excluded_eid_dict = (
                    _torch_load_trusted_cache(excluded_eids_path)
                    if os.path.exists(excluded_eids_path)
                    else {}
                )
                # Load other attributes from node_info
                self.node_id_map = self.node_info.get('id_map', {})
                self.node_name_map = self.node_info.get('name_map', {})
                self.node_type_map = self.node_info.get('type_map', {})
                self.node_type_alias_map = self.node_info.get('type_alias_map', {})
                self.node_dglid_to_orig = self.node_info.get('dglid_to_orig', {})
                self.id_to_node_data = self.node_info.get('id_to_node_data', {})
            else:
                logging.info(f"Loading metadata from single file: {metadata_path}")
                metadata = _torch_load_trusted_cache(metadata_path)
                self.node_info = metadata['node_info']
                self.train_eid_dict = metadata['train_eid_dict']
                self.val_eid_dict = metadata['val_eid_dict']
                self.test_eid_dict = metadata['test_eid_dict']
                self.num_nodes_dict = metadata['num_nodes_dict']
                self.split_metadata = metadata.get('split_metadata')
                self.fold_splits = metadata.get('fold_splits')
                self.excluded_eid_dict = metadata.get('excluded_eid_dict', {})
                self.node_id_map = self.node_info.get('id_map', {})
                self.node_name_map = self.node_info.get('name_map', {})
                self.node_type_map = self.node_info.get('type_map', {})
                self.node_type_alias_map = self.node_info.get('type_alias_map', {})
                self.node_dglid_to_orig = self.node_info.get('dglid_to_orig', {})
                self.id_to_node_data = self.node_info.get('id_to_node_data', {})

            self._normalize_loaded_name_map()

            if not all([self.node_info, self.train_eid_dict is not None, self.val_eid_dict is not None, self.test_eid_dict is not None, self.num_nodes_dict]):
                logging.error("One or more metadata components are missing from cache. Cache is invalid.")
                return False
            logging.info("Metadata loaded successfully.")

            # Load and check filter status *before* loading node features
            is_filtered_from_cache = False
            filter_status_file = os.path.join(metadata_dir, 'is_filtered.pt')
            if (os.path.exists(filter_status_file)):
                is_filtered_from_cache = _torch_load_trusted_cache(filter_status_file)
                logging.info(f"Loaded filter status from cache: {is_filtered_from_cache}")
            else:
                logging.warning("Filter status file not found in cache. Assuming data is not filtered.")

            if self.filter_isolated_nodes != is_filtered_from_cache:
                logging.warning(
                    "Cached filter status (%s) does not match requested filter_isolated_nodes (%s). "
                    "Ignoring cache to rebuild graphs with correct filtering.",
                    is_filtered_from_cache,
                    self.filter_isolated_nodes
                )
                self.g_full = self.g_train = self.g_val = self.g_test = None
                self.node_info = self.train_eid_dict = self.val_eid_dict = self.test_eid_dict = self.num_nodes_dict = None
                self.node_features = None
                return False

            cache_settings_file = os.path.join(metadata_dir, 'cache_settings.pt')
            requested_cache_settings = self._get_cache_settings()
            if os.path.exists(cache_settings_file):
                cached_cache_settings = torch.load(cache_settings_file)
                if cached_cache_settings != requested_cache_settings:
                    logging.warning(
                        "Cached graph settings %s do not match requested graph settings %s. Ignoring cache.",
                        cached_cache_settings,
                        requested_cache_settings,
                    )
                    self.g_full = self.g_train = self.g_val = self.g_test = None
                    self.node_info = self.train_eid_dict = self.val_eid_dict = self.test_eid_dict = self.num_nodes_dict = None
                    self.node_features = None
                    return False
            elif not self._is_default_cache_settings():
                logging.warning(
                    "Cache settings file not found, but non-default graph settings were requested (%s). Ignoring cache.",
                    requested_cache_settings,
                )
                self.g_full = self.g_train = self.g_val = self.g_test = None
                self.node_info = self.train_eid_dict = self.val_eid_dict = self.test_eid_dict = self.num_nodes_dict = None
                self.node_features = None
                return False

            if not is_filtered_from_cache:
                logging.warning("Cached data is not marked as filtered. This might lead to issues if isolated nodes were not handled.")

            features_path = self._get_feature_cache_filepath(cache_dir)
            if not self._load_node_features(features_path):
                logging.warning(
                    "Cached graph metadata is valid but node features are stale. "
                    "Regenerating only features while preserving graph/split caches."
                )
                self._create_node_features()
                self._save_node_features(features_path)

            logging.info("Node features (if present in cache and compatible) loaded successfully from cache.")
            return True

        except Exception as e:
            logging.error(f"Error loading data from cache directory {cache_dir}: {e}", exc_info=True)
            logging.warning("Cleaning up incompatible cache to force regeneration...")
            self._cleanup_cache(cache_dir)
            self.g_full = self.g_train = self.g_val = self.g_test = None
            self.node_info = self.train_eid_dict = self.val_eid_dict = self.test_eid_dict = self.num_nodes_dict = None
            self.node_features = None
            return False

    def _normalize_loaded_name_map(self):
        """Ensure the cached name_map follows the canonical structure keyed by node type."""
        if not self.node_dglid_to_orig or not self.id_to_node_data:
            logging.warning("Cannot rebuild name_map due to missing metadata (dglid_to_orig or id_to_node_data).")
            return

        rebuilt_name_map = {}
        sample_value = next(iter(self.node_name_map.values())) if self.node_name_map else None
        if isinstance(sample_value, dict):
            rebuilt_name_map.update(self.node_name_map)
        elif self.node_name_map:
            logging.info("Legacy name_map structure detected in cache. Rebuilding canonical mapping.")

        rebuilt_any = False
        for ntype, dgl_mapping in self.node_dglid_to_orig.items():
            existing_lookup = rebuilt_name_map.get(ntype, {})
            if isinstance(existing_lookup, dict) and existing_lookup:
                continue

            name_lookup = _build_name_lookup_from_dgl_mapping(dgl_mapping, self.id_to_node_data)
            if name_lookup:
                rebuilt_name_map[ntype] = name_lookup
                rebuilt_any = True

        if rebuilt_name_map:
            self.node_name_map = rebuilt_name_map
            if rebuilt_any:
                logging.info("Rebuilt or filled missing name_map entries from cached metadata.")

        if self.g_full is not None:
            type_alias_map = {}
            for canonical_type in self.g_full.ntypes:
                type_alias_map[canonical_type] = canonical_type
                lower_alias = canonical_type.lower()
                type_alias_map.setdefault(lower_alias, canonical_type)
            self.node_type_alias_map = type_alias_map

        if isinstance(self.node_info, dict):
            self.node_info['id_map'] = self.node_id_map
            self.node_info['name_map'] = self.node_name_map
            self.node_info['type_map'] = self.node_type_map
            self.node_info['type_alias_map'] = self.node_type_alias_map
            self.node_info['dglid_to_orig'] = self.node_dglid_to_orig
            self.node_info['id_to_node_data'] = self.id_to_node_data
            self.node_info['num_nodes_dict'] = self.num_nodes_dict

    def _save_to_cache(self, cache_dir):
        graphs_path, metadata_path, features_path = self._get_cache_filepaths(cache_dir)

        try:
            os.makedirs(cache_dir, exist_ok=True)
            
            # CRITICAL: Log verification of cache consistency
            logging.info("=" * 80)
            logging.info("CACHE CONSISTENCY CHECK - Before saving to cache:")
            logging.info("=" * 80)
            for ntype in self.g_full.ntypes:
                num_nodes_in_graph = self.g_full.num_nodes(ntype)
                num_nodes_in_dict = self.num_nodes_dict.get(ntype, -1)
                logging.info(f"  Node type '{ntype}':")
                logging.info(f"    - g_full.num_nodes(): {num_nodes_in_graph}")
                logging.info(f"    - num_nodes_dict: {num_nodes_in_dict}")
                
                if num_nodes_in_graph != num_nodes_in_dict:
                    logging.error(f"    ⚠️  MISMATCH DETECTED!")
                
                # Check feature embedding size if features are available
                if self.node_features and ntype in self.node_features:
                    feature_module = self.node_features[ntype]
                    if isinstance(feature_module, nn.Embedding):
                        embedding_size = feature_module.weight.shape[0]
                        if embedding_size == num_nodes_in_graph:
                            logging.info(f"    ✓ Feature embedding size matches: {embedding_size}")
                        else:
                            logging.error(f"    ✗ Feature embedding size MISMATCH: {embedding_size} != {num_nodes_in_graph}")
            logging.info("=" * 80)

            logging.info(f"Saving graphs to {graphs_path}...")
            dgl.save_graphs(graphs_path, [self.g_full, self.g_train, self.g_val, self.g_test])
            logging.info("Graphs saved successfully.")

            logging.info(f"Saving metadata to {metadata_path} directory...")
            # 创建metadata目录而不是单个文件
            metadata_dir = metadata_path.replace('.pt', '')
            os.makedirs(metadata_dir, exist_ok=True)
            
            # 分开保存每个元数据组件，以减少内存消耗
            try:
                torch.save(self.node_info, os.path.join(metadata_dir, 'node_info.pt'))
            except Exception as e:
                logging.error(f"Error saving node_info: {e}", exc_info=True)
                # Fallback or re-raise
                
            try:
                torch.save(self.train_eid_dict, os.path.join(metadata_dir, 'train_eid_dict.pt'))
            except Exception as e:
                logging.error(f"Error saving train_eid_dict: {e}", exc_info=True)

            try:
                torch.save(self.val_eid_dict, os.path.join(metadata_dir, 'val_eid_dict.pt'))
            except Exception as e:
                logging.error(f"Error saving val_eid_dict: {e}", exc_info=True)
                
            try:
                torch.save(self.test_eid_dict, os.path.join(metadata_dir, 'test_eid_dict.pt'))
            except Exception as e:
                logging.error(f"Error saving test_eid_dict: {e}", exc_info=True)
                
            try:
                torch.save(self.num_nodes_dict, os.path.join(metadata_dir, 'num_nodes_dict.pt'))
            except Exception as e:
                logging.error(f"Error saving num_nodes_dict: {e}", exc_info=True)

            try:
                torch.save(self.split_metadata, os.path.join(metadata_dir, 'split_metadata.pt'))
                torch.save(self.fold_splits, os.path.join(metadata_dir, 'fold_splits.pt'))
                torch.save(self.excluded_eid_dict, os.path.join(metadata_dir, 'excluded_eid_dict.pt'))
            except Exception as e:
                logging.error(f"Error saving split audit metadata: {e}", exc_info=True)
                
            # Save filter status
            try:
                torch.save(self._data_is_filtered, os.path.join(metadata_dir, 'is_filtered.pt'))
            except Exception as e:
                logging.error(f"Error saving filter status: {e}", exc_info=True)

            try:
                torch.save(self._get_cache_settings(), os.path.join(metadata_dir, 'cache_settings.pt'))
            except Exception as e:
                logging.error(f"Error saving cache settings: {e}", exc_info=True)
                
            logging.info("All metadata components saved successfully.")

            features_path = self._get_feature_cache_filepath(cache_dir)
            self._save_node_features(features_path)
            logging.info("Node features saved successfully to cache.")

        except Exception as e:
            logging.error(f"Error saving data to cache directory {cache_dir}: {e}", exc_info=True)
            if os.path.exists(graphs_path): os.remove(graphs_path)
            
            # 清理元数据文件或目录
            metadata_dir = metadata_path.replace('.pt', '')
            if os.path.isdir(metadata_dir):
                import shutil
                shutil.rmtree(metadata_dir)
            elif os.path.exists(metadata_path):
                os.remove(metadata_path)
                
            if self.feature_cache_path:
                logging.warning(
                    "Preserving shared node feature cache after graph cache save failure: %s",
                    features_path,
                )
            elif os.path.exists(features_path):
                os.remove(features_path)

    def _apply_source_ablation(self, edges_df):
        mode = (self.source_ablation_mode or "none").strip().lower()
        if mode in {"none", "", "off"}:
            return edges_df

        if mode not in {"exclude", "include"}:
            logging.warning(
                "Unsupported source_ablation_mode '%s'. No source-level ablation applied.",
                self.source_ablation_mode,
            )
            return edges_df

        if not self.source_ablation_sources:
            logging.warning(
                "source_ablation_mode='%s' was requested but no source_ablation_sources were provided. No source-level ablation applied.",
                mode,
            )
            return edges_df

        source_cols = [col for col in ("node_1_source", "node_2_source") if col in edges_df.columns]
        if not source_cols:
            logging.warning(
                "Source-level ablation requested but edge source columns were not found. Available columns: %s",
                edges_df.columns.tolist(),
            )
            return edges_df

        node1_match = (
            edges_df["node_1_source"].astype(str).str.strip().str.lower().isin(self.source_ablation_sources)
            if "node_1_source" in edges_df.columns
            else pd.Series(False, index=edges_df.index)
        )
        node2_match = (
            edges_df["node_2_source"].astype(str).str.strip().str.lower().isin(self.source_ablation_sources)
            if "node_2_source" in edges_df.columns
            else pd.Series(False, index=edges_df.index)
        )

        match_mode = (self.source_ablation_match or "any_endpoint").strip().lower()
        if match_mode == "node1":
            matched = node1_match
        elif match_mode == "node2":
            matched = node2_match
        elif match_mode in {"both", "both_endpoints"}:
            matched = node1_match & node2_match
        elif match_mode in {"any", "any_endpoint"}:
            matched = node1_match | node2_match
        else:
            logging.warning(
                "Unsupported source_ablation_match '%s'. Falling back to any_endpoint.",
                self.source_ablation_match,
            )
            matched = node1_match | node2_match

        before_count = len(edges_df)
        if mode == "exclude":
            filtered_df = edges_df.loc[~matched].copy()
        else:
            filtered_df = edges_df.loc[matched].copy()

        after_count = len(filtered_df)
        logging.info(
            "Source-level ablation mode=%s sources=%s match=%s retained %d/%d edge(s).",
            mode,
            sorted(self.source_ablation_sources),
            match_mode,
            after_count,
            before_count,
        )
        if after_count == 0:
            logging.warning("Source-level ablation removed all edges; downstream graph construction may fail.")
        return filtered_df

    def _node_read_usecols(self):
        required = {'id', 'type', 'name'}
        if self.generate_embeddings:
            required.update({'sequence', 'comments'})
        return required

    def _edge_read_usecols(self):
        required = {'node_1_id', 'edge_type', 'node_2_id'}
        if self.use_evidence_weights:
            required.add('edge_weight')
        if self.source_ablation_mode not in {"none", "", "off"}:
            required.update({'node_1_source', 'node_2_source'})
        return required

    @staticmethod
    def _normalized_usecols(required_columns):
        return lambda col: col.lower().replace(' ', '_') in required_columns

    def _apply_graph_filters(self, edges_df):
        before_edges = len(edges_df)
        if self.graph_relation_types and 'edge_type' in edges_df.columns:
            rel_mask = edges_df['edge_type'].astype(str).str.strip().str.lower().isin(self.graph_relation_types)
            edges_df = edges_df.loc[rel_mask].copy()
            logging.info(
                "Graph relation filter retained %d/%d edge(s) for relation types=%s.",
                len(edges_df),
                before_edges,
                sorted(self.graph_relation_types),
            )

        if self.graph_node_types:
            before_nodes = len(self.nodes_df)
            node_mask = self.nodes_df['type'].astype(str).str.strip().str.lower().isin(self.graph_node_types)
            self.nodes_df = self.nodes_df.loc[node_mask].reset_index(drop=True)
            valid_node_ids = set(self.nodes_df['id'])
            endpoint_mask = (
                edges_df['node_1_id'].isin(valid_node_ids)
                & edges_df['node_2_id'].isin(valid_node_ids)
            )
            before_endpoint_filter = len(edges_df)
            edges_df = edges_df.loc[endpoint_mask].copy()
            logging.info(
                "Graph node-type filter retained %d/%d node(s) and %d/%d edge(s) for node types=%s.",
                len(self.nodes_df),
                before_nodes,
                len(edges_df),
                before_endpoint_filter,
                sorted(self.graph_node_types),
            )

        if edges_df.empty:
            logging.warning("Graph filters removed all edges; downstream graph construction may fail.")
        return edges_df

    def _load_and_process_data(self):
        """Loads node and edge data from CSV files, specifying all columns as strings."""
        logging.info("Loading node and edge data...")
        try:
            node_usecols = self._node_read_usecols()
            edge_usecols = self._edge_read_usecols()
            logging.info("Node CSV columns requested: %s", sorted(node_usecols))
            logging.info("Edge CSV columns requested: %s", sorted(edge_usecols))

            self.nodes_df = pd.read_csv(
                self.node_csv_path,
                dtype=str,
                keep_default_na=False,
                na_values=[''],
                usecols=self._normalized_usecols(node_usecols),
            )
            edges_df = pd.read_csv(
                self.edge_csv_path,
                dtype=str,
                keep_default_na=False,
                na_values=[''],
                usecols=self._normalized_usecols(edge_usecols),
            )
            
            # Normalize column names: lowercase and replace spaces with underscores
            self.nodes_df.columns = [c.lower().replace(' ', '_') for c in self.nodes_df.columns]
            edges_df.columns = [c.lower().replace(' ', '_') for c in edges_df.columns]

            logging.info(f"Loaded {len(self.nodes_df)} nodes and {len(edges_df)} edges.")
            logging.info(f"Normalized node columns: {self.nodes_df.columns.tolist()}")
            logging.info(f"Normalized edge columns: {edges_df.columns.tolist()}")
            edges_df = self._apply_source_ablation(edges_df)
            edges_df = self._apply_graph_filters(edges_df)
            if not self.use_evidence_weights:
                logging.info("Evidence-weight ablation enabled: all graph edge weights will be set to 1.0.")

        except FileNotFoundError as e:
            logging.error(f"Error: CSV file not found. {e}")
            raise
        except Exception as e:
            logging.error(f"Error loading CSV files: {e}")
            raise

        if self.nodes_df.empty:
            raise ValueError("No nodes are available after graph filtering.")
        if edges_df.empty:
            raise ValueError("No edges are available after graph filtering.")

        # --- Process Nodes ---
        self.nodes_df = self.nodes_df.reset_index(drop=True)
        self.nodes_df['_dgl_id'] = self.nodes_df.groupby('type', sort=False).cumcount().astype(np.int64)
        self.num_nodes_dict = self.nodes_df.groupby('type', sort=False).size().astype(int).to_dict()

        self.id_to_node_data = self.nodes_df.set_index('id').to_dict('index')
        self.node_type_map = self.nodes_df.set_index('id')['type'].to_dict()

        self.node_id_map = {}
        self.node_dglid_to_orig = {}
        for ntype, group in self.nodes_df.groupby('type', sort=False):
            ids = group['id'].tolist()
            dgl_ids = group['_dgl_id'].astype(np.int64).tolist()
            self.node_dglid_to_orig[ntype] = {
                int(dgl_id): orig_id for dgl_id, orig_id in zip(dgl_ids, ids)
            }
            for dgl_id, orig_id in zip(dgl_ids, ids):
                self.node_id_map[orig_id] = {'type': ntype, 'dgl_id': int(dgl_id)}
        
        node_type_lookup = self.nodes_df.set_index('id')['type']
        node_dgl_lookup = self.nodes_df.set_index('id')['_dgl_id']
        edges_df['src_type'] = edges_df['node_1_id'].map(node_type_lookup)
        edges_df['dst_type'] = edges_df['node_2_id'].map(node_type_lookup)
        edges_df['src_dgl_id'] = edges_df['node_1_id'].map(node_dgl_lookup)
        edges_df['dst_dgl_id'] = edges_df['node_2_id'].map(node_dgl_lookup)

        valid_edges = (
            edges_df['src_type'].notna()
            & edges_df['dst_type'].notna()
            & edges_df['src_dgl_id'].notna()
            & edges_df['dst_dgl_id'].notna()
        )
        invalid_edge_count = int((~valid_edges).sum())
        if invalid_edge_count:
            logging.info("Dropping %d edge(s) whose endpoints are missing from the node table.", invalid_edge_count)
            edges_df = edges_df.loc[valid_edges].copy()

        if edges_df.empty:
            raise ValueError("No valid edges remain after endpoint mapping.")

        edges_df['src_dgl_id'] = edges_df['src_dgl_id'].astype(np.int64)
        edges_df['dst_dgl_id'] = edges_df['dst_dgl_id'].astype(np.int64)

        self.graph_data = {}
        self.edge_list_dict = {}
        self.edge_weights_full = {}
        grouped_edges = edges_df.groupby(['src_type', 'edge_type', 'dst_type'], sort=False, observed=True)
        for canonical_etype, group in grouped_edges:
            src_ids = group['src_dgl_id'].to_numpy(dtype=np.int64, copy=True)
            dst_ids = group['dst_dgl_id'].to_numpy(dtype=np.int64, copy=True)
            self.graph_data[canonical_etype] = (
                torch.from_numpy(src_ids),
                torch.from_numpy(dst_ids),
            )
            self.edge_list_dict[canonical_etype] = self.graph_data[canonical_etype]

            if self.use_evidence_weights and 'edge_weight' in group.columns:
                weights = pd.to_numeric(group['edge_weight'], errors='coerce').fillna(1.0).to_numpy(dtype=np.float32, copy=True)
            else:
                weights = np.ones(len(group), dtype=np.float32)
            self.edge_weights_full[canonical_etype] = torch.from_numpy(weights)

        if not self.graph_data:
            raise ValueError("No graph edge types were created from the input edges.")

        self.g_full = dgl.heterograph(self.graph_data, num_nodes_dict=self.num_nodes_dict)

        # Assign weights
        for etype, weights in self.edge_weights_full.items():
            if etype in self.g_full.canonical_etypes and self.g_full.num_edges(etype) > 0:
                self.g_full.edges[etype].data['weight'] = weights.float()

        # Build name->DGL ID lookup per node type, keeping lowercase aliases for robust matching
        rebuilt_name_map = {}
        for ntype, dgl_to_orig in self.node_dglid_to_orig.items():
            name_lookup = _build_name_lookup_from_dgl_mapping(dgl_to_orig, self.id_to_node_data)
            if name_lookup:
                rebuilt_name_map[ntype] = name_lookup

        self.node_name_map = rebuilt_name_map

        # Track canonical type for each alias to keep downstream consumers consistent
        type_alias_map = {}
        for canonical_type in self.g_full.ntypes:
            type_alias_map[canonical_type] = canonical_type
            lower_alias = canonical_type.lower()
            type_alias_map.setdefault(lower_alias, canonical_type)
        self.node_type_alias_map = type_alias_map
        
        logging.info("Finished loading and processing data.")

    def _log_edge_type_counts(self, graph, graph_name):
        """Log the number of edges per canonical edge type for the given graph."""
        if graph is None:
            logging.warning(f"Graph '{graph_name}' is None; cannot report edge types.")
            return

        base_etypes = self.g_full.canonical_etypes if self.g_full is not None else graph.canonical_etypes
        available_etypes = set(graph.canonical_etypes)

        logging.info(f"Edge types in {graph_name}:")
        if not base_etypes:
            logging.info("  (no edge types present)")
            return

        for etype in base_etypes:
            count = graph.num_edges(etype) if etype in available_etypes else 0
            src, rel, dst = etype
            logging.info(f"  {src} --{rel}--> {dst}: {count}")

    def _log_edge_split_summary(self):
        """Log how many edges of each type are assigned to train/val/test splits."""
        logging.info("Edge split summary (Total / Train / Val / Test / Excluded):")
        for etype in self.g_full.canonical_etypes:
            total = self.g_full.num_edges(etype)
            train_cnt = len(self.train_eid_dict.get(etype, []))
            val_cnt = len(self.val_eid_dict.get(etype, []))
            test_cnt = len(self.test_eid_dict.get(etype, []))
            excluded_cnt = len(self.excluded_eid_dict.get(etype, []))
            src, rel, dst = etype
            logging.info(
                f"  {src} --{rel}--> {dst}: Total={total}, Train={train_cnt}, "
                f"Val={val_cnt}, Test={test_cnt}, Excluded={excluded_cnt}"
            )

    @staticmethod
    def _read_id_file(path):
        if not path:
            return set()
        with open(path, 'r') as f:
            return {line.strip() for line in f if line.strip()}

    def _original_ids_to_dgl_ids_by_type(self, original_ids):
        """Map original application IDs to DGL per-type IDs."""
        normalized_ids = {str(node_id).strip() for node_id in original_ids if str(node_id).strip()}
        ids_by_type = {ntype: set() for ntype in self.g_full.ntypes}

        if not normalized_ids:
            return ids_by_type

        for ntype, dgl_to_orig in self.node_dglid_to_orig.items():
            for dgl_id, orig_id in dgl_to_orig.items():
                if str(orig_id).strip() in normalized_ids:
                    ids_by_type.setdefault(ntype, set()).add(int(dgl_id))

        found = sum(len(v) for v in ids_by_type.values())
        if found < len(normalized_ids):
            logging.warning(
                "Mapped %d/%d requested original node ID(s) to DGL IDs for split assignment.",
                found,
                len(normalized_ids),
            )
        else:
            logging.info("Mapped all %d requested original node ID(s) to DGL IDs for split assignment.", found)
        return ids_by_type

    def _parse_split_target_outputs(self):
        if not self.split_target_output:
            return []
        return [item.strip() for item in str(self.split_target_output).split(',') if item.strip()]

    def _load_external_typed_edge_pairs(self):
        path = self.split_external_edges_path
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                f"External edge split file not found: {path}"
            )
        pair_df = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            na_values=[''],
        )
        pair_df.columns = [c.strip().lower().replace(' ', '_') for c in pair_df.columns]
        required = {'disease_id', 'marker_id', 'marker_type'}
        missing = required - set(pair_df.columns)
        if missing:
            raise ValueError(
                f"External edge split file is missing columns: {sorted(missing)}"
            )

        typed_pairs = set()
        unmatched = []
        for row in pair_df.to_dict('records'):
            disease_id = str(row.get('disease_id') or '').strip()
            marker_id = str(row.get('marker_id') or '').strip()
            marker_type = str(row.get('marker_type') or '').strip().lower()
            disease_record = self.node_id_map.get(disease_id)
            marker_record = self.node_id_map.get(marker_id)
            if (
                not disease_record
                or not marker_record
                or disease_record.get('type') != 'disease'
                or marker_record.get('type') != marker_type
            ):
                unmatched.append((disease_id, marker_id, marker_type))
                continue
            disease_dgl = int(disease_record['dgl_id'])
            marker_dgl = int(marker_record['dgl_id'])
            typed_pairs.add(('disease', disease_dgl, marker_type, marker_dgl))
            typed_pairs.add((marker_type, marker_dgl, 'disease', disease_dgl))

        if not typed_pairs:
            raise ValueError(
                f"No external edge pairs from {path} matched graph nodes"
            )
        logging.info(
            "Loaded %d directed external pair(s) from %d row(s); unmatched rows=%d.",
            len(typed_pairs),
            len(pair_df),
            len(unmatched),
        )
        if unmatched:
            logging.warning("External edge rows absent from graph nodes: %s", unmatched[:20])
        return typed_pairs

    def _split_edges(self):
        """
        Splits edges into training, validation, and test sets for each edge type.
        """
        logging.info("Splitting edges into training, validation, and test sets...")
        self.train_eid_dict = {}
        self.val_eid_dict = {}
        self.test_eid_dict = {}
        self.fold_splits = None
        self.split_metadata = None
        self.excluded_eid_dict = {}

        # Report existing edge types before splitting
        self._log_edge_type_counts(self.g_full, "full graph before split")

        splitter = None
        strategy = (self.split_strategy or "random").lower()

        if strategy == "random":
            splitter = RandomEdgeSplitter(
                val_size=self.val_size,
                test_size=self.test_size,
                random_state=self.random_state,
            )
        elif strategy == "node_random":
            splitter = NodeRandomSplitter(
                val_size=self.val_size,
                test_size=self.test_size,
                random_state=self.random_state,
            )
        elif strategy == "node_kfold":
            splitter = NodeKFoldSplitter(
                n_splits=int(self.split_num_folds or 5),
                fold_index=int(self.split_fold_index or 0),
                val_fold_index=(
                    int(self.split_val_fold_index)
                    if self.split_val_fold_index is not None
                    else None
                ),
                random_state=self.random_state,
            )
        elif strategy == "specific_ids":
            if not self.split_specific_ids_path or not os.path.exists(self.split_specific_ids_path):
                logging.warning(f"Strategy 'specific_ids' chosen but file not found: {self.split_specific_ids_path}. Falling back to random.")
                splitter = RandomEdgeSplitter(
                    val_size=self.val_size,
                    test_size=self.test_size,
                    random_state=self.random_state,
                )
            else:
                with open(self.split_specific_ids_path, 'r') as f:
                    specific_ids = [line.strip() for line in f if line.strip()]

                typed_test_nodes, unmatched_ids = resolve_typed_node_ids(
                    specific_ids,
                    self.g_full,
                    self.node_id_map,
                    self.node_dglid_to_orig,
                )
                matched_count = sum(len(node_ids) for node_ids in typed_test_nodes.values())
                logging.info(
                    "Splitting from %d requested original ID(s); matched %d typed node(s).",
                    len(specific_ids),
                    matched_count,
                )
                if unmatched_ids:
                    logging.warning(
                        "%d specific ID(s) were not matched to any typed node: %s",
                        len(unmatched_ids),
                        unmatched_ids[:20],
                    )
                splitter = SpecificIDEdgeSplitter(
                    typed_test_nodes=typed_test_nodes,
                    val_size=self.val_size,
                    random_state=self.random_state,
                )

        elif strategy == "external_edges":
            splitter = ExternalEdgePairSplitter(
                typed_test_pairs=self._load_external_typed_edge_pairs(),
                val_size=self.val_size,
                random_state=self.random_state,
            )

        elif strategy in ("independent_cohort", "cohort", "cohort_validation"):
            target_outputs = self._parse_split_target_outputs()
            cohort_node_type = self.split_target_input or self.split_disease_node_type or "disease"
            explicit_test_nodes = None
            explicit_val_nodes = None

            if self.split_cohort_ids_path:
                if os.path.exists(self.split_cohort_ids_path):
                    cohort_ids = self._read_id_file(self.split_cohort_ids_path)
                    ids_by_type = self._original_ids_to_dgl_ids_by_type(cohort_ids)
                    explicit_test_nodes = ids_by_type.get(cohort_node_type, set())
                    logging.info(
                        "Independent-cohort split using %d explicit test cohort node(s) for type '%s'.",
                        len(explicit_test_nodes),
                        cohort_node_type,
                    )
                else:
                    logging.warning(
                        "Independent-cohort IDs file not found: %s. Falling back to random cohort selection.",
                        self.split_cohort_ids_path,
                    )

            if self.split_val_cohort_ids_path:
                if os.path.exists(self.split_val_cohort_ids_path):
                    cohort_ids = self._read_id_file(self.split_val_cohort_ids_path)
                    ids_by_type = self._original_ids_to_dgl_ids_by_type(cohort_ids)
                    explicit_val_nodes = ids_by_type.get(cohort_node_type, set())
                    logging.info(
                        "Independent-cohort split using %d explicit validation cohort node(s) for type '%s'.",
                        len(explicit_val_nodes),
                        cohort_node_type,
                    )
                else:
                    logging.warning(
                        "Independent-cohort validation IDs file not found: %s. Falling back to random validation cohort selection.",
                        self.split_val_cohort_ids_path,
                    )

            splitter = IndependentCohortSplitter(
                val_size=self.val_size,
                test_size=self.test_size,
                cohort_node_type=cohort_node_type,
                target_node_types=target_outputs,
                random_state=self.random_state,
                holdout_scope=self.split_holdout_scope,
                explicit_test_nodes=explicit_test_nodes,
                explicit_val_nodes=explicit_val_nodes,
            )

        elif strategy == "category":
            if not self.split_category_map:
                logging.warning("Category-aware split selected but no category map provided. Falling back to random split.")
                splitter = RandomEdgeSplitter(
                    val_size=self.val_size,
                    test_size=self.test_size,
                    random_state=self.random_state,
                )
            else:
                splitter = CategoryEdgeSplitter(
                    val_size=self.val_size,
                    test_size=self.test_size,
                    category_assignments=self.split_category_map,
                    random_state=self.random_state,
                )
        elif strategy in ("disease", "disease-centric", "disease_centered"):
            splitter = DiseaseCentricSplitter(
                val_size=self.val_size,
                test_size=self.test_size,
                disease_node_type=self.split_disease_node_type,
                random_state=self.random_state,
            )
        elif strategy in ("exclude", "type-exclusion", "exclude_types"):
            splitter = TypeExclusionSplitter(
                val_size=self.val_size,
                test_size=self.test_size,
                exclude_node_types=self.split_exclude_types,
                action=self.split_exclude_action,
                random_state=self.random_state,
            )
        elif strategy == "kfold":
            splitter = KFoldEdgeSplitter(
                n_splits=max(2, int(self.split_num_folds or 5)),
                fold_index=int(self.split_fold_index or 0),
                val_fold_index=(
                    int(self.split_val_fold_index)
                    if self.split_val_fold_index is not None
                    else None
                ),
                val_fraction=max(0.0, float(self.val_size)),
                random_state=self.random_state,
            )
        elif strategy == "target_exclude":
            if not self.split_target_input or not self.split_target_output:
                logging.warning("Strategy 'target_exclude' selected but input/output types missing. Falling back to random.")
                splitter = RandomEdgeSplitter(
                    val_size=self.val_size,
                    test_size=self.test_size,
                    random_state=self.random_state,
                )
            else:
                logging.info(f"Splitting strategy: target_exclude. Moving edges between {self.split_target_input} and [{self.split_target_output}] to Test.")
                target_in = self.split_target_input
                target_outs = set(self.split_target_output.split(','))
                
                # Identify Target Edges and Non-Target Edges
                target_etypes = []
                non_target_etypes = []
                
                n_target_edges = 0
                n_non_target_edges = 0
                
                for etype in self.g_full.canonical_etypes:
                    src, rel, dst = etype
                    # Check both directions if undirected conceptualization, but strictly matching types
                    is_target = False
                    if (src == target_in and dst in target_outs) or (dst == target_in and src in target_outs):
                        is_target = True
                    
                    count = self.g_full.num_edges(etype)
                    if is_target:
                        target_etypes.append(etype)
                        n_target_edges += count
                    else:
                        non_target_etypes.append(etype)
                        n_non_target_edges += count
                
                n_total = n_target_edges + n_non_target_edges
                target_test_size = int(n_total * self.test_size)
                
                # Calculate required test edges from non-target types
                needed_from_non_target = max(0, target_test_size - n_target_edges)
                
                # Fraction for non-target types
                if n_non_target_edges > 0:
                    non_target_test_fraction = needed_from_non_target / n_non_target_edges
                else:
                    non_target_test_fraction = 0.0
                    
                logging.info(f"  Target Edges: {n_target_edges} (All -> Test)")
                logging.info(f"  Non-Target Edges: {n_non_target_edges}. Allocated to Test: {needed_from_non_target} (Ratio: {non_target_test_fraction:.4f})")
                
                rng = np.random.default_rng(self.random_state)
                
                for etype in self.g_full.canonical_etypes:
                    n_edges = self.g_full.num_edges(etype)
                    indices = np.arange(n_edges)
                    
                    if etype in target_etypes:
                        # ALL to test
                        self.test_eid_dict[etype] = indices
                        self.val_eid_dict[etype] = np.array([], dtype=np.int64)
                        self.train_eid_dict[etype] = np.array([], dtype=np.int64)
                    else:
                        # Helper for random split
                        n_test = int(n_edges * non_target_test_fraction)
                        n_val = int(n_edges * self.val_size) 
                        
                        # Ensure we don't exceed n_edges
                        if n_test + n_val > n_edges:
                             n_val = max(0, n_edges - n_test)
                        
                        perm = rng.permutation(n_edges)
                        test_idx = perm[:n_test]
                        val_idx = perm[n_test:n_test+n_val]
                        train_idx = perm[n_test+n_val:]
                        
                        self.test_eid_dict[etype] = indices[test_idx]
                        self.val_eid_dict[etype] = indices[val_idx]
                        self.train_eid_dict[etype] = indices[train_idx]
                
                splitter = None
        else:
            logging.warning(f"Unknown split strategy '{self.split_strategy}'. Falling back to random split.")
            splitter = RandomEdgeSplitter(
                val_size=self.val_size,
                test_size=self.test_size,
                random_state=self.random_state,
            )

        if splitter is not None:
            split_result: SplitResult = splitter.split(self.g_full)

            empty = np.empty(0, dtype=np.int64)
            for etype in self.g_full.canonical_etypes:
                self.train_eid_dict[etype] = split_result.train_eids.get(etype, empty)
                self.val_eid_dict[etype] = split_result.val_eids.get(etype, empty)
                self.test_eid_dict[etype] = split_result.test_eids.get(etype, empty)

            if split_result.folds:
                self.fold_splits = split_result.folds
                logging.info(f"Generated {len(self.fold_splits)} fold(s) for cross-validation.")
            self.split_metadata = split_result.metadata
            if split_result.metadata:
                self.excluded_eid_dict = split_result.metadata.get(
                    "excluded_cross_edges",
                    {},
                )
                excluded_total = int(
                    split_result.metadata.get("excluded_cross_edge_total", 0)
                )
                if excluded_total:
                    logging.info(
                        "Excluded %d cross-split edge(s) from all datasets.",
                        excluded_total,
                    )

        logging.info("Edge splitting complete.")
        self._log_edge_split_summary()

    def _build_graphs(self):
        """
        Builds the training, validation, and test graphs based on the edge splits.
        """
        logging.info("Building DGL graphs for train, validation, and test sets...")
        
        # Create training graph from the full graph by removing held-out and
        # explicitly excluded cross-split edges.
        if self.val_size > 0 or self.test_size > 0 or self.excluded_eid_dict:
            edges_to_remove = {}
            for etype in self.g_full.canonical_etypes:
                val_eids = self.val_eid_dict.get(etype, np.array([]))
                test_eids = self.test_eid_dict.get(etype, np.array([]))
                excluded_eids = self.excluded_eid_dict.get(etype, np.array([]))
                eids_to_remove_for_etype = np.concatenate(
                    [val_eids, test_eids, excluded_eids]
                ).astype(np.int64, copy=False)
                if len(eids_to_remove_for_etype) > 0:
                    edges_to_remove[etype] = torch.from_numpy(eids_to_remove_for_etype)
            
            # Iteratively remove edges for each type, as required by DGL for heterographs
            g_train_temp = self.g_full
            for etype, eids in edges_to_remove.items():
                before_count = g_train_temp.num_edges(etype)
                g_train_temp = dgl.remove_edges(g_train_temp, eids, etype=etype)
                after_count = g_train_temp.num_edges(etype)
                removed_count = before_count - after_count
                if removed_count > 0:
                    logging.info(
                        f"Removed {removed_count} edge(s) of type {etype} when "
                        "excluding held-out/cross-split edges."
                    )
                if after_count == 0 and before_count > 0:
                    logging.warning(f"Edge type {etype} now has 0 edge(s) in the training graph after val/test split.")
            self.g_train = g_train_temp
        else:
            # If no validation or test set, the training graph is the full graph
            self.g_train = self.g_full

        self.g_val = dgl.edge_subgraph(self.g_full, self.val_eid_dict, relabel_nodes=False) if self.val_size > 0 else dgl.heterograph({k: ([], []) for k in self.g_full.canonical_etypes}, num_nodes_dict=self.num_nodes_dict)
        self.g_test = dgl.edge_subgraph(self.g_full, self.test_eid_dict, relabel_nodes=False) if self.test_size > 0 else dgl.heterograph({k: ([], []) for k in self.g_full.canonical_etypes}, num_nodes_dict=self.num_nodes_dict)

        # Remove self-loops from the training graph, only for homogeneous edge types
        g_train_temp = self.g_train
        for etype in g_train_temp.canonical_etypes:
            src_type, _, dst_type = etype
            if src_type == dst_type:
                before_edges = g_train_temp.num_edges(etype)
                g_train_temp = dgl.remove_self_loop(g_train_temp, etype=etype)
                after_edges = g_train_temp.num_edges(etype)
                removed_edges = before_edges - after_edges
                if removed_edges > 0:
                    logging.info(f"Removed {removed_edges} self-loop edge(s) for edge type {etype}.")
                if after_edges == 0 and before_edges > 0:
                    logging.warning(f"Edge type {etype} lost all edges after self-loop removal.")
            else:
                logging.info(f"Skipping self-loop removal for bipartite edge type: {etype}")
        self.g_train = g_train_temp

        # Save edge weights before filtering (since filtering will relabel edges)
        edge_weights_before_filter = {}
        for etype in self.g_full.canonical_etypes:
            if 'weight' in self.g_full.edges[etype].data:
                edge_weights_before_filter[etype] = self.g_full.edges[etype].data['weight'].clone()
        
        # Save edge IDs for val and test graphs before filtering (fallback if dgl.EID missing after filtering)
        val_eids_before_filter = {}
        test_eids_before_filter = {}
        for etype in self.g_full.canonical_etypes:
            if self.g_val.num_edges(etype) > 0:
                val_eids_before_filter[etype] = self.g_val.edges[etype].data[
                    dgl.EID
                ].clone()
            if self.g_test.num_edges(etype) > 0:
                test_eids_before_filter[etype] = self.g_test.edges[etype].data[
                    dgl.EID
                ].clone()
        
        # Filter out isolated nodes from all graphs unless disabled
        if self.filter_isolated_nodes:
            self._filter_isolated_nodes()

            # After filtering/relabeling, rebuild mappings so name->dglid and orig->dglid stay consistent
            self._rebuild_node_mappings_after_filter()
        else:
            logging.info("Skipping isolated-node filtering per configuration.")

        # Copy edge weights to subgraphs
        # After filtering, g_train's EID data still refers to the original g_full before filtering
        # So we use the saved weights from before filtering
        for etype in self.g_full.canonical_etypes:
            # Assign weights to train graph
            if self.g_train.num_edges(etype) > 0 and etype in edge_weights_before_filter:
                # Check if etype exists in edata[dgl.EID] (may not exist if all edges were removed)
                if dgl.EID in self.g_train.edges[etype].data:
                    train_eids_in_full = self.g_train.edata[dgl.EID][etype]
                    # Check if all edge IDs are within bounds
                    if train_eids_in_full.max() < len(edge_weights_before_filter[etype]):
                        self.g_train.edges[etype].data['weight'] = edge_weights_before_filter[etype][train_eids_in_full]
                    else:
                        logging.warning(f"Skipping weight assignment for train graph edge type {etype} due to index mismatch after filtering.")
                else:
                    logging.debug(f"Edge type {etype} not found in g_train edge data's EID; skipping weight assignment.")
            
            # Assign weights to val graph
            if self.g_val.num_edges(etype) > 0 and etype in edge_weights_before_filter:
                if dgl.EID in self.g_val.edges[etype].data:
                    val_eids = self.g_val.edges[etype].data[dgl.EID]
                else:
                    val_eids = val_eids_before_filter.get(etype)

                if val_eids is None:
                    logging.warning(f"Skipping weight assignment for val graph edge type {etype} because edge IDs are unavailable.")
                else:
                    # Ensure the feature length matches the number of edges after filtering
                    if len(val_eids) == self.g_val.num_edges(etype) and val_eids.max() < len(edge_weights_before_filter[etype]):
                        self.g_val.edges[etype].data['weight'] = edge_weights_before_filter[etype][val_eids]
                    else:
                        logging.warning(f"Skipping weight assignment for val graph edge type {etype} due to size/index mismatch after filtering.")

            # Assign weights to test graph
            if self.g_test.num_edges(etype) > 0 and etype in edge_weights_before_filter:
                if dgl.EID in self.g_test.edges[etype].data:
                    test_eids = self.g_test.edges[etype].data[dgl.EID]
                else:
                    test_eids = test_eids_before_filter.get(etype)

                if test_eids is None:
                    logging.warning(f"Skipping weight assignment for test graph edge type {etype} because edge IDs are unavailable.")
                else:
                    # Ensure the feature length matches the number of edges after filtering
                    if len(test_eids) == self.g_test.num_edges(etype) and test_eids.max() < len(edge_weights_before_filter[etype]):
                        self.g_test.edges[etype].data['weight'] = edge_weights_before_filter[etype][test_eids]
                    else:
                        logging.warning(f"Skipping weight assignment for test graph edge type {etype} due to size/index mismatch after filtering.")

        self._log_edge_type_counts(self.g_train, "training graph after split")
        self._log_edge_type_counts(self.g_val, "validation graph after split")
        self._log_edge_type_counts(self.g_test, "test graph after split")

        logging.info("Finished building graphs.")
        self._data_is_filtered = bool(self.filter_isolated_nodes) # Set flag based on actual filtering
        log_memory_usage("End of _build_graphs", self.device)

    def _filter_isolated_nodes(self):
        """
        Removes nodes that have no edges in the training graph from all graphs
        (g_full, g_train, g_val, g_test) to prevent errors in models like GAT.
        """
        logging.info("Filtering isolated nodes based on training graph...")

        pre_node_counts = {
            'full': {ntype: self.g_full.num_nodes(ntype) for ntype in self.g_full.ntypes},
            'train': {ntype: self.g_train.num_nodes(ntype) for ntype in self.g_train.ntypes},
            'val': {ntype: self.g_val.num_nodes(ntype) for ntype in self.g_val.ntypes},
            'test': {ntype: self.g_test.num_nodes(ntype) for ntype in self.g_test.ntypes}
        }
        pre_edge_counts = {
            'full': {etype: self.g_full.num_edges(etype) for etype in self.g_full.canonical_etypes},
            'train': {etype: self.g_train.num_edges(etype) for etype in self.g_train.canonical_etypes},
            'val': {etype: self.g_val.num_edges(etype) for etype in self.g_val.canonical_etypes},
            'test': {etype: self.g_test.num_edges(etype) for etype in self.g_test.canonical_etypes}
        }
        
        nodes_to_keep = {}
        for ntype in self.g_train.ntypes:
            nodes_of_type = self.g_train.nodes(ntype)
            total_in_degrees = torch.zeros(self.g_train.num_nodes(ntype), dtype=torch.int32)
            total_out_degrees = torch.zeros(self.g_train.num_nodes(ntype), dtype=torch.int32)

            for srctype, etype, dsttype in self.g_train.canonical_etypes:
                # Check if the current node type is the destination for this edge type
                if dsttype == ntype:
                    total_in_degrees += self.g_train.in_degrees(nodes_of_type, etype=(srctype, etype, dsttype))
                # Check if the current node type is the source for this edge type
                if srctype == ntype:
                    total_out_degrees += self.g_train.out_degrees(nodes_of_type, etype=(srctype, etype, dsttype))
            
            # Keep nodes that have at least one connection in the training graph
            connected_nodes = torch.where((total_in_degrees + total_out_degrees) > 0)[0]
            nodes_to_keep[ntype] = self.g_train.nodes(ntype)[connected_nodes]

        # If no nodes are being removed, we can skip the expensive subgraph operations
        if all(len(nodes_to_keep[ntype]) == self.g_full.num_nodes(ntype) for ntype in self.g_full.ntypes):
            logging.info("No isolated nodes found in the training graph. No filtering needed.")
            return

        # Use dgl.node_subgraph to create new graphs with only the nodes to keep.
        # This also relabels the nodes and edges correctly.
        logging.info("Applying filter to g_full, g_train, g_val, and g_test...")
        self.g_full = dgl.node_subgraph(self.g_full, nodes_to_keep)
        self.g_train = dgl.node_subgraph(self.g_train, nodes_to_keep)
        self.g_val = dgl.node_subgraph(self.g_val, nodes_to_keep)
        self.g_test = dgl.node_subgraph(self.g_test, nodes_to_keep)

        # After subgraphing, DGL relabels node IDs. dgl.NID stores original IDs; we rebuild maps later.

        post_node_counts = {
            'full': {ntype: self.g_full.num_nodes(ntype) for ntype in self.g_full.ntypes},
            'train': {ntype: self.g_train.num_nodes(ntype) for ntype in self.g_train.ntypes},
            'val': {ntype: self.g_val.num_nodes(ntype) for ntype in self.g_val.ntypes},
            'test': {ntype: self.g_test.num_nodes(ntype) for ntype in self.g_test.ntypes}
        }
        post_edge_counts = {
            'full': {etype: self.g_full.num_edges(etype) for etype in self.g_full.canonical_etypes},
            'train': {etype: self.g_train.num_edges(etype) for etype in self.g_train.canonical_etypes},
            'val': {etype: self.g_val.num_edges(etype) for etype in self.g_val.canonical_etypes},
            'test': {etype: self.g_test.num_edges(etype) for etype in self.g_test.canonical_etypes}
        }

        for graph_name in post_node_counts:
            for ntype, before in pre_node_counts.get(graph_name, {}).items():
                after = post_node_counts[graph_name].get(ntype, 0)
                if after < before:
                    logging.info(f"Node type '{ntype}' reduced from {before} to {after} node(s) in {graph_name} due to isolated-node filtering.")
                if before > 0 and after == 0:
                    logging.warning(f"Node type '{ntype}' was completely removed from {graph_name} during isolated-node filtering.")

        for graph_name in post_edge_counts:
            for etype, before in pre_edge_counts.get(graph_name, {}).items():
                after = post_edge_counts[graph_name].get(etype, 0)
                if after < before:
                    logging.info(f"Edge type {etype} reduced from {before} to {after} edge(s) in {graph_name} during isolated-node filtering.")
                if before > 0 and after == 0:
                    logging.warning(f"Edge type {etype} was completely removed from {graph_name} during isolated-node filtering.")

        # After filtering, the node counts and mappings are invalid. We must update them.
        logging.info("Updating node metadata after filtering...")
        
        # 1. Update num_nodes_dict
        self.num_nodes_dict = {ntype: self.g_full.num_nodes(ntype) for ntype in self.g_full.ntypes}

        # 2. Update node_dglid_to_orig mapping
        # The '.dgl.NID' feature stores the original node IDs from before the subgraph operation.
        new_dglid_to_orig = {}
        for ntype in self.g_full.ntypes:
            new_dglid_to_orig[ntype] = {}
            original_nids = self.g_full.nodes[ntype].data[dgl.NID] # These are old DGL IDs
            for new_dgl_id, old_dgl_id_tensor in enumerate(original_nids):
                old_dgl_id = old_dgl_id_tensor.item()
                # We need to map the new_dgl_id to the original application-wide ID,
                # which was stored in the old node_dglid_to_orig mapping.
                if ntype in self.node_dglid_to_orig and old_dgl_id in self.node_dglid_to_orig[ntype]:
                    original_app_id = self.node_dglid_to_orig[ntype][old_dgl_id]
                    new_dglid_to_orig[ntype][new_dgl_id] = original_app_id
                else:
                    logging.warning(f"Could not find original app ID for old DGL ID {old_dgl_id} of ntype {ntype} during filtering update.")
        
        self.node_dglid_to_orig = new_dglid_to_orig
        logging.info("Node metadata (num_nodes_dict, node_dglid_to_orig) updated after filtering.")

    def _prepare_node_info(self):
        """
        Prepares a dictionary with essential node information for the model.
        Consolidates all node-related mappings into a single dictionary for easy access.
        """
        logging.info("Preparing consolidated node info dictionary...")
        self.node_info = {
            'id_map': self.node_id_map,
            'name_map': self.node_name_map,
            'type_map': getattr(self, 'node_type_map', None) or self.nodes_df.set_index('id')['type'].to_dict(),
            'type_alias_map': self.node_type_alias_map,
            'dglid_to_orig': self.node_dglid_to_orig,
            'id_to_node_data': self.id_to_node_data,
            'num_nodes_dict': self.num_nodes_dict
        }
        logging.info("Node info dictionary prepared.")

    def _rebuild_node_mappings_after_filter(self):
        """
        After node_subgraph relabels nodes, rebuild name_map, id_map, and dglid_to_orig
        so that downstream lookup (predict) can resolve names to valid DGL IDs.
        """
        logging.info("Rebuilding node mappings after filtering/relabeling...")
        # Preserve the pre-filter mapping (old DGL ID -> original node ID) so we can
        # translate node_subgraph's dgl.NID values back to original node IDs.
        previous_dglid_to_orig = self.node_dglid_to_orig

        self.node_id_map = {}
        self.node_name_map = {}
        self.node_dglid_to_orig = {}

        for ntype in self.g_full.ntypes:
            self.node_id_map[ntype] = {}
            self.node_name_map[ntype] = {}
            self.node_dglid_to_orig[ntype] = {}

            if dgl.NID not in self.g_full.nodes[ntype].data:
                logging.warning(f"dgl.NID missing for ntype {ntype} after filtering; skipping mapping rebuild.")
                continue

            orig_ids_tensor = self.g_full.nodes[ntype].data[dgl.NID]
            orig_ids = orig_ids_tensor.detach().cpu().tolist()

            for new_id, old_dgl_id in enumerate(orig_ids):
                original_app_id = previous_dglid_to_orig.get(ntype, {}).get(int(old_dgl_id))
                if original_app_id is None:
                    logging.warning(
                        "Missing previous dgl->orig mapping for ntype=%s old_dgl_id=%s; skipping node in rebuild.",
                        ntype,
                        old_dgl_id,
                    )
                    continue

                self.node_dglid_to_orig[ntype][new_id] = original_app_id
                self.node_id_map[ntype][original_app_id] = new_id

                node_data = _get_node_record_by_original_id(self.id_to_node_data, original_app_id)
                name_val = _extract_node_display_name(node_data)
                if name_val is not None:
                    try:
                        self.node_name_map[ntype][str(name_val)] = new_id
                    except Exception:
                        pass

        # Keep type aliases; num_nodes_dict should reflect current g_full
        self.num_nodes_dict = {ntype: self.g_full.num_nodes(ntype) for ntype in self.g_full.ntypes}
        logging.info("Rebuilt mappings: node_id_map, node_name_map, node_dglid_to_orig, num_nodes_dict updated.")

    def get_graphs(self):
        """Returns the full, training, validation, and test graphs."""
        return self.g_full, self.g_train, self.g_val, self.g_test

    def get_fold_splits(self):
        """Return edge ID assignments for each fold if k-fold splitting was performed."""
        return self.fold_splits

    def get_split_metadata(self):
        """Return node partitions and cross-split edge audit data, when available."""
        return self.split_metadata

    def get_node_features(self):
        """
        Returns the node features. If the graph has been filtered, this method
        returns a new nn.ModuleDict with embedding layers correctly sized and
        weighted for the compacted graph.
        """
        if not self._data_is_filtered:
            logging.debug("Data is not filtered. Returning original node features.")
            return self.node_features

        logging.info("Data is filtered. Checking if node features have already been updated...")
        
        # Check if node features have already been resized to match the filtered graph
        # This happens when _create_or_load_node_features(force_recreate=True) is called after _build_graphs()
        features_already_filtered = True
        for ntype, feature_layer in self.node_features.items():
            if isinstance(feature_layer, nn.Embedding):
                graph_num_nodes = self.g_full.num_nodes(ntype)
                feature_num_nodes = feature_layer.weight.shape[0]
                if feature_num_nodes != graph_num_nodes:
                    features_already_filtered = False
                    logging.info(f"  {ntype}: Feature size {feature_num_nodes} != Graph size {graph_num_nodes} - needs filtering")
                    break
                else:
                    logging.debug(f"  {ntype}: Feature size {feature_num_nodes} == Graph size {graph_num_nodes} ✓")
        
        if features_already_filtered:
            logging.info("Node features are already sized to match the filtered graph. Returning as-is.")
            return self.node_features
        
        logging.info("Node features need to be filtered to match the compacted graph.")
        
        # This is a state where the graph has been filtered, but the features have not been updated yet.
        # We need to create a new set of feature embeddings that correspond to the nodes remaining in the graph.
        
        new_features = nn.ModuleDict()
        
        for ntype, original_feature_layer in self.node_features.items():
            # Check if the item is an nn.Embedding layer, which is what we need to resize.
            if not isinstance(original_feature_layer, nn.Embedding):
                # If it's not an embedding (e.g., a projection layer), just copy it.
                new_features[ntype] = original_feature_layer
                continue

            # These are the original DGL IDs of the nodes that remain in the filtered graph for this ntype.
            # dgl.NID is added by dgl.node_subgraph and holds the original node IDs.
            if dgl.NID not in self.g_full.nodes[ntype].data:
                logging.error(f"'{dgl.NID}' not found in filtered graph for ntype '{ntype}'. Cannot create filtered features.")
                # Fallback to original features to avoid crashing, though this may lead to other errors.
                return self.node_features
                
            remaining_node_indices = self.g_full.nodes[ntype].data[dgl.NID]
            
            num_remaining_nodes = self.g_full.num_nodes(ntype)
            embedding_dim = original_feature_layer.embedding_dim

            # Create a new embedding layer for the filtered nodes.
            new_embedding_layer = nn.Embedding(num_remaining_nodes, embedding_dim)
            
            # Copy the weights for the remaining nodes from the old embedding layer to the new one.
            # The `remaining_node_indices` tensor contains the original indices of the nodes that were kept.
            new_weights = original_feature_layer.weight.data[remaining_node_indices]
            new_embedding_layer.weight.data.copy_(new_weights)
            
            # Preserve the requires_grad property
            new_embedding_layer.weight.requires_grad = original_feature_layer.weight.requires_grad
            
            new_features[ntype] = new_embedding_layer

        return new_features

    def get_node_info(self):
        """Returns the consolidated node info dictionary."""
        return self.node_info
