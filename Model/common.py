import torch
import os
import logging
import psutil  # For system memory usage

# ========== DEBUG LOGGING CONTROL ==========
# Set to True to enable detailed debug logging for model internals
# Set to False to disable debug logging
DEBUG_LOGGING = False  # Global flag to control debug output

# --- Memory Logging Function ---
def log_memory_usage(stage="", device=None):
    """Logs current RAM and GPU memory usage."""
    process = psutil.Process(os.getpid())
    ram_used_gb = process.memory_info().rss / (1024 * 1024 * 1024)  # Resident Set Size in GB
    ram_percent = process.memory_percent()
    log_message = f"Memory Usage ({stage}): CPU RAM Used: {ram_used_gb:.2f} GB ({ram_percent:.1f}%)"

    if torch.cuda.is_available() and (device is None or device.type == 'cuda'):
        # If device is not specified, it will use the current device.
        # Use torch.cuda.mem_get_info() for a more accurate total usage, similar to nvidia-smi.
        # It returns (free memory, total memory) in bytes.
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        used_bytes = total_bytes - free_bytes
        used_gb = used_bytes / (1024**3)
        total_gb = total_bytes / (1024**3)
        
        # Also log PyTorch's own view for debugging.
        gpu_allocated_gb = torch.cuda.memory_allocated(device) / (1024**3)
        gpu_reserved_gb = torch.cuda.memory_reserved(device) / (1024**3)
        
        device_str = ""
        if device is not None:
            try:
                device_str = f" on {torch.cuda.get_device_name(device)} ({device})"
            except:
                device_str = f" on device {device}"


        log_message += f" | GPU{device_str} Total Used: {used_gb:.2f} GB / {total_gb:.2f} GB"
        log_message += f" (PyTorch Allocated: {gpu_allocated_gb:.2f} GB, Reserved: {gpu_reserved_gb:.2f} GB)"
    else:
        log_message += " | GPU: Not available"
    logging.info(log_message)


default_batch_size_for_embedding = 4096

def setup_logging(seed):
    """Initializes logging and sets a seed for reproducibility."""
    log_filename = f'training_log_seed_{seed}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_filename}")


def log_explainability_summary(explainability_payload, prefix=""):
    """Pretty-print explainability payloads produced by the model."""
    if not explainability_payload:
        return

    header = f"{prefix} Explainability" if prefix else "Explainability"
    fusion_mode = explainability_payload.get('fusion_mode', 'unknown')
    logging.info(f"{header}: fusion_mode={fusion_mode}")

    fusion_weights = explainability_payload.get('fusion_weights', {}) or {}
    if fusion_weights:
        weight_str = ", ".join(f"{ntype}={weight:.3f}" for ntype, weight in fusion_weights.items())
        logging.info(f"  Fusion weights: {weight_str}")

    per_type = explainability_payload.get('per_type', {}) or {}

    def _format_entries(entries, value_key, label):
        if not entries:
            return
        formatted = ", ".join(
            f"idx {item.get('node_idx', -1)}={item.get(value_key, 0.0):.3f}"
            for item in entries
        )
        logging.info(f"    {label}: {formatted}")

    for ntype, entry in per_type.items():
        logging.info(f"  Node type: {ntype}")
        _format_entries(entry.get('hgcn_topk_norm'), 'norm', 'HGCN top-n norm')
        _format_entries(entry.get('transformer_topk_norm'), 'norm', 'Transformer top-n norm')
        _format_entries(entry.get('fused_topk_norm'), 'norm', 'Fused top-n norm')
        _format_entries(entry.get('alignment_topk'), 'value', 'HGNN↔Transformer cosine')

        alignment_summary = entry.get('alignment_summary') or {}
        if alignment_summary:
            summary_str = ", ".join(f"{k}={v:.3f}" for k, v in alignment_summary.items())
            logging.info(f"    Alignment summary: {summary_str}")

    edge_alignment = explainability_payload.get('edge_alignment', {}) or {}
    for etype, edges in edge_alignment.items():
        if not edges:
            continue

        def _format_edge(edge):
            base = f"({edge.get('src_idx', -1)}->{edge.get('dst_idx', -1)}:{edge.get('similarity', 0.0):.3f})"
            path = edge.get('path')
            if path:
                return f"{base} [{path}]"
            return base

        formatted = ", ".join(_format_edge(edge) for edge in edges)
        logging.info(f"  Edge {etype}: {formatted}")