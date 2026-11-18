#!/usr/bin/env python3
"""Precompute BioGT graph and node-feature caches without starting training."""

import argparse
import logging
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Model.modelData import DGLGraphCreator


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edges-csv", required=True)
    parser.add_argument("--nodes-csv", required=True)
    parser.add_argument("--graph-cache-path", required=True)
    parser.add_argument("--feature-cache-path", required=True)
    parser.add_argument("--local-model-path", default="./models_cache")
    parser.add_argument("--default-embed-size", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.0625)
    parser.add_argument("--test-split", type=float, default=0.0625)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--disable-sequence-features", action="store_true")
    parser.add_argument("--merged-sequence-features", action="store_true")
    parser.add_argument("--split-strategy", default="random")
    parser.add_argument("--split-target-input", default="disease")
    parser.add_argument("--split-target-output", default="protein,gene,rna,metabolite")
    parser.add_argument("--filter-isolated-nodes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--regenerate-features-after-filter", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        logging.info("Using CUDA for text feature generation: %s", torch.cuda.get_device_name(args.gpu))
    else:
        device = torch.device("cpu")
        logging.info("Using CPU for feature generation")

    os.makedirs(args.graph_cache_path, exist_ok=True)
    os.makedirs(args.feature_cache_path, exist_ok=True)

    graph_creator = DGLGraphCreator(
        node_csv_path=args.nodes_csv,
        edge_csv_path=args.edges_csv,
        embed_size=args.default_embed_size,
        val_size=args.val_split,
        test_size=args.test_split,
        random_state=args.seed,
        graph_cache_path=args.graph_cache_path,
        feature_cache_path=args.feature_cache_path,
        device=device,
        local_model_base_path=args.local_model_path,
        generate_embeddings=True,
        use_sequence_features=not args.disable_sequence_features,
        merged_sequence_features=args.merged_sequence_features,
        split_strategy=args.split_strategy,
        split_target_input=args.split_target_input,
        split_target_output=args.split_target_output,
        filter_isolated_nodes=args.filter_isolated_nodes,
        regenerate_features_after_filter=args.regenerate_features_after_filter,
    )
    graphs = graph_creator.get_graphs()
    features = graph_creator.get_node_features()
    logging.info("Prepared %d graphs and %d feature modules", len(graphs), len(features))
    logging.info("Graph cache: %s", args.graph_cache_path)
    logging.info("Feature cache: %s", args.feature_cache_path)


if __name__ == "__main__":
    main()
