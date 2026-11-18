# download_models.py
from sentence_transformers import SentenceTransformer
from transformers import EsmModel, EsmTokenizer
import os
import json
from huggingface_hub import snapshot_download
import time
import argparse
import traceback

# A dictionary of all models that might be used in the project.
# This allows for pre-downloading all necessary models.
ALL_MODELS = {
    'protein': 'facebook/esm2_t6_8M_UR50D',
    'nucleotide': 'InstaDeepAI/nucleotide-transformer-v2-50m-multi-species', # For gene, dna, rna
    'smiles': 'seyonec/PubChem10M_SMILES_BPE_450k', # For compound, metabolite, drug
    'sentence': 'sentence-transformers/all-MiniLM-L6-v2', # For comments
    'default_text': 'albert-base-v2' # For other text-based node types
}

def check_and_fix_tokenizer_config(model_path):
    """
    检查并修复指定模型路径下的tokenizer_config.json文件。
    如果model_max_length大于vocab_size，则将其设置为vocab_size。
    """
    tokenizer_config_path = os.path.join(model_path, 'tokenizer_config.json')
    # ESM模型的词汇表文件是 vocab.txt
    vocab_path = os.path.join(model_path, 'vocab.txt')

    if not os.path.exists(tokenizer_config_path) or not os.path.exists(vocab_path):
        print(f"  - 在 {model_path} 中未找到 tokenizer_config.json 或 vocab.txt，跳过检查。")
        return

    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_size = len(f.readlines())

        with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 检查 model_max_length 是否存在且大于 vocab_size
        if 'model_max_length' in config and config['model_max_length'] > vocab_size:
            original_value = config['model_max_length']
            # 将 model_max_length 调整为词汇表大小
            config['model_max_length'] = vocab_size
            print(f"  - 在 {model_path} 中检测到不匹配。")
            print(f"    'model_max_length' 从 {original_value} 调整为 {vocab_size}。")
            
            with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
    except Exception as e:
        print(f"  - 检查或修复配置文件时出错: {e}")

def download_model_with_retry(model_id, max_retries=3, delay=10):
    """
    带重试逻辑的下载函数。
    """
    for attempt in range(max_retries):
        try:
            print(f"  -> 正在处理: {model_id} (尝试 {attempt + 1}/{max_retries})")
            
            if 'sentence-transformers' in model_id:
                # 对于Sentence Transformers，使用其自己的类来加载，这更可靠
                print("  - 使用 SentenceTransformer 加载器。")
                SentenceTransformer(model_id)
                # sentence-transformer不需要进行config检查
                print(f"  ✓ {model_id} 下载并缓存成功。")
                return True
            else:
                # 对于其他模型，使用 from_pretrained
                print("  - 使用 Hugging Face Transformers 加载器。")
                EsmTokenizer.from_pretrained(model_id)
                EsmModel.from_pretrained(model_id)
                
                # 下载完成后，获取本地路径以进行验证
                print("  - 获取本地模型路径...")
                local_model_path = snapshot_download(repo_id=model_id)
                
                # 对下载的模型进行配置检查和修复
                print("  - 检查并修复tokenizer配置...")
                check_and_fix_tokenizer_config(local_model_path)
                
                print(f"  ✓ {model_id} 下载并验证成功。")
                return True

        except Exception as e:
            print(f"  ✗ 下载 {model_id} 时出错: {e}")
            if attempt + 1 < max_retries:
                print(f"  - 等待 {delay} 秒后重试...")
                time.sleep(delay)
            else:
                print(f"  ✗ 已达到最大重试次数，{model_id} 下载失败。")
                return False

def download_model(model_name, cache_dir):
    """
    Downloads a single model from Hugging Face Hub to a specified cache directory.
    Handles SentenceTransformer models differently by saving them to a specific path.
    """
    try:
        print(f"Downloading model: {model_name} to cache: {cache_dir}")

        if 'sentence-transformers' in model_name:
            # For SentenceTransformer models, we load and then save them to a path
            # that matches the logic in modelData.py (replacing '/' with '_').
            print(f"  - Detected SentenceTransformer model. Loading '{model_name}'...")
            model = SentenceTransformer(model_name)
            
            # Create a directory name that matches the loading logic in modelData.py
            local_model_dir_name = model_name.replace("/", "_")
            save_path = os.path.join(cache_dir, local_model_dir_name)
            
            print(f"  - Saving model to: {save_path}")
            model.save(save_path)
            print(f"Successfully downloaded and saved {model_name} to {save_path}")

        else:
            # For all other models, use snapshot_download.
            # The files will be stored in a nested directory structure inside cache_dir
            # e.g., cache_dir/models--facebook--esm2_t6_8M_UR50D
            print(f"  - Using snapshot_download for '{model_name}'.")
            snapshot_download(repo_id=model_name, cache_dir=cache_dir, resume_download=True)
            print(f"Successfully downloaded {model_name} via snapshot_download.")

    except Exception as e:
        print(f"Failed to download model {model_name}. Error: {e}")
        traceback.print_exc()

def download_all_models(base_path):
    """
    Downloads all models defined in ALL_MODELS to a structured directory.
    """
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
        print(f"Created base directory: {base_path}")

    # Use a set to avoid downloading the same model multiple times
    unique_model_names = set(ALL_MODELS.values())

    for model_name in unique_model_names:
        # The cache_dir is the base path. The download function will handle subdirectories.
        download_model(model_name, cache_dir=base_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download all required models from Hugging Face Hub.")
    parser.add_argument(
        '--path',
        type=str,
        default='./models_cache',
        help='The base directory to download models to. Defaults to ./models_cache'
    )
    args = parser.parse_args()

    print(f"Starting model download process to base path: {args.path}")
    download_all_models(args.path)
    print("Model download process finished.")
