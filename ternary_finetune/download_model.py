from huggingface_hub import snapshot_download


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Download model weights from Hugging Face Hub')
    parser.add_argument('--repo-id', type=str, default="meta-llama/Meta-Llama-3-8B",
                      help='Repository ID on Hugging Face Hub (default: meta-llama/Meta-Llama-3-8B)')
    parser.add_argument('--local-dir', type=str, default="model_checkpoints/huggingface/original/Meta-Llama-3-8B",
                      help='Local directory to save model weights (default: model_checkpoints/huggingface/original/Meta-Llama-3-8B)')

    args = parser.parse_args()

    ignore_patterns = ["original/*"] # Llama3 models in the Hub contain the original checkpoints. We just want the HF checkpoint stored in the safetensor format

    snapshot_download(repo_id=args.repo_id,
                     local_dir=args.local_dir, 
                     local_dir_use_symlinks=False,
                     ignore_patterns=ignore_patterns)
