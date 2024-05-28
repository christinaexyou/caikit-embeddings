import argparse
import os
import sys
from caikit_nlp.modules.text_embedding import EmbeddingModule

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Bootstrap and save a model for the TextEmbedding module"
    )
    parser.add_argument(
        "-m", "--model-name-or-path", required=True, help="Source model name or path"
    )
    parser.add_argument(
        "-o", "--output-path", required=True, help="Output model config directory"
    )

    args = parser.parse_args()
    return args.model_name_or_path, args.output_path

if __name__ == "__main__":
    model_name_or_path, output_path = _parse_args()
    print("MODEL NAME OR PATH: ", model_name_or_path)
    print("OUTPUT PATH:        ", output_path)
    EmbeddingModule(model_name_or_path).save(output_path)
