"""Download HuggingFace models for the V1 license plate detection application.

This script downloads the required models:
- YOLOv11 license plate detection model
- TrOCR base printed text recognition model
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


MODELS = {
    "yolo": {
        "repo_id": "morsetechlab/yolov11-license-plate-detection",
        "description": "YOLOv11 License Plate Detection",
    },
    "trocr": {
        "repo_id": "microsoft/trocr-base-printed",
        "description": "TrOCR Base Printed Text Recognition",
    },
}

DEFAULT_MODELS_DIR = Path("models")


def download_models(output_dir: Path, models: list[str] | None = None) -> None:
    """Download specified models from HuggingFace Hub.

    Args:
        output_dir: Directory to save downloaded models.
        models: List of model keys to download. If None, downloads all models.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if models is None:
        models = list(MODELS.keys())

    for model_key in models:
        if model_key not in MODELS:
            print(f"Warning: Unknown model key '{model_key}', skipping.")
            continue

        model_info = MODELS[model_key]
        repo_id = model_info["repo_id"]
        description = model_info["description"]
        local_dir = output_dir / model_key

        print(f"\nDownloading {description}...")
        print(f"  Repository: {repo_id}")
        print(f"  Local directory: {local_dir}")

        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
        )
        print(f"  Download complete: {local_dir}")

    print("\nAll models downloaded successfully!")
    print(f"Models directory: {output_dir.resolve()}")


def main() -> None:
    """Main entry point for model download script."""
    parser = argparse.ArgumentParser(
        description="Download HuggingFace models for license plate detection"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_MODELS_DIR),
        help=f"Output directory for models (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        default=None,
        help="Specific models to download (default: all)",
    )

    args = parser.parse_args()
    download_models(Path(args.output_dir), args.models)


if __name__ == "__main__":
    main()
