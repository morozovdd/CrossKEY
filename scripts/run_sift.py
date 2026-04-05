import argparse
import logging
import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.sift import SIFT3D

logger = logging.getLogger("crosskey.sift")


def parse_args():
    parser = argparse.ArgumentParser(description="Run SIFT3D keypoint extraction")
    parser.add_argument("--input-dir", type=str, default="./data/img",
                        help="Input data directory containing mr/ and synthetic_us/ subdirs")
    parser.add_argument("--output-dir", type=str, default="./data/sift_output",
                        help="Output directory for SIFT descriptors")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of parallel workers")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging")
    return parser.parse_args()


def get_nifti_files(folder_path):
    """Get all .nii.gz files from a folder, sorted."""
    folder = Path(folder_path)
    if not folder.exists():
        return []
    return sorted(str(f) for f in folder.glob("*.nii.gz"))


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    os.makedirs(args.output_dir, exist_ok=True)
    try:
        sift = SIFT3D()
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)

    mr_files = get_nifti_files(os.path.join(args.input_dir, "mr"))
    if mr_files:
        logger.info("Processing %d MR files...", len(mr_files))
        mr_output = os.path.join(args.output_dir, "mr")
        os.makedirs(mr_output, exist_ok=True)
        sift.process_images(mr_files, mr_output, preprocess=True, max_workers=args.workers)

    synth_files = get_nifti_files(os.path.join(args.input_dir, "synthetic_us"))
    if synth_files:
        logger.info("Processing %d synthetic US files...", len(synth_files))
        synth_output = os.path.join(args.output_dir, "synthetic_us")
        os.makedirs(synth_output, exist_ok=True)
        sift.process_images(synth_files, synth_output, preprocess=True, max_workers=args.workers)

    logger.info("SIFT processing completed. Results: %s", args.output_dir)


if __name__ == "__main__":
    main()
