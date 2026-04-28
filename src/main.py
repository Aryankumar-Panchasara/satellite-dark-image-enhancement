# src/main.py
# Batch runner for the complete pipeline (single-folder dataset).
# Dataset folder directly contains images (no train/test/input/target).

import os
import argparse

from utils import ensure_dir, imread_rgb_float01, imwrite_rgb_uint8
from pipeline import enhance_one_image


def list_images(folder: str) -> list[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    if not os.path.isdir(folder):
        return []
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])


def main():
    parser = argparse.ArgumentParser(description="Satellite Dark Image Enhancement")

    parser.add_argument("--input", type=str, required=True,
                        help="Path to input image OR folder")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output image OR folder")

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    # CASE 1: Single image
    if os.path.isfile(input_path):
        S = imread_rgb_float01(input_path)
        enhanced, _ = enhance_one_image(S)

        out_dir = os.path.dirname(output_path)
        if out_dir:
            ensure_dir(out_dir)

        imwrite_rgb_uint8(output_path, enhanced)

        print(f"✅ Saved enhanced image to: {output_path}")

    # CASE 2: Folder (dataset)
    elif os.path.isdir(input_path):
        if os.path.splitext(output_path)[1]:
            print("❌ Output must be a folder when input is a folder")
            return

        ensure_dir(output_path)

        names = list_images(input_path)
        print(f"Processing {len(names)} images...")

        for i, fname in enumerate(names, 1):
            try:
                in_path = os.path.join(input_path, fname)
                S = imread_rgb_float01(in_path)
                enhanced, _ = enhance_one_image(S)

                out_path = os.path.join(output_path, fname)
                imwrite_rgb_uint8(out_path, enhanced)

                if i % 25 == 0 or i == len(names):
                    print(f"Processed {i}/{len(names)}")

            except Exception as e:
                print(f"[ERROR] Skipping {fname}: {e}")

        print(f"\n✅ Done. Results saved to: {output_path}")

    else:
        print("❌ Invalid input path")

if __name__ == "__main__":
    main()