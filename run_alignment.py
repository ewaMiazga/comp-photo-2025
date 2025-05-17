import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from alignment import align_and_crop_raw_images


def main():
    """
    Use example:
        python run_alignment.py --original dataset_raw/long_exp --filtered dataset_raw/filter_long_exp
    """
    parser = argparse.ArgumentParser(description="Runs alignment on photos from 2 folders and saves aligned photos as torch tensors")
    parser.add_argument("--original", type=str, help="Path to folder with original photos")
    parser.add_argument("--filtered", type=str, help="Path to folder with filtered photos")


    args = parser.parse_args()

    original_fns = sorted(os.listdir(args.original))
    filtered_fns = sorted(os.listdir(args.filtered))

    file_paths_pairs = [
        (f"{args.original}/{fn}", f"{args.filtered}/{filtered_fn}")
        for fn, filtered_fn in zip(original_fns, filtered_fns)
    ]

    original_mosaics = []
    filtered_mosaics = []
    for (orig_path, filter_path) in tqdm(file_paths_pairs, desc="Aligning photos", unit="photo", dynamic_ncols=True):
        result = align_and_crop_raw_images(orig_path, filter_path)
        if result is None:
            continue

        orig_result, filter_result = result
        original_mosaics.append(orig_result["mosaic"])
        filtered_mosaics.append(filter_result["mosaic"])

    original_mosaics_torch = torch.tensor(np.stack(original_mosaics, axis=3), dtype=torch.float32).permute(3, 0, 1, 2)
    filtered_mosaics_torch = torch.tensor(np.stack(filtered_mosaics, axis=3), dtype=torch.float32).permute(3, 0, 1, 2)

    print(original_mosaics_torch.shape, filtered_mosaics_torch.shape)

    split_path_orig = args.original.split("/")
    split_path_filter = args.filtered.split("/")
    save_dest_orig = f"{'/'.join(split_path_orig[:-1])}/{split_path_orig[-1]}.pt"
    save_dest_filter = f"{'/'.join(split_path_filter[:-1])}/{split_path_filter[-1]}.pt"

    torch.save(original_mosaics_torch, save_dest_orig)
    torch.save(filtered_mosaics_torch, save_dest_filter)
    


if __name__ == "__main__":
    main()
