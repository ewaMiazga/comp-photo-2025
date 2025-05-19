import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from alignment import align_and_crop_raw_images


def main():
    """
    Use examples:
        python run_alignment.py --original dataset_raw/long_exp --filtered dataset_raw/filter_long_exp
        python run_alignment.py --original dataset_raw/long_exp --filtered dataset_raw/filter_long_exp --short_exp dataset_raw/short_exp
    """
    parser = argparse.ArgumentParser(description="Runs alignment on photos from 2 or 3 folders and saves aligned photos as torch tensors")
    parser.add_argument("--original", type=str, help="Path to folder with original photos")
    parser.add_argument("--filtered", type=str, help="Path to folder with filtered photos")
    # Optional argument for low exposure images
    parser.add_argument("--short_exp", type=str, default=None, help="Path to folder with low exposure photos")
    parser.add_argument("--n_photos", type=str, default=None, help="Number of photos to process")

    args = parser.parse_args()

    original_fns = sorted(os.listdir(args.original))
    filtered_fns = sorted(os.listdir(args.filtered))
    short_exp_fns = None
    if args.short_exp:
        short_exp_fns = sorted(os.listdir(args.short_exp))

    if short_exp_fns:
        file_paths_tuples = [
            (f"{args.original}/{fn}", f"{args.filtered}/{filtered_fn}", f"{args.short_exp}/{short_exp_fn}")
            for fn, filtered_fn, short_exp_fn in zip(original_fns, filtered_fns, short_exp_fns)
        ]
    else:
        file_paths_tuples = [
            (f"{args.original}/{fn}", f"{args.filtered}/{filtered_fn}")
            for fn, filtered_fn in zip(original_fns, filtered_fns)
        ]

    original_mosaics = []
    filtered_mosaics = []
    short_exp_mosaics = []
    if args.n_photos:
        n_photos = int(args.n_photos)
        if len(file_paths_tuples) > n_photos:
            file_paths_tuples = file_paths_tuples[:n_photos]
    for file_paths_tuple in tqdm(file_paths_tuples, desc="Aligning photos", unit="photo", dynamic_ncols=True):
        result = align_and_crop_raw_images(*file_paths_tuple)
        if result is None:
            continue
        if short_exp_fns:
            orig_result, filter_result, short_exp_result = result
            original_mosaics.append(orig_result["mosaic"])
            filtered_mosaics.append(filter_result["mosaic"])
            short_exp_mosaics.append(short_exp_result["mosaic"])
        else:
            orig_result, filter_result = result
            original_mosaics.append(orig_result["mosaic"])
            filtered_mosaics.append(filter_result["mosaic"])



    split_path_orig = args.original.split("/")
    split_path_filter = args.filtered.split("/")
    save_dest_filter = f"{'/'.join(split_path_filter[:-1])}/{split_path_filter[-1]}.pt"
    save_dest_orig = f"{'/'.join(split_path_orig[:-1])}/{split_path_orig[-1]}.pt"

    # Save the original mosaics as a torch tensor
    original_mosaics_torch = torch.tensor(np.stack(original_mosaics, axis=3), dtype=torch.float32).permute(3, 0, 1, 2)
    torch.save(original_mosaics_torch, save_dest_orig)
    print(f"Saved original mosaics to {save_dest_orig}")
    # Clear the original mosaics from memory
    del original_mosaics

    # Save the filtered mosaics as a torch tensor
    filtered_mosaics_torch = torch.tensor(np.stack(filtered_mosaics, axis=3), dtype=torch.float32).permute(3, 0, 1, 2)
    torch.save(filtered_mosaics_torch, save_dest_filter)
    print(f"Saved filtered mosaics to {save_dest_filter}")
    # Clear the filtered mosaics from memory
    del filtered_mosaics

    # Save the short exposure mosaics as a torch tensor
    if short_exp_fns:
        short_exp_mosaics_torch = torch.tensor(np.stack(short_exp_mosaics, axis=3), dtype=torch.float32).permute(3, 0, 1, 2)
        save_dest_short_exp = f"{'/'.join(args.short_exp.split('/')[:-1])}/{args.short_exp.split('/')[-1]}.pt"
        torch.save(short_exp_mosaics_torch, save_dest_short_exp)
        print(f"Saved short exposure mosaics to {save_dest_short_exp}")
        # Clear the short exposure mosaics from memory
        del short_exp_mosaics

    print("Done!")
    


if __name__ == "__main__":
    main()
