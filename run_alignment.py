# import os
# import argparse
# import numpy as np
# import torch
# from tqdm import tqdm
# from alignment import align_and_crop_raw_images

# def save_in_chunks(images_list, save_dest, num_chunks):
#     N = len(images_list)
#     k = N // num_chunks
#     for i in range(num_chunks):
#         start = i * k
#         # make the *last* chunk go to the very end
#         end   = (i + 1) * k if i < num_chunks - 1 else N

#         chunk = images_list[start:end]
#         # stack into a torch tensor: shape (chunk_size, H, W, C)
#         t = torch.stack([torch.from_numpy(img) for img in chunk], dim=0)
#         # permute to (C, H, W, chunk_size) if you really want that
#         t = t.permute(3, 1, 2, 0).float()

#         torch.save(t, f"{save_dest}_chunk{i}.pt")
#         print(f"  → saved {end-start} images to {save_dest}_chunk{i}.pt")

#         # free before next chunk
#         del t
#         torch.cuda.empty_cache()

# def main():
#     """
#     Use examples:
#         python run_alignment.py --original dataset_raw/long_exp --filtered dataset_raw/filter_long_exp
#         python run_alignment.py --original dataset_raw/long_exp --filtered dataset_raw/filter_long_exp --short_exp dataset_raw/short_exp
#         python run_alignment.py --original dataset_raw/long_exp --n_photos 10 --downscale 2
#     """
#     parser = argparse.ArgumentParser(description="Runs alignment on photos from 2 or 3 folders and saves aligned photos as torch tensors")
#     parser.add_argument("--original", type=str, help="Path to folder with original photos")
#     parser.add_argument("--filtered", type=str, help="Path to folder with filtered photos")
#     # Optional argument for low exposure images
#     parser.add_argument("--short_exp", type=str, default=None, help="Path to folder with low exposure photos")
#     parser.add_argument("--n_photos", type=int, default=None, help="Number of photos to process")
#     parser.add_argument("--downscale", type=int, default=1, help="Downscale factor for the images")
#     parser.add_argument("--save_chunks", type=int, default=1, help="How many chunks the tensor should be saved in")


#     args = parser.parse_args()

#     original_fns = sorted(os.listdir(args.original))
#     filtered_fns = sorted(os.listdir(args.filtered))
#     short_exp_fns = None
#     if args.short_exp:
#         short_exp_fns = sorted(os.listdir(args.short_exp))

#     if short_exp_fns:
#         file_paths_tuples = [
#             (f"{args.original}/{fn}", f"{args.filtered}/{filtered_fn}", f"{args.short_exp}/{short_exp_fn}")
#             for fn, filtered_fn, short_exp_fn in zip(original_fns, filtered_fns, short_exp_fns)
#         ]
#     else:
#         file_paths_tuples = [
#             (f"{args.original}/{fn}", f"{args.filtered}/{filtered_fn}")
#             for fn, filtered_fn in zip(original_fns, filtered_fns)
#         ]


#     original_mosaics = []
#     filtered_mosaics = []
#     short_exp_mosaics = []
#     if args.n_photos:
#         n_photos = int(args.n_photos)
#         if len(file_paths_tuples) > n_photos:
#             file_paths_tuples = file_paths_tuples[:n_photos]
#     for file_paths_tuple in tqdm(file_paths_tuples, desc="Aligning photos", unit="photo", dynamic_ncols=True):
#         result = align_and_crop_raw_images(*file_paths_tuple, downscale_factor=args.downscale)
#         if result is None:
#             continue
#         if short_exp_fns:
#             orig_result, filter_result, short_exp_result = result
#             original_mosaics.append(orig_result["mosaic"])
#             filtered_mosaics.append(filter_result["mosaic"])
#             short_exp_mosaics.append(short_exp_result["mosaic"])
#         else:
#             orig_result, filter_result = result
#             original_mosaics.append(orig_result["mosaic"])
#             filtered_mosaics.append(filter_result["mosaic"])



#     split_path_orig = args.original.split("/")
#     split_path_filter = args.filtered.split("/")
#     save_dest_filter = f"{'/'.join(split_path_filter[:-1])}/{split_path_filter[-1]}.pt"
#     save_dest_orig = f"{'/'.join(split_path_orig[:-1])}/{split_path_orig[-1]}.pt"

#     # Save the original mosaics as a torch tensor
    
#     save_in_chunks(original_mosaics, save_dest_orig, args.save_chunks)
#     # Clear the original mosaics from memory
#     del original_mosaics

#     # Save the filtered mosaics as a torch tensor
#     save_in_chunks(filtered_mosaics, save_dest_filter, args.save_chunks)

#     # Clear the filtered mosaics from memory
#     del filtered_mosaics

#     # Save the short exposure mosaics as a torch tensor
#     if short_exp_fns:
#         save_dest_short_exp = f"{'/'.join(args.short_exp.split('/')[:-1])}/{args.short_exp.split('/')[-1]}.pt"
#         save_in_chunks(short_exp_mosaics, save_dest_short_exp, args.save_chunks)
#         del short_exp_mosaics

#     print("Done!")
    


# if __name__ == "__main__":
#     main()


import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from utils.alignment import align_and_crop_raw_images
from concurrent.futures import ProcessPoolExecutor


def process_file(task):
    """
    Unpack task tuple and run alignment. Returns mosaics or None.
    task: (orig_path, filt_path, short_path_or_None, downscale)
    """
    orig_path, filt_path, short_path, downscale = task
    if short_path:
        result = align_and_crop_raw_images(orig_path, filt_path, short_path, downscale_factor=downscale)
    else:
        result = align_and_crop_raw_images(orig_path, filt_path, downscale_factor=downscale)
    if result is None:
        return None
    if short_path:
        orig_res, filt_res, short_res = result
        return orig_res["mosaic"], filt_res["mosaic"], short_res["mosaic"]
    else:
        orig_res, filt_res = result
        return orig_res["mosaic"], filt_res["mosaic"]


def save_in_chunks(images_list, save_dest, num_chunks):
    N = len(images_list)
    k = N // num_chunks
    dx = 1.0  # not used here, placeholder if needed
    for i in range(num_chunks):
        start = i * k
        end = (i + 1) * k if i < num_chunks - 1 else N

        chunk = images_list[start:end]
        # stack into a torch tensor: shape (chunk_size, H, W, C)
        t = torch.stack([torch.from_numpy(img) for img in chunk], dim=0)
        # permute to (C, H, W, chunk_size)
        t = t.permute(3, 1, 2, 0).float()

        torch.save(t, f"{save_dest}_chunk{i}.pt")
        print(f"  → saved {end-start} images to {save_dest}_chunk{i}.pt")

        del t
        torch.cuda.empty_cache()


def main():

    parser = argparse.ArgumentParser(
        description="Runs alignment on photos from 2 or 3 folders and saves aligned photos as torch tensors (multi-threaded)"
    )
    parser.add_argument("--original", type=str, required=True,
                        help="Path to folder with original photos")
    parser.add_argument("--filtered", type=str, required=True,
                        help="Path to folder with filtered photos")
    parser.add_argument("--short_exp", type=str, default=None,
                        help="Path to folder with low exposure photos")
    parser.add_argument("--n_photos", type=int, default=None,
                        help="Number of photos to process")
    parser.add_argument("--downscale", type=int, default=1,
                        help="Downscale factor for the images")
    parser.add_argument("--save_chunks", type=int, default=1,
                        help="How many chunks the tensor should be saved in")
    parser.add_argument("--start_index", type=int, default=0,
                        help="What file to start on")

    args = parser.parse_args()
    original_fns = sorted(os.listdir(args.original))
    filtered_fns = sorted(os.listdir(args.filtered))
    short_exp_fns = sorted(os.listdir(args.short_exp)) if args.short_exp else None

    if args.short_exp:
        file_paths = [
            (f"{args.original}/{o}", f"{args.filtered}/{f}", f"{args.short_exp}/{s}", args.downscale)
            for o, f, s in zip(original_fns, filtered_fns, short_exp_fns)
        ]
    else:
        file_paths = [
            (f"{args.original}/{o}", f"{args.filtered}/{f}", None, args.downscale)
            for o, f in zip(original_fns, filtered_fns)
        ]
        
    file_paths = file_paths[args.start_index:]

    if args.n_photos:
        file_paths = file_paths[:args.n_photos]
    

    # Prepare lists to collect mosaics
    original_mosaics = []
    filtered_mosaics = []
    short_exp_mosaics = []

    for file_path in tqdm(file_paths, total=len(file_paths), desc="Aligning photos"):
        result = process_file(file_path)
        if result is None:
            continue
        if args.short_exp:
            o, f, s = result
            original_mosaics.append(o)
            filtered_mosaics.append(f)
            short_exp_mosaics.append(s)
        else:
            o, f = result
            original_mosaics.append(o)
            filtered_mosaics.append(f)

    # Construct save destinations
    split_orig = args.original.rstrip("/").split("/")
    split_filt = args.filtered.rstrip("/").split("/")
    base_orig = "/".join(split_orig[:-1]) + "/" + split_orig[-1] + "_start" + str(args.start_index)
    base_filt = "/".join(split_filt[:-1]) + "/" + split_filt[-1] + "_start" + str(args.start_index)

    # Save tensors in chunks
    save_in_chunks(original_mosaics, base_orig, args.save_chunks)
    del original_mosaics

    save_in_chunks(filtered_mosaics, base_filt, args.save_chunks)
    del filtered_mosaics

    if args.short_exp:
        split_short = args.short_exp.rstrip("/").split("/")
        base_short = "/".join(split_short[:-1]) + "/" + split_short[-1] + "_start" + str(args.start_index)
        save_in_chunks(short_exp_mosaics, base_short, args.save_chunks)
        del short_exp_mosaics

    print("Done!")


if __name__ == "__main__":
    main()