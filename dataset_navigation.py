import os.path
from os import listdir, path

def elements_not_in_all(sets: list[set]) -> set:
    if not sets:
        return set()
    all_elements = set.union(*sets)
    common_elements = set.intersection(*sets)
    return all_elements - common_elements

# def get_image_paths(dataset_name:str = 'dataset_raw', types=None):
#     if types is None:
#         types = ['filter_long_exp', 'long_exp', 'short_exp']
#     image_file_names = []
#     processed = {}
#     for idx, subfolder in enumerate(types):
#         image_dir_path = path.join('.', dataset_name, subfolder)
#         image_file_names = [f for f in listdir(image_dir_path) if path.isfile(path.join(image_dir_path, f)) and f[0] != '.']
#         processed[subfolder] = set(image_file_names)
#
#     common_elements = set.intersection(*list(processed.values()))
#     print(elements_not_in_all(list(processed.values())))
#     file_paths = {}
#     for subfolder in types:
#         file_paths[subfolder] = sorted([path.join('.', dataset_name, subfolder, file_name) for file_name in common_elements])
#
#     return file_paths

def get_image_paths(dataset_name:str ='dataset_raw', types=None):
    if types is None:
        types = ['filter_long_exp', 'long_exp', 'short_exp']

    # Get the list of image file names in each subfolder
    image_file_paths_per_type = {}
    for idx, subfolder in enumerate(types):
        image_dir_path = path.join('.', dataset_name, subfolder)
        image_file_paths = [path.join(image_dir_path, f) for f in sorted(listdir(image_dir_path)) if path.isfile(path.join(image_dir_path, f)) and f[0] != '.']
        image_file_paths_per_type[subfolder] = image_file_paths

    # Check if all folders have the same number of images
    len_old = -1
    for type, file_names in image_file_paths_per_type.items():
        len_new = len(file_names)
        if len_old != -1 and len_old != len_new:
            raise ValueError(f"Different number of images in different folders: {len_old} != {len_new}")
        len_old = len_new
    return image_file_paths_per_type




















if __name__=='__main__':
    print(get_image_paths())
