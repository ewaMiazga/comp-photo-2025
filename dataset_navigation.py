import os.path
from os import listdir, path

def elements_not_in_all(sets: list[set]) -> set:
    if not sets:
        return set()
    all_elements = set.union(*sets)
    common_elements = set.intersection(*sets)
    return all_elements - common_elements

def get_image_paths(dataset_name:str = 'dataset_raw', types=None):
    if types is None:
        types = ['filter_long_exp', 'long_exp', 'short_exp']
    image_file_names = []
    processed = {}
    for idx, subfolder in enumerate(types):
        image_dir_path = path.join('.', dataset_name, subfolder)
        image_file_names = [f for f in listdir(image_dir_path) if path.isfile(path.join(image_dir_path, f)) and f[0] != '.']
        processed[subfolder] = set(image_file_names)

    common_elements = set.intersection(*list(processed.values()))
    print(elements_not_in_all(list(processed.values())))
    file_paths = {}
    for subfolder in types:
        file_paths[subfolder] = sorted([path.join('.', dataset_name, subfolder, file_name) for file_name in common_elements])

    return file_paths


if __name__=='__main__':
    print(get_image_paths())
