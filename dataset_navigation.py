import os.path
from os import listdir, path

def elements_not_in_all(sets: list[set]) -> set:
    if not sets:
        return set()
    all_elements = set.union(*sets)
    common_elements = set.intersection(*sets)
    return all_elements - common_elements
def process_file_names(names):
    return [process_file_name(name) for name in names]

def process_file_name(name):
        space_loc = name.find(' ')
        period_loc = name.find('.')
        return name[:space_loc] + name[period_loc:]
def get_image_paths(dataset_name:str = 'dataset_raw', types=None):
    if types is None:
        types = ['filter_long_exp', 'long_exp', 'short_exp']
    image_file_names = []
    processed = {}
    processed_to_original = {}
    for idx, subfolder in enumerate(types):
        image_dir_path = path.join('.', dataset_name, subfolder)
        image_file_names = [f for f in listdir(image_dir_path) if path.isfile(path.join(image_dir_path, f))]

        processed_to_original[subfolder] = {process_file_name(original): original for original in image_file_names}
        processed[subfolder] = set(processed_to_original[subfolder].keys())

    common_elements = set.intersection(*list(processed.values()))
    print(elements_not_in_all(list(processed.values())))
    file_paths = {}
    for subfolder in types:
        file_paths[subfolder] = sorted([path.join('.', dataset_name, subfolder, processed_to_original[subfolder][file_name]) for file_name in common_elements])

    return file_paths
            #
            # other_image_file_names = sorted([(f, process_file_name(f)) for f in listdir(image_dir_path) if path.isfile(path.join(image_dir_path, f))]
            #
            # for file_name in processed_file_names:
            #         if file_name not in processed_other_file_names:
            #             print(f"File {file_name} in {types[0]} does not have a match in {subfolder}")
            #
            # for file_name in processed_other_file_names:
            #     if file_name not in processed_file_names:
            #         print(f"File {file_name} in {subfolder} does not have a match in {types[0]}")


if __name__=='__main__':
    print(get_image_paths())
