import os
import shutil


def get_data_dir_of_thgsp():
    abs_path_of_this_file = __file__
    data_dir = os.path.join(abs_path_of_this_file[:-8], "data")
    return data_dir


def remove_file_or_dir(path):
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)
    else:
        raise ValueError(f"file {path} is not a file or dir.")
