import os.path as path_m
import os
import shutil
from tqdm import tqdm


def copy(src: str, dest: str):
    print(f"Copying files from {src} into {dest}")
    files = os.listdir(src)

    pbar = tqdm(files)
    pbar.set_description(f'Processing: {src}')

    for file in pbar:
        shutil.copyfile(
            path_m.join(src, file),
            path_m.join(dest, file)
        )

def create_folder(path: str):
    print(f"Creating {path} folder")
    os.mkdir(path)

if __name__ == '__main__':
    custom_data_path = "./CustomData/"
    origin_path = "../HumanML3D/HumanML3D/"
    new_joints_name = "new_joints"
    new_joint_vecs_name = "new_joint_vecs"
    processed_text_name = "processed_text"


    humanml_texts_path = "../HumanML3D/HumanML3D/texts"
    new_humanml_texts_path = "dataset/HumanML3D/texts"

    copy(humanml_texts_path, new_humanml_texts_path)

    if path_m.exists(custom_data_path):
        print(f"Cleaning {custom_data_path} folder")
        shutil.rmtree(custom_data_path)
    
    create_folder(custom_data_path)

    new_joints_name_path = path_m.join(custom_data_path,new_joints_name)
    create_folder(new_joints_name_path)

    new_joint_vecs_path = path_m.join(custom_data_path, new_joint_vecs_name)
    create_folder(new_joint_vecs_path)

    processed_text_path = path_m.join(custom_data_path, processed_text_name)
    create_folder(processed_text_path)


    src = path_m.join(origin_path, new_joints_name)
    dest = path_m.join(custom_data_path, new_joints_name)
    copy(src, dest)

    
    src = path_m.join(origin_path, new_joint_vecs_name)
    dest = path_m.join(custom_data_path, new_joint_vecs_name)
    copy(src, dest)

    src = path_m.join("../HumanML3D/", processed_text_name)
    dest = path_m.join(custom_data_path, processed_text_name)
    copy(src, dest)

    mean_src = path_m.join(origin_path, "Mean.npy")
    mean_dest = path_m.join(custom_data_path, "Mean.npy")
    print(f"Copying 'Mean.py' from {mean_src} into {mean_dest}")
    shutil.copyfile(mean_src, mean_dest)

    std_src = path_m.join(origin_path, "Std.npy")
    std_dest = path_m.join(custom_data_path, "Std.npy")
    print(f"Copying 'Std.py' from {std_src} into {std_dest}")
    shutil.copyfile(std_src,std_dest)
   