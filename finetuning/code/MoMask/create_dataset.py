import os
import shutil
from enum import Enum
from torch.utils.data import random_split
import pandas as pd

DATASET_BASE_PATH = "dataset/HumanML3D"
POSE_DATA = "../HumanML3D/pose_data/"
POSE_DATA_CSV = "../HumanML3D/original-index.csv"
CUSTOM_DATA_CSV = "../HumanML3D/corpus.csv"


class ProcessedTextEnum(Enum):
    SRC = "../HumanML3D/processed_text"
    AMASS_SRC = "../HumanML3D/HumanML3D/texts"
    DEST = "dataset/HumanML3D/texts"

class NewJointVecsEnum(Enum):
    SRC = "../HumanML3D/HumanML3D/new_joint_vecs"
    DEST = "dataset/HumanML3D/new_joint_vecs"

class NewJointsEnum(Enum):
    SRC = "../HumanML3D/HumanML3D/new_joints"
    DEST = "dataset/HumanML3D/new_joints"

class MeanEnum(Enum):
    SRC = "../HumanML3D/HumanML3D/Mean.npy"
    DEST = "dataset/HumanML3D/Mean.npy"

class StdEnum(Enum):
    SRC = "../HumanML3D/HumanML3D/Std.npy"
    DEST = "dataset/HumanML3D/Std.npy"


def copy_folder(original, new_path):
    files = os.listdir(original)
    src_files_path = [
        os.path.join(original, f)
        for f in files
    ]

    dest_files_path = [
        os.path.join(new_path, f)
        for f in files
    ]

    for src, dest in zip(src_files_path, dest_files_path):
        if os.path.exists(dest):
            print(f"File already present: {dest}")
            continue

        print(
            f"Copying {src} into {dest}"
        )

        shutil.copy(src, dest)


def copy_file(original, new_path):
    if os.path.exists(new_path):
        print(f"File already present: {new_path}")
        return 
        
    shutil.copy(original, new_path)


def split_dataset(dataset):
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    return (
        train_dataset.indices,
        val_dataset.indices,
        test_dataset.indices
    )


def create_dataset(dataset, subset, name):
    path = os.path.join(DATASET_BASE_PATH, name)

    with open(path, mode="w") as f:
        for idx in subset:
            value = dataset[idx]
            f.write(f"{value}\n")


def copy_pipeline():
    copy_folder(
        ProcessedTextEnum.SRC.value, 
        ProcessedTextEnum.DEST.value
    )

    copy_folder(
        ProcessedTextEnum.AMASS_SRC.value, 
        ProcessedTextEnum.DEST.value
    )

    copy_folder(
        NewJointVecsEnum.SRC.value,
        NewJointVecsEnum.DEST.value
    )
    
    copy_folder(
        NewJointsEnum.SRC.value,
        NewJointsEnum.DEST.value
    )

    copy_file(
        MeanEnum.SRC.value,
        MeanEnum.DEST.value
    )

    copy_file(
        StdEnum.SRC.value,
        StdEnum.DEST.value
    )


def pose_data_dataset_builder():
    pose_data_folders = sorted(os.listdir(POSE_DATA))
    pose_data_frame = pd.read_csv(POSE_DATA_CSV)

    for dataset_name in pose_data_folders:
        print(f"Working on {dataset_name}")
        dataset_rows = pose_data_frame.loc[
            pose_data_frame["source_path"].str.contains(dataset_name),
            ["new_name"]
        ]

        if len(dataset_rows) == 0:
            print(f"\tSkipping: {dataset_name}")
            continue

        dataset = [
            pose_data_frame.iloc[idx]["new_name"].replace(".npy", "")
            for idx in dataset_rows.index
        ]

        train_idxs, val_idxs, test_idxs = split_dataset(dataset)
        create_dataset(dataset, train_idxs, f"trains/{dataset_name}-train.txt")
        create_dataset(dataset, val_idxs, f"vals/{dataset_name}-val.txt")
        create_dataset(dataset, test_idxs, f"tests/{dataset_name}-test.txt")

        print(f"Created the following files in base path {DATASET_BASE_PATH}: ")
        print(f"\ttrais/{dataset_name}-train.txt")
        print(f"\tvals/{dataset_name}-val.txt")
        print(f"\ttests/{dataset_name}-test.txt")


def custom_data_dataset_builder():
    pose_data_frame = pd.read_csv(CUSTOM_DATA_CSV)
    distinc_file_names = pose_data_frame["new_joint_name"].unique()
    dataset = [name.replace(".npy", "") for name in distinc_file_names]

    train_idxs, val_idxs, test_idxs = split_dataset(dataset)
    create_dataset(dataset, train_idxs, f"trains/Custom-train.txt")
    create_dataset(dataset, val_idxs, f"vals/Custom-val.txt")
    create_dataset(dataset, test_idxs, f"tests/Custom-test.txt")

    print(f"Created the following files in base path {DATASET_BASE_PATH}: ")
    print(f"\ttrais/Custom-train.txt")
    print(f"\tvals/Custom-val.txt")
    print(f"\ttests/Custom-test.txt")


def aggregate(folder, file):
    print(f"Aggregating the following folder: {folder}")
    txts = os.listdir(folder)
    all_lines = []

    if os.path.exists(file):
        os.remove(file)

    for txt in txts:
        txt_path = os.path.join(folder, txt)
        with open(txt_path) as f:
            file_lines = f.readlines()
            all_lines.append(file_lines)

    with open(file, mode="x") as f:
        for lines in all_lines:
            f.writelines(lines)



if __name__ == "__main__":
    # copy_pipeline()
    # pose_data_dataset_builder()
    # custom_data_dataset_builder()
    # aggregate("dataset/HumanML3D/trains", "dataset/HumanML3D/train.txt")
    # aggregate("dataset/HumanML3D/vals", "dataset/HumanML3D/val.txt")
    # aggregate("dataset/HumanML3D/tests", "dataset/HumanML3D/test.txt")


    test_dataset_path = "dataset/HumanML3D/Custom-test.txt"
    with open(test_dataset_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace("\n", "")
            txt_file = os.path.join(ProcessedTextEnum.DEST.value, f"{line}.txt")
            new_joints_file = os.path.join(NewJointsEnum.DEST.value, f"{line}.npy")
            new_joint_vecs_file = os.path.join(NewJointVecsEnum.DEST.value, f"{line}.npy")
            
            is_txt_file = os.path.exists(txt_file)
            is_new_joints_file = os.path.exists(new_joints_file)
            is_new_joint_vecs_file = os.path.exists(new_joint_vecs_file)

            if is_txt_file and is_new_joints_file and is_new_joint_vecs_file:
                print(f"{line}\n  {txt_file}:{is_txt_file}\n  {new_joints_file}:{is_new_joints_file}\n  {new_joint_vecs_file}:{is_new_joint_vecs_file}\n")
            else:
                print(f"[WARNING]: Missing {line} file")
                print(f"[WARNING]: {line}\n  {txt_file}:{is_txt_file}\n  {new_joints_file}:{is_new_joints_file}\n  {new_joint_vecs_file}:{is_new_joint_vecs_file}\n")

