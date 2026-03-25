import os
import shutil

CUSTOM_DATASET_PATH = "./amass_data/Custom"
JOINTS_PATH = "./joints"
POSE_DATA_PATH = "./pose_data"
NEW_JOINTS_PATH = "./HumanML3D/new_joints"
NEW_JOINTS_VECS_PATH = "./HumanML3D/new_joint_vecs"
PROCESSED_TEXT_PATH = "./processed_text"
ANIMATION_PATH = "./HumanML3D/animations"

def clear(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)

if __name__ == "__main__":
    # clear(CUSTOM_DATASET_PATH)
    clear(JOINTS_PATH)
    clear(POSE_DATA_PATH)
    clear(ANIMATION_PATH)
    clear(NEW_JOINTS_PATH)
    clear(NEW_JOINTS_VECS_PATH)
    clear(PROCESSED_TEXT_PATH)



