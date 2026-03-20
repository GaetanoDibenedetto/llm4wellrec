import json
from dataclasses import dataclass
from logconfig import create_logger
from typing import Literal, Callable
import csv
import os
import shutil
import statistics as stats
import fractions as fract

logger = create_logger()

JSON_PATH: str = "./annotations.json"
CSV_FILE_PATH: str = "./video-csv/"
ACTION_PROP: str = "video_second"
LATERAL_ACTION_PROP: str = "hand_distance_horizontal_from_mid_ankles_cm"
TEXTS_FOLDER_PATH: str = "texts/"
MAX_ZEROS: int = 5

MIN_D_FACTOR = 25.0
MAX_D_FACTOR = 175.0

@dataclass
class Action:
    video_second: int
    box_height_cm: float
    box_distance_horizontal_from_mid_ankles_cm: int | str
    hand_vertical_distance_from_floor_cm: int | str
    angle_of_asymmetry_degree: int

    def get_box_height(self) -> float:
        return self.box_height_cm


@dataclass
class LateralAction:
    action: Action
    hand_distance_horizontal_from_mid_ankles_cm: int | str

    def get_box_height(self) -> float:
        return self.action.box_height_cm


@dataclass
class Annotation:
    video: str
    box_width_cm: float
    box_height_cm: float
    box_depth_cm: float
    subject_height_cm: int
    action_start: Action | LateralAction
    action_end: Action | LateralAction
    subjective_judgment: str
    one_hand_lifting: bool
    multiple_subjects_lifting: bool
    subject_age: int
    subject_gender: str
    Note: str


def obj_parser(kwargs: dict):
    if ACTION_PROP in kwargs:
        if LATERAL_ACTION_PROP in kwargs:
            v: int | str = kwargs.pop(LATERAL_ACTION_PROP)
            action = Action(**kwargs)
            return LateralAction(action, v)
        
        return Action(**kwargs)
    return Annotation(**kwargs)


def remove_incorrect_video(annotation: Annotation) -> bool:
    black_list = [
        "C. CARRIOLA-TERRA 3quarti 1.mp4",
        "A. TERRA-CARRIOLA 3quarti 2.mp4",
        "D. TERRA-CARRIOLA 3quarti 2.mp4",
        "D. CARRIOLA-TERRA 3quarti 1.mp4",
        "D. CARRIOLA-TERRA 3quarti 2.mp4",
        "D. TERRA-CARRIOLA 3quarti 3.mp4",
        "A. CARRIOLA-TERRA 3quarti 1.mp4",
        "C. TERRA-CARRIOLA 3quarti 2.mp4",
        "D. CARRIOLA-TERRA 3quarti 3.mp4",
        "D. TERRA-CARRIOLA 3quarti 1.mp4",
        "A. TERRA-CARRIOLA 3quarti 1.mp4",
    ]

    return annotation.video not in black_list


def format_action(action: Callable[[Annotation], str]):
    def formatter(annotation: Annotation, **kwargs: float) -> str:
        return action(annotation, **kwargs)

    return formatter


def format_with_niosh(annotation: Annotation, **kwargs: float) -> str:
    height_start: float = annotation.action_start.get_box_height()
    height_end: float = annotation.action_end.get_box_height()
    direction_type: Literal["higher"] | Literal["lower"]
    quantity_type: Literal["much"] | Literal["slightly"]

    if height_end >= height_start:
        direction_type = "higher"
    else:
        direction_type = "lower"

    d_factor: float = abs(height_end - height_start)
    if d_factor < MIN_D_FACTOR:
        d_factor = MIN_D_FACTOR
    elif d_factor > MAX_D_FACTOR:
        d_factor = MAX_D_FACTOR

    dm_muliplier_median = kwargs["dm_muliplier_median"]
    d_multiplier: float = 0.82 + (4.5 / d_factor)
    is_low_vertical_displacment = fract.Fraction.from_float(d_multiplier) <= fract.Fraction.from_float(dm_muliplier_median)

    if is_low_vertical_displacment:
        quantity_type = "slightly"
    else:
        quantity_type = "much"

    return f"a person is moving a box to a {quantity_type} {direction_type} position"


def format_with_direction(annotation: Annotation, **kwargs: float) -> str:
    action_type: Literal["lifting"] | Literal["laying"]
    starting_point: Literal["from the ground"] | Literal["on the ground"]
    height_start: float = annotation.action_start.get_box_height()
    height_end: float = annotation.action_end.get_box_height()

    if height_end >= height_start:
        action_type = "lifting"
        starting_point = "from the ground"
    else:
        action_type = "laying"
        starting_point = "on the ground"
        

    return f"a person is {action_type} a box {starting_point}"

def format_with_gender(annotation: Annotation, **kwargs: float) -> str:
    gender: str = "woman"

    if annotation.subject_gender == "M":
        gender = "man"

    
    return f"a {gender} is moving something, with both hands"

def format_with_gender_and_action(annotation: Annotation, **kwargs: float) -> str:
    gender: Literal["woman"] | Literal["man"] = "woman"

    if annotation.subject_gender == "M":
        gender = "man"

    action_type: Literal["picking up"] | Literal["putting down"]
    height_start: float = annotation.action_start.get_box_height()
    height_end: float = annotation.action_end.get_box_height()

    if height_end >= height_start:
        action_type = "picking up"
    else:
        action_type = "putting down"

    
    return f"a {gender} is {action_type} something, with both hands"

def build_file_names(file_name: str) -> tuple[str, str]:
    normal_file: str = file_name.replace(".mp4", ".txt")
    mirrored_file: str = f"M{normal_file}"

    return (normal_file, mirrored_file)

def get_video_folder() -> list[str]:
    folders: list[str] = os.listdir("..")
    video_folders = sorted(
        filter(
            lambda f: f.startswith("video_"), folders
        )
    )
    
    
    return video_folders

def read_annotation_json() -> list[Annotation]:
    data: list[Annotation] = []
    with open(JSON_PATH) as file:
        data = json.load(file, object_hook=obj_parser)
    
    annotations: list[Annotation] = list(filter(remove_incorrect_video, data))

    return annotations

def get_gender(subject_gender: str) -> str:
    if subject_gender == "M":
        return "male"
    
    return "female"

def create_csv_row():
    row: list[str] = []

    def row_builder(column: str):
        nonlocal row
        row.append(column)

        return row
    return row_builder

def normalize_videos_name(folder, videos) -> list[str]:
    n_videos = []

    for v in videos:
        if " " in v:
            new_v = v.replace(" ", "-")
            old_path = f"../{folder}/{v}"
            new_path = f"../{folder}/{new_v}"

            os.rename(old_path, new_path)

            if not os.path.exists(new_path):
                raise ValueError(f"Cannot rename {old_path} to {new_path}")


        n_videos.append(v)

    return n_videos

def normalize_annotations_video_name(annotation: Annotation) -> Annotation:
    if " " in annotation.video:
        annotation.video = annotation.video.replace(" ", "-")
        
    return annotation

def clear(folder: str):
        shutil.rmtree(folder)


def dm_median(annotations: list[Annotation]) -> float:
    d_multipliers: list[float] = []

    for annotation in annotations:
        height_start: float = annotation.action_start.get_box_height()
        height_end: float = annotation.action_end.get_box_height()
        d_factor: float = abs(height_end - height_start)

        if d_factor < MIN_D_FACTOR:
            d_factor = MIN_D_FACTOR
        elif d_factor > MAX_D_FACTOR:
            d_factor = MAX_D_FACTOR

        d_multiplier: float = 0.82 + (4.5 / d_factor)
        d_multipliers.append(d_multiplier)
    
    d_multipliers = sorted(d_multipliers)

    return stats.median(d_multipliers)


def count_based_on_thresh(annotations: list[Annotation], threshold: float) -> int:
    counter: int = 0

    for annotation in annotations:
        height_start: float = annotation.action_start.get_box_height()
        height_end: float = annotation.action_end.get_box_height()
        d_factor: float = abs(height_end - height_start)
      
        if d_factor < MIN_D_FACTOR:
            d_factor = MIN_D_FACTOR
        elif d_factor > MAX_D_FACTOR:
            d_factor = MAX_D_FACTOR

        d_multiplier: float = 0.82 + (4.5 / d_factor)
        if fract.Fraction.from_float(d_multiplier) <= fract.Fraction.from_float(threshold):
            counter += 1

    return counter


if __name__ == "__main__":
    if os.path.exists(CSV_FILE_PATH):
        logger.info(f"Cleaning {CSV_FILE_PATH}")
        clear(CSV_FILE_PATH)

    os.mkdir(CSV_FILE_PATH)

    if os.path.exists(TEXTS_FOLDER_PATH):
        logger.info(f"Cleaning {TEXTS_FOLDER_PATH}")
        clear(TEXTS_FOLDER_PATH)

    os.mkdir(TEXTS_FOLDER_PATH)
    

    video_folders: list[str] = get_video_folder()
    annotations: list[Annotation] = read_annotation_json()
    annotations = list(
        map(normalize_annotations_video_name, annotations)
    )

    # logger.info(count_based_on_thresh(annotations, dm_mean(annotations)))

    dm_muliplier_median: float = dm_median(annotations)
    logger.info("Folders found: %s", ", ".join(video_folders))

    file_counter = 1
    for video_folder in video_folders:
        videos: list[str] = sorted(
            os.listdir(f"../{video_folder}/")
        )

        videos = normalize_videos_name(video_folder, videos)
        csv_file_path: str = f"{os.path.join(CSV_FILE_PATH, video_folder)}.csv"
        logger.info("Working on: %s", csv_file_path)
        annotation_path: str = os.path.join(TEXTS_FOLDER_PATH, video_folder)
        logger.info("Annotations will be placed here: %s", annotation_path)

        if not os.path.exists(annotation_path):
            os.mkdir(annotation_path)

        current_annotations =  sorted(
            [a for a in annotations if a.video in videos],
            key=lambda a: a.video
        )


        with open(csv_file_path, mode="a") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=",")

            for annotation in current_annotations:
                video_path = os.path.join(video_folder, annotation.video)

                if video_path is None:
                    break

                if not os.path.exists(f"../{video_path}"):
                    logger.warning(
                        "[WARNING] the following video '%s' does not exists! ",
                        video_path
                    )

                    continue
                
                logger.info("Creating an annotaion for: %s", annotation.video)

                normal_file, mirrored_file = build_file_names(annotation.video)
                normal_file_path = f"{os.path.join(annotation_path, normal_file)}"
                mirrored_file_path = f"{os.path.join(annotation_path, mirrored_file)}"

                dir_formatter = format_action(format_with_direction)
                gender_formatter = format_action(format_with_gender)
                multiple_formatter = format_action(format_with_gender_and_action)
                niosh_formatter = format_action(format_with_niosh)

                lines = [
                    f"{dir_formatter(annotation)}\n",
                    f"{gender_formatter(annotation)}\n",
                    f"{multiple_formatter(annotation)}\n",
                    f"{niosh_formatter(annotation, dm_muliplier_median=dm_muliplier_median)}\n",
                ]

                with open(normal_file_path, 'a') as w:
                    w.writelines(lines)

                with open(mirrored_file_path, 'a') as w:
                    w.writelines(lines)

                gender = get_gender(annotation.subject_gender)

                row_builder = create_csv_row() 
                row = row_builder(video_path)
                row = row_builder(gender)
                row = row_builder(normal_file_path)
                row = row_builder(mirrored_file_path)

                csv_writer.writerow(row)
                logger.info("Added the following row: %s", ", ".join(row))

                file_counter += 1
            
            logger.info("Finished for: %s", video_folder)
        
    logger.info("Processed %d file(s)", file_counter)