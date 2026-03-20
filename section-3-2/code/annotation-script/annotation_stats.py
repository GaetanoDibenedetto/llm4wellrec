import os
import statistics as stat

if __name__ == "__main__":
    BASE_PATH = "texts"
    folders = os.listdir(BASE_PATH)

    line_length: list[int] = []

    for folder in folders:
        folder_path = os.path.join(BASE_PATH, folder)
        folder_content = os.listdir(folder_path)

        for file in folder_content:
            file_path = os.path.join(folder_path, file)

            with open(file) as fptr:
                lines = list(map(lambda l: l.strip().replace("\n", ""), fptr.readlines()))
                line_length.extend([len(l) for l in lines])

    print(f"Media lunghezza: {stat}")