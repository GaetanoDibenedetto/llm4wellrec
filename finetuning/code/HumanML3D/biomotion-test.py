import os
import csv

if __name__ == "__main__":
    path = "./amass_data/BioMotionLab_NTroje"
    current_folders = sorted(os.listdir(path))

    real_folders = {}
    total = 0
    with open("./test.csv") as f:
        reader = csv.reader(f)
        
        for line in reader:
            f_name, n_items = line
            real_folders[f_name] = int(n_items)
            total += int(n_items)

    print(f"Total items: {total}")

    for folder in list(real_folders.keys()):
        if folder in current_folders:
            sub_path = os.path.join(path, folder)
            sub_path_items = len(os.listdir(sub_path))

            if sub_path_items == real_folders[folder]:
                continue
            else:
                missing_items = real_folders[folder] - sub_path_items
                print(f"[WARNING] folder '{folder}' missed {missing_items} item(s)")
        else:
            print(f"[WARNING] missing folder '{folder}'")

    