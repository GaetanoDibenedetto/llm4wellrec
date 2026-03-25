import os


if __name__ == "__main__":
    path = "HumanML3D/texts"
    files = sorted(os.listdir(path))
    total_lines = 0

    for file in files:
        file_p = os.path.join(path, file)
        with open(file_p) as f:
            total_lines += len(f.readlines())
        
    print(total_lines)