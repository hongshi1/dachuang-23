import os

def process_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            process_txt_file(file_path)

def process_txt_file(file_path):
    output_lines = []

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) == 2:
                text = parts[0]
                num = float(parts[1])
                new_num = num * 1000
                if new_num == 0:
                    new_line = f"{text} 0\n"
                else:
                    new_line = f"{text} {new_num}\n"

                output_lines.append(new_line)

    with open(file_path, 'w') as f:
        f.writelines(output_lines)

if __name__ == "__main__":
    folder_path = "C:/Users/lenovo/Desktop/dachuang/dachuang-23/data/txt"
    process_files_in_folder(folder_path)
    print("ok")
