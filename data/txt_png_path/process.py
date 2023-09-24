import os

def process_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            process_file(file_path)
def process_file(file_path):
    output_lines = []
    with open(file_path, 'r') as f:
        for line in f:
            new_line = line.rstrip('\n') + '  1\n'  # 你可以根据需求调整空格的数量
            output_lines.append(new_line)

    with open(file_path, 'w') as f:
        f.writelines(output_lines)

if __name__ == "__main__":
    folder_path = "./"  # 请将这里替换为你的文件夹路径
    process_files_in_folder(folder_path)
    print("处理完成")
