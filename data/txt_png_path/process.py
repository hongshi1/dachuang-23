import os
import csv

def process_files_in_folder(folder_path, ori_folder_path):
    # 列出文件夹下的所有文件
    file_list = os.listdir(ori_folder_path)
    file_name = os.listdir(folder_path)
    for filename in file_name:
        for ori_filename in file_list:
            filename1, extension1 = os.path.splitext(filename)
            filename2, extension2 = os.path.splitext(ori_filename)
            if filename.endswith(".txt") and ori_filename.endswith(".csv") and filename1 == filename2:
                file_path = os.path.join(folder_path, filename)
                ori_file_path = os.path.join(ori_folder_path, ori_filename)
                process_file(file_path, ori_file_path)

def process_file(file_path, ori_file_path):
    output_lines = []
    with open(file_path, 'r') as f, open(ori_file_path, 'r') as ori:
        csv_reader = csv.DictReader(ori)
        for line in f:
            for ori_csv in csv_reader:
                module_name = ori_csv["Name"]
                new_name = module_name.replace(".", "_") + ".png"
                path = line.split("\t")[0]
                name_txt = path.split("src_java_")[-1]
                if new_name == name_txt:
                    new_line = path + " " + ori_csv["bug"] + " " + ori_csv["loc"] + "\n"  # 你可以根据需求调整空格的数量
                    output_lines.append(new_line)
                    break

    with open(file_path, 'w') as f:
        f.writelines(output_lines)

if __name__ == "__main__":
    folder_path = "./"  # 请将这里替换为你的文件夹路径

    # 指定文件夹路径
    ori_folder_path = "C:/Users/gxccc/Desktop/PROMISE-02"

    process_files_in_folder(folder_path,ori_folder_path)
    print("处理完成")
