import numpy as np
import pandas as pd
import os
import re
def convert_to_vec_path(image_path):
    vec_path = image_path.replace('data/img', 'data/imgVec')
    vec_path = os.path.splitext(vec_path)[0] + '.npy'
    return vec_path

def convert_to_ast_vector_path(image_path):
    ast_path = image_path.replace('data/img/grb_img', 'data/embedding')
    ast_path = os.path.splitext(ast_path)[0] + '.npy'
    return ast_path

if __name__ == '__main__':
    ori_path = '../data/txt_png_path'
    cols = ['name', 'version', 'Name', 'wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3', 'loc',
            'dam', 'moa', 'mfa', 'cam', 'ic', 'cbm', 'amc', 'max_cc', 'avg_cc', 'bug']
    txt_list = os.listdir(ori_path)
    for txt_path_file in txt_list:
        if txt_path_file.endswith('txt'):
            txt_fileName = txt_path_file.split('.txt')[0]
            txt_fileName_path = os.path.join(ori_path, txt_path_file)
            csv_data = pd.read_csv('../data/promise_csv/'+txt_fileName+'.csv', usecols=cols)
            # 在最后一列之前追加两列
            for i in range(100):
                col_name = f'ast_data_{i + 1}'
                if col_name not in csv_data.columns:
                    csv_data.insert(len(csv_data.columns) - 1, col_name, '')
            for i in range(128):
                col_name = f'imgVec_data_{i + 1}'
                if col_name not in csv_data.columns:
                    csv_data.insert(len(csv_data.columns) - 1, col_name, '')

            with open(txt_fileName_path, 'r') as f:
                for line in f:
                    path = line.split(' ')[0]
                    imgVec_path = convert_to_vec_path(path)
                    ast_path = convert_to_ast_vector_path(path)
                    imgVec_data = np.load(imgVec_path).astype(float)
                    ast_data = np.load(ast_path).astype(float)

                    module_name = path.split('src_java_')[1]
                    module_name = module_name.split('.png')[0]
                    module_name = module_name.replace('_', '.')
                    for index, row in csv_data.iterrows():
                        if module_name == row['Name']:
                            for i in range(100):
                                column_index = csv_data.columns.get_loc(f'ast_data_{i + 1}')
                                csv_data.iloc[index, column_index] = ast_data[i]
                            for i in range(128):
                                column_index = csv_data.columns.get_loc(f'imgVec_data_{i + 1}')
                                csv_data.iloc[index, column_index] = imgVec_data[0][i]
                            break
            csv_data.to_csv('../data/promise_csv/' + txt_fileName + '.csv', index=False)
            print(txt_fileName+".csv：", "已完成")
    print('全部完成！')