import os
import pandas as pd
import numpy as np

def pre_data():
    path = ''
    cols = ['name', 'version', 'Name', 'wmc', 'dit','noc','cbo','rfc','lcom','ca','ce','npm','lcom3','loc','dam','moa','mfa','cam','ic','cbm','amc','max_cc','avg_cc','bug']
    for file in os.listdir(path):
        # 检查文件是否以.csv结尾
        if file.endswith('.csv') and file != 'null':
            file_path = os.path.join(path, file)
            # 打开.csv文件
            data = pd.read_csv(file_path, usecols=cols)
            for index, row in data.iterrows():
                another_path = '../../data/txt_png_path/'+row['name'].lower()+'-'+str(row['version']).strip()+'.txt'
                with open(another_path, 'r') as f:
                    flag = 0
                    for line in f:
                        parts = line.split()
                        file_name, _ = os.path.splitext(parts[0])
                        module_name = file_name.split(row['name'].lower()+'-'+str(row['version']).strip()+'_src_java_')[-1]
                        module_name = module_name.replace('_', '.')
                        if module_name.upper() == row['Name'].upper():
                            flag = 1
                            break
                    if flag == 0:
                        data = data.drop(index)
                        print(row['name']+'-'+str(row['version']) + ' ' + row['Name'])
            data.to_csv(file_path, index=False)
            print(file+' success.')





if __name__ == '__main__':
    pre_data()