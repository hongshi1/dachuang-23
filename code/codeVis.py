import numpy
import os,shutil
import re
import binascii
from random import *
import colorMap
import cv2  # pip install opencv-python==4.3.0.38
import time
#代码可视化的文件

rs = Random()

base_path = '/home/'
save_path = '../data/img/'
color_path = 'grb_img/'

def getMatrixfrom_bin(filename, width = 512, oneRow = False):
    with open(filename, 'rb') as f:
        content = f.read()
    hexst = binascii.hexlify(content)
    fh = numpy.array([int(hexst[i:i+2],16) for i in range(0, len(hexst), 2)])
    if oneRow is False:
        rn = len(fh)/width
        fh = numpy.reshape(fh[:rn*width],(-1,width))
    fh = numpy.uint8(fh)
    return fh

def getMatrixfrom_asm(filename, startindex = 0, pixnum = 89478485):
    with open(filename, 'rb') as f:
        f.seek(startindex, 0)
        content = f.read(pixnum)
    hexst = binascii.hexlify(content)
    fh = numpy.array([int(hexst[i:i+2],16) for i in range(0, len(hexst), 2)])
    fh = numpy.uint8(fh)
    return fh

def get_FileSize(filePath):
    filePath = str(filePath)
    #filePath = unicode(filePath,'utf8')
    fsize = os.path.getsize(filePath)
    size = fsize/float(1024)
    return round(size,2)

def get_FileSize(filePath):
    filePath = str(filePath)
    #filePath = unicode(filePath,'utf8')  # 'unicode()' exists in python2 but bot python3
    fsize = os.path.getsize(filePath)
    size = fsize/float(1024)
    return round(size,2)


if __name__ == '__main__':
    if not os.path.exists('../data//txt/null'):
        os.makedirs('../data//txt/null')  # 创建一个新目录
    # 遍历指定文件夹/目录中的所有文件（夹）
    for txtfile in os.listdir('../data//txt'):
        if txtfile == '.DS_Store':
            continue
        if txtfile == '._.DS_Store':
            continue
        if txtfile == 'readme.md':
            continue
        if txtfile == 'null':
            continue

        path_project = save_path + color_path + txtfile.split('.txt')[0]

        if not os.path.exists(path_project):
            os.makedirs(path_project)

        if not os.path.exists(path_project + '/buggy/'):
            os.makedirs(path_project + '/buggy/')

        if not os.path.exists(path_project + '/clean/'):
            os.makedirs(path_project + '/clean/')

        filename = '../data/txt/'+txtfile
        f = open(filename)
        class_num = 0
        java_num = 0
        no_num = 0
        idx_null = 0
        not_exist = []  # define a list
        txt_path_save = ''
        # 遍历每行数据
        for line in f:
            idx_null = idx_null + 1
            filename = txtfile.split('.txt')[0] + '/src/java/'
            f_path = ''
            if line.rsplit('.', maxsplit=1)[0][-1] != '0':
                line = filename + line.replace('\t', '/').replace(' ', '/').replace('.', '/')
            else:
                line = filename + line.replace('\t', '/').replace(' ', '/').replace('.','/')
                line = ".".join(line.rsplit('/', maxsplit=1))

            f_path = '../data/archives/'+line.rsplit('/', maxsplit=1)[0]  # 截取倒数第三个字符之前的所有字符，每行末尾都有一个换行符'\n'
            while f_path[-1] == '/':
                f_path = f_path[:-1]
            # label = line[-2:-1]  # 倒数第二个字符(即缺陷标签，倒数第一个字符是换行符'\n')，其中1表示有缺陷，0表示无缺陷；如果缺陷数目为10/20...，这么处理是错误的
            label = line.split('/')[-1]  # 截取分割后最后一部分
            label = label.split('\n')[0]  # 截取分割后的第一部分


            # start = time.clock()
            if os.path.exists(f_path+'.class'):
                class_num = class_num + 1
                size = get_FileSize(f_path+'.class')
                if size == 0:  # 当前.class文件里的内容为空
                    break
                im = colorMap.get_new_color_img(f_path+'.class')
                if float(label) > 0:  # 有缺陷样本,  label == '1' —— 无法正确处理缺陷数目大于1的样本
                    path_save = path_project +'/buggy/'+''.join(line[:-3]).replace('/','_')+'.png'
                    cv2.imwrite(path_save, im)
                    # im.save(path_save)  # python2
                    numpy.save(path_save, im)  # python3
                else:
                    path_save = path_project +'/clean/'+''.join(line[:-3]).replace('/','_')+'.png'
                    cv2.imwrite(path_save, im)
                    # im.save(path_save)  # python2
                    numpy.save(path_save, im)  # python3
            elif os.path.exists(f_path+'.java'):
                java_num = java_num + 1
                size = get_FileSize(f_path + '.java')
                if size == 0:  # 当前.java文件里的内容为空
                    break
                im = colorMap.get_new_color_img(f_path+'.java')

                try:
                    float_number = float(label)
                except ValueError:
                    print("无法将字符串转换为浮点数:", label)
                    print(line)


                if float(label) > 0:  # label == '1'
                    path_save = path_project + '//buggy/' + ''.join(line[:-3]).replace('/', '_') + '.png'
                    cv2.imwrite(path_save, im)
                    # im.save(path_save)
                    numpy.save(path_save, im)
                else:
                    path_save = path_project + '/clean/' + ''.join(line[:-3]).replace('/', '_') + '.png'
                    cv2.imwrite(path_save, im)
                    # im.save(path_save)  # python2
                    numpy.save(path_save, im)  # python3
                txt_path_save = txt_path_save + path_save + '\t' + label + '\n'
            else:  # 如果文件不存在
                no_num = no_num + 1  # 空模块计数
                not_exist.append(idx_null)  # 空模块的索引
            # end = time.clock()
            # image_time = str(end-start)
        # 有的模块在源码中不存在，需要记录其相对于txt文件中的索引
        numpy.savetxt('../data/txt/null/' + txtfile.split('.txt')[0] + '_null_ins.txt', numpy.array(not_exist), fmt='%d')
        # 保存PNG路径
        with open('../data/txt_png_path/' + txtfile.split('.txt')[0] + '.txt', 'w') as f:  # 指定文件若存在则打开，不存在则创建
            f.write(txt_path_save)
            f.close()
        print(class_num)
        print(java_num)
        print(no_num)