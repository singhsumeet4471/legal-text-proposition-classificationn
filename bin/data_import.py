import numpy
import os.path

cur_path = os.path.dirname(__file__)
new_path = os.path.relpath('..\\dataset', cur_path)

def data_txt_import_string(file_name):
    file = open(new_path + '\\' + file_name, 'r', encoding='utf-8')
    data = file.readline()
    return data


def data_txt_import_array(file_name):
    file = open(new_path + '\\' + file_name, 'r', encoding='utf-8')
    list = []
    for line in file:
        list.append(line)

    data = ''.join(list)
    return data.lower()

def data_csv_import():
    return
