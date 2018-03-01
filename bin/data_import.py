import numpy
import os.path

cur_path = os.path.dirname(__file__)

new_path = os.path.relpath('..\\dataset', cur_path)


def data_txt_import_string(file_name):
    file = open(new_path + '\\' + file_name, 'r')
    data = file.readline()
    return data


def data_txt_import_array(file_name):
    file = open(new_path + '\\' + file_name, 'r')
    data = []
    for line in file:
        data.append(line)
    return data


def data_csv_import():
    return
