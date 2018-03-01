import numpy


def data_txt_import_string(file_name):
    file = open("./dataset/" + file_name, 'r')
    data = file.readline()
    return data


def data_txt_import_array(file_name):
    file = open("./dataset/" + file_name, 'r')
    data=[]
    for line in file:
        data.append(line)
    return data

def data_csv_import():
    return
