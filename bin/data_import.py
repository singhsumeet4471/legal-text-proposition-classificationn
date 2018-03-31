import numpy
import os.path
import csv
import requests
from bs4 import BeautifulSoup
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


def data_csv_import(file_name):
    with open(new_path + '\\' + file_name) as csvfile:
        readCSV = csv.reader(csvfile)
        csv_col_data = []
        for row in readCSV:
            # check for exact duplicate entry
            if row[4].strip() not in csv_col_data:
                csv_col_data.append(row[4].strip() + ".")
    data = ''.join(csv_col_data)
    return data


def remove_duplicates(list):
    newlist = []
    for item in list:
        if item not in newlist:
            newlist.append(item)
    return newlist

def web_scraping():


    url = "http://www.gesetze-im-internet.de/englisch_abgg/englisch_abgg.html#p0333"

    r = requests.get(url)

    data = r.text

    soup = BeautifulSoup(data)
    print(data)
