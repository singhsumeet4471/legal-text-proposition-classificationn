import numpy
import os.path
import csv
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import re

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
    NoneType = type(None)
    url = "http://www.gesetze-im-internet.de/englisch_abgg/englisch_abgg.html#p0333"
    r = requests.get(url)

    online_data = r.text
    data= []
    soup = BeautifulSoup(online_data, 'html.parser')
    list = soup.find_all('p')

    csv_directory_path="H:\\University of Passau\\Projects\\Text Mining\\"
    date_time = datetime.now()
    filename = date_time.strftime('%Y%m%d%H%M%S') + '.csv'
    csv_file = open(csv_directory_path + filename, 'w', newline='')
    csv_writer = csv.writer(csv_file, )
    csv_header = ['file_name', 'id', 'class', 'tocase', 'text']
    csv_writer.writerow(csv_header)

    for p in list:
        data.clear()
        data.append('p')
        data.append('id')
        data.append('class')
        data.append('case')
        data.append(re.sub('[^A-Za-z0-9 ]+', '', p.text.rstrip()))
        if len(data) > 0:
            csv_writer.writerow(data)
        print(p.text)

    csv_file.close()
