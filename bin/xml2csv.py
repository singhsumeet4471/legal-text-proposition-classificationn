import os
import csv
import codecs
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup, BeautifulStoneSoup
import unicodedata
import re
from datetime import datetime
# convert_xml2csv('H:\\University of Passau\\Projects\\Text Mining\\corpus\\corpus\\corpus\\citations_class_updated', 'H:\\University of Passau\\Projects\\Text Mining\\')
def convert_xml2csv(xml_directory_path, csv_directory_path):
    xml_content = ''
    data = []
    date_time= datetime.now()
    filename = date_time.strftime('%Y%m%d%H%M%S')+'.csv'
    csv_file = open(csv_directory_path + filename, 'w', newline='')
    csv_writer = csv.writer(csv_file, )
    csv_header = ['file_name','id', 'class', 'tocase', 'text']
    csv_writer.writerow(csv_header)
    for filename in os.listdir(xml_directory_path):
        if not filename.endswith('.xml'): continue
        fullname = os.path.join(xml_directory_path, filename)
        infile = open(fullname, "r")
        xml_content = infile.read()
        soup = BeautifulSoup(xml_content, 'xml')
        for citation in soup.find_all('citation'):
            citation_attr = dict(citation.attrs)
            citation_class = citation.find('class')
            citation_case_name = citation.find('tocase')
            citation_text = citation.find('text')
            #putting data in array
            data.clear()
            data.append(filename.rstrip())
            data.append(citation_attr.get('id'))
            data.append(citation_class.text.rstrip())
            data.append(citation_case_name.text.rstrip())
            data.append(re.sub('[^A-Za-z0-9 ]+', '', citation_text.text.rstrip()))
            if len(data) > 0:
                csv_writer.writerow(data)
            continue

    csv_file.close()
    return
