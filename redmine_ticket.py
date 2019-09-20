import datetime
import time
from os import listdir
from os.path import isfile, join
import pandas

myPath = 'C:\\Users\\E0196722\\Desktop\\REDMINE_files'
onlyFiles = [f for f in listdir(myPath) if isfile(join(myPath, f))]
list_sql = list()


def convert_to_date(timestamp):
    readable = datetime.date.fromtimestamp(int(time.mktime(timestamp.timetuple()))).isoformat()
    return readable


i = 1
for f in onlyFiles:
    if "xlsx" in f:
        file_directory = myPath + '\\' + f
        excel_file = pandas.read_excel(file_directory)
        announcement_date = excel_file['announcement_date'][0]
        effective_date = excel_file['effective_date'][0]
        id_bb_unique = excel_file['id_bb_unique'][0]
        list_sql.append(excel_file['SQL_query'][0])
for i in list_sql:
    print(i)
