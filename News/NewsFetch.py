import json
import csv
import requests

topic = 'Tesla'
from_time = '2016-01-01'
to_time = '2018-09-11'
sortBy = 'publishedAt'
pageSize='100'
api = '01bad423ab464b8c96173b021f5428ed'
lng = 'en'
url = ('https://newsapi.org/v2/everything?q='+topic+'&to='+to_time+'&from='+from_time+'&sortBy='+sortBy+'&apiKey='+api+'&pageSize='+pageSize+'&page=1'+'&language='+lng)

response = requests.get(url)
st = response.text
js = json.loads(st)
print(js["articles"][1]["description"])
f = open("News.json",'w',encoding='utf-8')
f.write(st)
f.close()

file = open("input_data.csv","w",encoding='utf-8')

if True:
    #Making a csv file - Initializing the file writer
    f = csv.writer(file)

    #Declaring the header
    f.writerow(["Id", "Title", "Description", "URL", "Timestamp"])
    k = 1
    #Writing the json documents as records in csv
    for i in js["articles"]:
        f.writerow([
            k,
            i["title"],
            i["description"],
            i["url"],
            i["publishedAt"]
        ])
        k+=1
file.close()


url = ('https://newsapi.org/v2/everything?q='+topic+'&from='+from_time+'&sortBy='+sortBy+'&apiKey='+api+'&pageSize='+pageSize+'&page=2')

response = requests.get(url)
st = response.text
js = json.loads(st)
print(js["articles"][1]["description"])
f = open("News1.json",'w',encoding='utf-8')
f.write(st)
f.close()

file = open("input_data2.csv","w",encoding='utf-8')

if True:
    #Making a csv file - Initializing the file writer
    f = csv.writer(file)

    #Declaring the header
    f.writerow(["Id", "Title", "Description", "URL", "Timestamp"])
    k = 1
    #Writing the json documents as records in csv
    for i in js["articles"]:
        f.writerow([
            k,
            i["title"],
            i["description"],
            i["url"],
            i["publishedAt"]
        ])
        k+=1
file.close()


