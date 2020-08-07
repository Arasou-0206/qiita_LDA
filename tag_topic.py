import json
import csv
from collections import Counter
import pickle

with open('tag-total.pkl', 'rb') as f:
    tag_list = pickle.load(f)
tag = list(csv.reader(open('tags.txt', 'r', encoding='UTF-8')))
tag_sort = sorted(tag_list.items(), key=lambda x: x[1], reverse=True)
with open('LDA_data.csv') as f:
    reader = csv.reader(f)
    l = [row for row in reader]

data = []
for a in range(100):  # top100 tag
    scale = [0] * 12
    for i in range(len(l)):
        for j in range(len(tag[i])):
            if(tag[i][j] == tag_sort[a][0]):
                for x in range(0, len(l[i]), 2):
                    n = int(l[i][x])
                    d = float(scale[n])
                    d += float(l[i][x + 1])
                    scale[n] = d
    data.append([tag_sort[a][0]])
    data.append(scale)

with open('LDA_tag_data.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(data)
