from bs4 import BeautifulSoup
import json
import csv
from janome.tokenizer import Tokenizer

t = Tokenizer()

with open('result.txt', 'w') as f:
    writer = csv.writer(f)
    for i, row in enumerate(open('items20191218.json')):
        obj = json.loads(row)
        text = obj['rendered_body']
        soup = BeautifulSoup(text, 'html.parser')
        for e in soup.find_all('div', class_='code-frame'):
            e.clear()
        tokens = [token.base_form for token in t.tokenize(soup.get_text())]
        writer.writerow(tokens)
