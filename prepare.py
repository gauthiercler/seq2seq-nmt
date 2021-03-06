from urllib.request import urlopen, Request

import requests
from bs4 import BeautifulSoup
from os import path

from zipfile import ZipFile
from tqdm import tqdm

from torchtext import data


class DataLoader:
    base_url = 'https://www.manythings.org/anki'

    def __init__(self):
        self.dataset = None

    def prepare(self, lang):
        src = data.Field()
        trg = data.Field()
        self.dataset = data.TabularDataset(
            path=f'data/{lang}.txt',
            format='TSV', fields=[('source', src), ('target', trg)])

        return self.dataset

    def download_and_extract(self, lang):
        print('Downloading corpus')
        if path.exists(f'data/{lang}.txt'):
            return
        url = f'{self.base_url}/{lang}-eng.zip'
        request = Request(url, None, {'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) '
                                                    'Gecko/2009021910 Firefox/3.0.7'})
        archive = urlopen(request)
        fd = open(f'/tmp/{lang}.zip', 'wb')
        fd.write(archive.read())
        fd.close()
        zf = ZipFile(f'/tmp/{lang}.zip')
        zf.extract(member=f'{lang.split("-")[0]}.txt', path=f'data/')

        return self

    @staticmethod
    def get_available_lang():
        r = requests.get(DataLoader.base_url, headers={"User-Agent": "XY"})
        data = r.text
        soup = BeautifulSoup(data, features="html.parser")

        lang = []
        for link in soup.find_all('a'):
            href = link.get('href')
            if href is not None and href.endswith('.zip'):
                lang.append(href.split('.')[0].split('-')[0])

        return lang
