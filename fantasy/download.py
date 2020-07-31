import requests
import json
import time
import random
from bs4 import BeautifulSoup as bs
import lxml.html.soupparser as sp
from urllib.parse import urljoin, urlencode
import subprocess


def get_possible_authors():
    url = 'https://en.wikipedia.org/wiki/List_of_fantasy_authors'
    r = requests.get(url)
    d = sp.fromstring(r.text)
    exclude = ['Fantasy', 'List of fantasy novels', 'List of high fantasy fiction', 'List of horror fiction authors',
               'List of science fiction authors', 'Lists of authors', 'website', 'Science Fiction and Fantasy Writers of America']
    pauthors = [dict(name=x.text, url=urljoin(url, x.get('href')))
                for x in d.xpath('//div[@class="mw-parser-output"]/ul/li/a')
                if x.text not in exclude]
    return pauthors


def assume_one(l):
    assert len(l) == 1, l
    return l[0]


def get_possible_novels():
    url = 'https://en.wikipedia.org/wiki/List_of_fantasy_novels'
    r = requests.get(url)
    d = sp.fromstring(r.text)
    urls = d.xpath('//div[@class="mw-parser-output"]/ul/li/a/@href')
    for url2 in urls:
        url2 = urljoin(url, url2)
        r = requests.get(url2)
        d = sp.fromstring(r.text)
        for li in d.xpath('//div[@class="mw-parser-output"]/ul/li'):
            try:
                try:
                    title = assume_one(li.xpath('i/a'))
                except AssertionError:
                    title = assume_one(li.xpath('a/i'))
                title_url = urljoin(url, title.get('href'))
            except AssertionError:
                try:
                    title = assume_one(li.xpath('i'))
                    title_url = None
                except AssertionError:
                    continue
            title_text = title.text
            authors = [dict(url=urljoin(url, x.get('href')), name=x.text) for x in li.xpath('a')]
            yield dict(url=title_url, title=title_text, authors=authors)


def search_tpb(query):
    params = dict(q=query, cat=601)
    return requests.get(f'https://apibay.org/q.php?{urlencode(params)}').json()


def get_torrent(info_hash):
    link = 'magnet:?xt=urn:btih:' + info_hash
    subprocess.call(['transmission-remote', '-a', link])


def all_authors():
    to_mb = 1048576
    status = json.load(open('data/status.json'))
    authors = [x['name'] for x in json.load(open('data/fantasy-authors.json'))
               if x['name'] not in status['authors']]
    for author in authors:
        print(author, end=' ')
        results = [x for x in search_tpb(author) if x['info_hash'] not in status['magnets'] \
                   and int(x['seeders']) > 0 and int(x['size']) / to_mb < 20]
        if results:
            print('downloading', results[0])
            get_torrent(results[0]['info_hash'])
            status['magnets'].append(results[0]['info_hash'])
        else:
            print('skipping')
            status['authors'].append(author)
            json.dump(status, open('data/status.json', 'w'))
            time.sleep(random.randint(1, 100))
