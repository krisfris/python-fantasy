import os
import glob
import json
import ebooklib
from ebooklib import epub
import lxml.html.soupparser as sp

from bs4 import BeautifulSoup as bs


def pretty_html(html):
    """Use beautifulsoup to prettify html."""
    return bs(html, features='lxml').prettify()


def boom(l):
    """Join list of text and normalize whitespace."""
    return ' '.join(' '.join(l).split())


def main():
    for filename in glob.glob('data/epub/*'):
        try:
            book = ebooklib.epub.read_epub(filename)
        except:
            # Skip unreadable epub files
            continue

        def get_dc(field):
            result = book.get_metadata('DC', field)
            return result[0][0] if result else None

        # Info from metadata is not always available or correct
        # but shall be kept as is for now
        title, creator = get_dc('title'), get_dc('creator')

        # Simply extract all text from the book and normalize whitespace
        text = boom([boom(sp.fromstring(x.get_body_content(),
                                        features='html.parser').xpath('//text()'))
                     for x in book.get_items()
                     if x.get_type() == ebooklib.ITEM_DOCUMENT])

        # Dump content + metadata to json file
        basename = os.path.basename(filename)
        json.dump(dict(title=title, creator=creator, text=text, epub=basename),
                  open(os.path.join('data/json',
                                    os.path.splitext(basename)[0] + '.json'), 'w'))


if __name__ == '__main__':
    main()
