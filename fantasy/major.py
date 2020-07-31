"""Mnemonic major system based on fantasy literature."""

import os
import re
import sys
import json
import glob
import click
import string
import time
import json
import random
import curses
import operator
import crayons
import itertools
import unicodedata
import requests
import subprocess
import functools
import collections
import eng_to_ipa
import spacy
import nltk
import lxml.html.soupparser


# You need to run "spacy download en" to get the English model
nlp = spacy.load('en')

# Mapping from number to sounds
num_to_phones = {0: ['s', 'z'], 1: ['t', 'd', 'ð'], 2: ['n', 'ŋ'], 3: ['m'], 4: ['r'],
                 5: ['l'], 6: ['ʤ', 'ʧ', 'ʃ', 'ʒ'], 7: ['k', 'g'], 8: ['f', 'v', 'θ'],
                 9: ['p', 'b']}

# Reverse mapping from sound to number
phone_to_num = {x: k for k, v in num_to_phones.items() for x in v}

ipa_dict_path = 'data/ipa-dict.json'

# Punctuation including unicode chars
punctuation = ''.join(chr(i) for i in range(sys.maxunicode)
                      if unicodedata.category(chr(i)).startswith('P'))


def preprocess(text):
    """Strip punctuation between words, normalize space and lowercase."""
    return ' '.join(x.strip(punctuation).lower().replace('’', '\'')
                    for x in text.split())


def remove_double_quotation_marks(s):
    return s.translate(str.maketrans(dict.fromkeys(['\u201C', '\u201D', '\u0022'])))


def harry_potter_text():
    """Return entire content of the Harry Potter books in a single string."""
    data = []
    for filename in glob.glob('data/json/Harry Potter*'):
        with open(filename) as f:
            data.append(json.load(f)['text'])
    return ' '.join(data)


def major_decode_from_ipa(ipa):
    """Convert IPA to number sequence."""
    result = []
    for char in ipa:
        if (num := phone_to_num.get(char)) is not None:
            result.append(num)
    return result


def get_ipa_pref(word, ipa):
    """Let user choose preferred IPA for word."""
    print()
    print(f'Choose IPA for word "{word}":')
    for i, w in enumerate(ipa):
        print(f'{i}: {w}')
    choice = int(input('Your choice: '))
    print()
    return ipa[choice]


class NoIPAFound(Exception):
    """Failed to find IPA representation in ipa_dict."""


def word_to_ipa(ipa_dict, word):
    """Return IPA for word."""
    try:
        return ipa_dict[word]
    except KeyError:
        raise NoIPAFound(word)


def text_to_ipa(ipa_dict, text):
    """Return all IPA for text."""
    words = preprocess(text).split()
    return ' '.join(word_to_ipa(ipa_dict, x) for x in words)


def has_ipa(ipa_dict, text):
    """Test if there is IPA for text."""
    try:
        text_to_ipa(ipa_dict, text)
    except NoIPAFound:
        return False
    return True


def major_decode_from_text(ipa_dict, text, group_by_words=False):
    """Decode text and return number sequence."""
    return [major_decode_from_ipa(text_to_ipa(ipa_dict, x)) for x in text.split()] \
            if group_by_words else major_decode_from_ipa(text_to_ipa(ipa_dict, text))


def load_json_file_or_dict(filename):
    """Load data from json file if exists otherwise return empty dict."""
    if os.path.isfile(filename):
        with open(filename) as f:
            return json.load(f)
    return dict()


def save_to_json_file(data, filename):
    """Save data to json file."""
    with open(filename, 'w') as f:
        json.dump(data, f)


def load_ipa_dict():
    """Load IPA dict from json file."""
    return load_json_file_or_dict(ipa_dict_path)


def save_ipa_dict(ipa_dict):
    """Save IPA dict to json file."""
    save_to_json_file(ipa_dict, ipa_dict_path)


def populate_ipa_dict_from_text(text):
    """Get all IPA information from eng_to_ipa and save to ipa_dict."""
    ipa_dict = load_ipa_dict()
    words = preprocess(text).split()
    for word in set(words) - set(ipa_dict.keys()):
        ipa = eng_to_ipa.convert(word, retrieve_all=True, keep_punct=False,
                                 stress_marks=False)
        ipa_dict[word] = ipa
    save_ipa_dict(ipa_dict)


def disambiguate_ipas_in_ipa_dict():
    """Disambiguate IPAs for each word by choosing one."""
    ipa_dict = load_ipa_dict()
    try:
        for word, ipa in ipa_dict.items():
            if isinstance(ipa, list):
                if len(ipa) == 1 \
                   or len(set(tuple(major_decode_from_ipa(x)) for x in ipa)) == 1:
                    ipa_dict[word] = ipa[0]
                else:
                    ipa_dict[word] = get_ipa_pref(word, ipa)
    except:
        save_ipa_dict(ipa_dict)
        raise
    save_ipa_dict(ipa_dict)


def ipa_from_youglish(word):
    """Scrape IPA for word from youglish.com."""
    url = f'https://youglish.com/pronounce/{word}/english?'
    while True:
        print(f'Scraping word "{word}" from youglish...', end='', flush=True)
        response = requests.get(url)
        if 'Usage limit exceeded' in response.text:
            raise Exception('YouGlish usage limit exceeded')
        root = lxml.html.soupparser.fromstring(response.text)
        if root.xpath('//div[@class="g-recaptcha"]'):
            print('RECAPTCHA')
            input(f'Open {url}, submit CAPTCHA challenge, press enter to continue.')
        else:
            break
    time.sleep(random.random() * 3)
    d = root.xpath('//div[@id="phoneticPanel"]/div/ul[@class="transcript"]'
                  '/li/span[contains(text(), "Traditional IPA")]'
                  '/following-sibling::text()')
    if d:
        print('SUCCESS')
        return d[0].strip(' ˈ')
    print('FAILED')


def find_missing_ipas(text):
    """Attempt to source missing IPAs or delete from ipa_dict, frequent words first."""
    ct = collections.Counter(preprocess(text).split())
    ipa_dict = load_ipa_dict()
    for word, count in ct.most_common():
        if word not in ipa_dict:
            continue
        ipa = ipa_dict[word]
        if m := re.match(r'(\S+)\*', ipa):
            print(ct[word], end='\t')
            ipa = ipa_from_youglish(m.group(1))
            if ipa:
                ipa_dict[word] = ipa
            else:
                del ipa_dict[word]
            save_ipa_dict(ipa_dict)


def numseq_to_str(numseq):
    """Convert number sequence to string."""
    return ''.join(str(x) for x in numseq)


def str_to_numseq(s):
    """Convert string to number sequence."""
    return [int(x) for x in s if x.isdigit()]


def train_decoding():
    """Interactively train decoding of words."""
    ipa_dict_items = list(load_ipa_dict().items())
    while True:
        word, ipa = random.choice(ipa_dict_items)
        numseq = major_decode_from_ipa(ipa)
        while True:
            print('Word is', crayons.blue(word, bold=True))
            user_input = input('Enter number: ')
            user_numseq = str_to_numseq(user_input)
            if numseq == user_numseq:
                print(f'{word} -> {ipa} -> {numseq}')
                print(crayons.green('Correct!\n'))
                break
            else:
                print(crayons.red('Try again!\n'))


def train_encoding(min_numseq_len=1, max_numseq_len=4):
    """Interactively train encoding of numbers."""
    ipa_dict = load_ipa_dict()
    while True:
        numseq_len = random.randint(min_numseq_len, max_numseq_len)
        numseq = [random.randint(0, 9) for _ in range(numseq_len)]
        numseq_str = numseq_to_str(numseq)
        while True:
            print('Number is', crayons.magenta(numseq_str, bold=True))
            user_input = input('Enter text: ')
            try:
                user_ipa = text_to_ipa(ipa_dict, user_input.strip())
            except NoIPAFound as e:
                print(crayons.red(f'No IPA found for "{e.args[0]}". Try again!\n'))
                continue
            user_numseq = major_decode_from_ipa(user_ipa)
            print(f'{user_input} -> {user_ipa} -> {user_numseq}')
            if user_numseq == numseq:
                print(crayons.green('Correct!\n'))
                break
            else:
                print(crayons.red('Try again!\n'))


def load_numseq_index(filename):
    """Load (number sequence -> text) index."""
    index = load_json_file_or_dict(filename)
    return {k: set(v) for k, v in index.items()}


def save_numseq_index(index, filename):
    """Save (number sequence -> text) index."""
    index = {k: list(v) for k, v in index.items()}
    save_to_json_file(index, filename)


def combine_numseq_indexes(indexes):
    """Combine multiple (number sequence -> text) indexes into one."""
    d = dict()
    for index in indexes:
        for k, v in index.items():
            for x in v:
                d.setdefault(k, set()).add(x)
    return d


def build_numseq_index_from_strings(filename, ipa_dict, strings, extend_index=True):
    """Create a index for the mapping from number sequences to text and save it."""
    index = load_numseq_index(filename) if extend_index else dict()
    for string in strings:
        try:
            numseq = major_decode_from_text(ipa_dict, string)
        except NoIPAFound:
            continue
        numseq_str = numseq_to_str(numseq)
        index.setdefault(numseq_str,  set()).add(string)
    save_numseq_index(index, filename)


def gen_chunks_with_grammar(text, grammar, tags, loop=1, min_token_num=1):
    """Generate chunks of text using an NLTK grammar."""
    cp = nltk.RegexpParser(grammar, loop=loop)
    for sent in nltk.tokenize.sent_tokenize(text):
        tokens = nltk.tokenize.word_tokenize(preprocess(sent))
        tagged = nltk.pos_tag(tokens)
        for subtree in cp.parse(tagged):
            if isinstance(subtree, nltk.Tree) and subtree.label() in tags:
                leaves = subtree.leaves()
                if len(leaves) >= min_token_num:
                    yield ' '.join(x[0] for x in leaves)


def noun_phrases_from_text(text):
    """Extract noun phrases from text."""
    grammar = r"""
        NBAR:
            {<JJ.*>*<NN.*>+}  # Adjectives and Nouns
        NP:
            {<NBAR>}
            {<NBAR><IN><NBAR>}  # Above, connected with in/of/...
    """
    return gen_chunks_with_grammar(text, grammar, ['NP'], min_token_num=2)


def clauses_from_text(text):
    """Extract clauses from text."""
    grammar = r"""
    NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
    PP: {<IN><NP>}               # Chunk prepositions followed by NP
    VP: {<VB.*><NP|PP|CLAUSE>+$} # Chunk verbs and their arguments
    CLAUSE: {<NP><VP>}           # Chunk NP, VP
    """
    return gen_chunks_with_grammar(text, grammar, ['CLAUSE'])


def nouns_from_text(text):
    """Extract nouns from text."""
    tokens = nltk.tokenize.word_tokenize(preprocess(text))
    return set(word for word, pos in nltk.pos_tag(tokens)
               if pos[:2] == 'NN')


def build_numseq_indexes(ipa_dict, text):
    """Build all relevant indexes for generating numseq encodings."""
    build_numseq_index_from_strings('data/numseq-sentence-index.json', ipa_dict,
                                    nltk.tokenize.sent_tokenize(remove_double_quotation_marks(text)))
    build_numseq_index_from_strings('data/numseq-clause-index.json', ipa_dict,
                                    clauses_from_text(text))
    build_numseq_index_from_strings('data/numseq-noun-phrase-index.json', ipa_dict,
                                    noun_phrases_from_text(text))
    build_numseq_index_from_strings('data/numseq-noun-index.json', ipa_dict,
                                    nouns_from_text(text))
    build_numseq_index_from_strings('data/numseq-word-index.json', ipa_dict,
                                    ipa_dict.keys())


def find_encodings_for_numseq_with_index(numseq, index):
    """Check index for numseq and return encoding if found."""
    numseq_str = numseq_to_str(numseq)
    return list(index.get(numseq_str, []))


def find_encodings_for_numseq_with_index_file(numseq, filename):
    """Load index file and pass to find_encodings_for_numseq_with_index."""
    index = load_numseq_index(filename)
    return find_encodings_for_numseq_with_index(numseq, index)


def find_sentences_for_numseq(numseq):
    return find_encodings_for_numseq_with_index_file(numseq,
                                                     'data/numseq-sentence-index.json')


def find_noun_phrases_for_numseq(numseq):
    return find_encodings_for_numseq_with_index_file(numseq,
                                                     'data/numseq-noun-phrase-index.json')


def find_clauses_for_numseq(numseq):
    return find_encodings_for_numseq_with_index_file(numseq,
                                                     'data/numseq-clause-index.json')


def find_nouns_for_numseq(numseq):
    return find_encodings_for_numseq_with_index_file(numseq,
                                                     'data/numseq-noun-index.json')


def find_words_for_numseq(numseq):
    return find_encodings_for_numseq_with_index_file(numseq,
                                                     'data/numseq-word-index.json')


def load_combined_numseq_index():
    files = glob.glob('data/numseq-*-index.json')
    return combine_numseq_indexes([load_numseq_index(x) for x in files])


def find_all_for_numseq(numseq):
    index = load_combined_numseq_index()
    return find_encodings_for_numseq_with_index(numseq, index)


def calc_coverage_for_numseq_len(length):
    index = load_combined_numseq_index()
    numseqs = gen_all_numseqs_of_length(length)
    encodings = [x for x in numseqs if find_encodings_for_numseq_with_index(x, index)]
    return len(encodings) / len(numseqs)


def interactive_find_noun_sequences_for_numseq(numseq):
    """Interactively find noun sequences for a number sequence."""
    nouns = list(load_numseq_index('data/numseq-noun-index.json').items())

    def _find(s):
        if not s:
            return []
        options = {v: n for n, w in nouns for v in w if s.startswith(n)}
        print('Remaining digits:', crayons.yellow(s, bold=True))
        print(', '.join(options.keys()))
        while True:
            user_noun = input('Choose next noun: ')
            if user_noun in options:
                break
        print()
        user_num = options[user_noun]
        remainder = s[len(user_num):]
        return [user_noun] + _find(remainder)

    return _find(numseq_to_str(numseq))


def interactive_find_for_numseq(find_func):
    while True:
        numseq = [int(x) for x in input('Enter number: ') if x.isdigit()]
        results = find_func(numseq)
        if results:
            print(crayons.green(results), '\n')
        else:
            print('No results found.\n')


def gen_all_numseqs_of_length(length):
    return [list(x) for x in itertools.product(range(10), repeat=length)]


def load_wordlist(filename):
    if os.path.isfile(filename):
        with open(filename) as f:
            return json.load(f)
    return dict()


def save_wordlist(wordlist, filename):
    with open(filename, 'w') as f:
        json.dump(wordlist, f)


def interactive_create_wordlist():
    filename = 'data/wordlist.json'
    wordlist = load_wordlist(filename)
    numbers = [numseq_to_str(x)
               for x in itertools.chain(w for z in range(1, 3)
                                        for w in gen_all_numseqs_of_length(z))]
    missing = sorted(set(numbers) - set(wordlist.keys()))
    for num in missing:
        print('Number:', crayons.green(num, bold=True))
        print('Ideas:', ', '.join(find_nouns_for_numseq(num)))
        user_noun = input('Enter chosen noun: ')
        wordlist[num] = user_noun
        save_wordlist(wordlist, filename)
        print()


def load_leitner_data(filename):
    return load_json_file_or_dict(filename)


def save_leitner_data(data, filename):
    save_to_json_file(data, filename)


def leitner_add_fact(fact, filename='data/leitner.json'):
    data = load_leitner_data(filename)
    facts = data.setdefault('facts', [])
    for f in facts:
        if fact['front'] == f['front']:
            f['back'] = fact['back']
            f['box'] = 0
            break
    else:
        facts.append(dict(front=fact['front'], back=fact['back'], box=0))
    save_leitner_data(data, filename)


def leitner_train(box, filename='data/leitner.json'):
    print('Training box:', box, '\n')
    data = load_leitner_data(filename)
    facts = data.setdefault('facts', [])
    random.shuffle(facts)
    for f in facts:
        if f['box'] == box:
            print('Front:', f['front'])
            user_back = input('Back: ')
            if user_back == f['back']:
                print(crayons.green('Correct!'))
                f['box'] = max(f['box'] + 1, 2)
            else:
                print(crayons.red('Wrong!'), 'Correct answer is', crayons.yellow(f['back']))
                f['box'] = 1
            print()
            save_leitner_data(data, filename)
    print('Done')


def interactive_leitner(filename='data/leitner.json', max_box=5):
    data = load_leitner_data(filename)
    counts = {x: 0 for x in range(max_box+1)}
    for f in data['facts']:
        counts[f['box']] += 1
    counts = sorted(counts.items(), key=operator.itemgetter(0))
    print('Leitner boxes:', ', '.join(f'{crayons.magenta(box, bold=True)}: {ct}'
                                      for box, ct in counts))
    while True:
        try:
            user_box = int(input('Choose box to train: '))
        except ValueError:
            continue
        if user_box >= 0 and user_box <= max_box:
            break
    print()
    leitner_train(user_box)
    print()


def add_wordlist_to_leitner():
    wordlist = load_json_file_or_dict('data/wordlist.json')
    for num, word in wordlist.items():
        leitner_add_fact(dict(front=num, back=word))


def group_numseq_str(numseq_str, group_size=2):
    """12345 -> 12 34 5"""
    return ' '.join(''.join(x)
                    for x in itertools.zip_longest(fillvalue='',
                                                   *[iter(numseq_str)] * group_size))


def curses_input(stdscr, prompt):
    curses.echo()
    stdscr.addstr(prompt)
    stdscr.refresh()
    user_input = []
    while True:
        key = stdscr.getkey()
        if key == '\n':
            break
        user_input.append(key)
    curses.noecho()
    return ''.join(user_input)


def mental_exercise(stdscr):
    curses.echo()
    a, b = (random.randint(0, 1000) for _ in range(2))
    while True:
        try:
            stdscr.clear()
            answer = int(curses_input(stdscr, f'{a} + {b} = '))
        except ValueError:
            continue
        if answer == a + b:
            break
    curses.noecho()


def interactive_memorize_numseq(numseq):
    numseq_str = numseq_to_str(numseq)
    numseq_str_grouped = group_numseq_str(numseq_str)

    def _func(stdscr):
        curses.init_pair(1, curses.COLOR_BLUE, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
        while True:
            stdscr.clear()
            stdscr.addstr('Number sequence: ')
            stdscr.addstr(numseq_str_grouped, curses.color_pair(1) | curses.A_BOLD)
            stdscr.refresh()

            stdscr.getkey()
            mental_exercise(stdscr)
            stdscr.clear()
            user_numseq_str = ''.join(x for x in curses_input(stdscr, 'Enter number: ')
                                      if x.isdigit())
            if user_numseq_str == numseq_str:
                stdscr.clear()
                stdscr.addstr('Correct!', curses.color_pair(2))
                stdscr.getkey()
                break
            else:
                stdscr.clear()
                stdscr.addstr('Wrong! Try again.', curses.color_pair(3))
                stdscr.getkey()

    curses.wrapper(_func)


def train_memorizing(min_numseq_len=4, max_numseq_len=8):
    while True:
        numseq_len = random.randint(min_numseq_len, max_numseq_len)
        numseq = [random.randint(0, 9) for _ in range(numseq_len)]
        interactive_memorize_numseq(numseq)
