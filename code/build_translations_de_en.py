from urllib.request import urlopen
from urllib.parse import urlencode
import urllib
import json
from gensim_download import pickle_rw
from vocab_vectors import embedding_languages_lgs


def translate_text(source_text, lg_from, lg_to):
    """
    # Fakhri Abbas found and modified this function from
    # http://codegist.net/snippet/python/google-translatepy_lotabout_python
    Keyword arguments:
    lg_from -- The language code for the source text argument
    lg_to -- The language code for the translated language
    source_text -- Text to be translated
    """

    url = 'https://translate.googleapis.com/translate_a/single?'

    params = []
    params.append('client=gtx')
    params.append('sl=' + lg_from)
    params.append('tl=' + lg_to)
    params.append('hl=en-US')
    params.append('dt=t')
    params.append('dt=bd')
    params.append('dj=1')
    params.append('source=input')
    params.append(urlencode({'q': source_text}))
    url += '&'.join(params)

    try:
        request = urllib.request.Request(url)
        browser = "Mozilla/5.0 (X11; Linux x86_64; rv:45.0) "
        browser += "Gecko/20100101 Firefox/45.0"
        request.add_header('User-Agent', browser)
        response = urllib.request.urlopen(request)
        dictionary = json.loads(response.read().decode('utf8'))
        return dictionary["sentences"][0]['trans']	
    except SocketError as e:
        if e.errno != errno.ECONNRESET:
            raise # Not error we are looking for
        pass # Handle error here.


def translate_vocab(vocab, lg_from, lg_to):
    """Pickle dictionary of translated vocab"""
    # Load the dictionary if it exists
    try:
        d = pickle_rw((lg_from + '_' + lg_to, 0), write=False)
    except:
        d = {}

    counter = 0
    # For each word in vocab
    for v in vocab:
        # If the word isn't already in the dictionary
        if v not in d:
            t = translate_text(v, lg_from, lg_to)
            d[v] = t

        counter += 1
        if counter % 100 == 0:
            print(counter)
        if counter % 10000 == 0:
            # Pickle dictionary
            pickle_rw((lg_from + '_' + lg_to, d))
    pickle_rw((lg_from + '_' + lg_to, d))
    print("Complete")
    return


if __name__ == '__main__':
    # Use fasttext embedding for translation vocabularies
    embedding = 'fasttext'

    # Load languages and lgs lists for embedding
    languages, lgs = embedding_languages_lgs(embedding)

    # List of all (lg_from, lg_to) combinations
    translations = [(a, b) for a in lgs for b in lgs if a != b]
    translations = [('de','en')]  # Temporary override

    # For each combination of lgs
    for translation in translations:
        lg_from, lg_to = translation

        # Load vocab for lg_from
        vocab = pickle_rw((lg_from + '_' + embedding + '_vocab', 0),
                          write=False)

        # Create/Update and Pickle Translation Dictionary
        translate_vocab(vocab, lg_from, lg_to)
