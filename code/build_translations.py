from urllib.request import urlopen
from urllib.parse import urlencode
import urllib
from google.cloud import translate
import json
from gensim_download import pickle_rw
from vocab_vectors import embedding_languages_lgs
import sys
from nltk.stem import WordNetLemmatizer
import os

# Enter Path for Credentials File
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="../data/key/client_secrets.json"


lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize("run",'v')

      
def translate_text(source_text, lg_from, lg_to):
    """
    lg_from -- The language code for the source text argument
    lg_to -- The language code for the translated language
    source_text -- Text to be translated
    """
    # Instantiates a client
    translate_client = translate.Client()

    # The text to translate
    text = source_text
    # The target language
    target = lg_to

    # Translates some text into Russian
    translation = translate_client.translate(
        text,
        target_language=target)

    #print(u'Text: {}'.format(text))
    #print(u'Translation: {}'.format(translation['translatedText']))
    # [END translate_quickstart]
    return translation['translatedText']

    

    
def lemmatize_word(word):
    """Lemmatize the word before putting it to Dictionary"""
    return lemmatizer.lemmatize(word,'v')
    
    
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
        if counter % 1000 == 0:
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

    lang_from = sys.argv[1]
    lang_to = sys.argv[2]
    
    # List of all (lg_from, lg_to) combinations
    translations = [(a, b) for a in lgs for b in lgs if a != b]
    translations = [(lang_from,lang_to)]

    # For each combination of lgs
    for translation in translations:
        lg_from, lg_to = translation

        # Load vocab for lg_from
        vocab = pickle_rw((lg_from + '_' + embedding + '_vocab', 0),
                          write=False)

        # Create/Update and Pickle Translation Dictionary
        translate_vocab(vocab, lg_from, lg_to)
