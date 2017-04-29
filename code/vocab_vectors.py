import numpy as np
import pickle
from gensim_download import pickle_rw


def embedding_languages_lgs(embedding):
    """Load languages and lgs lists for embedding"""
    if embedding == 'gensim':
        languages, lgs = pickle_rw(('gensim_languages', 0),
                                   ('gensim_lgs', 0), write=False)
    elif embedding == 'polyglot':
        # polyglot doesn't have languages, so use lgs for both
        languages, lgs = pickle_rw(('polyglot_lgs', 0),
                                   ('polyglot_lgs', 0), write=False)
    elif embedding == 'fasttext':
        languages, lgs = pickle_rw(('fasttext_languages', 0),
                                   ('fasttext_lgs', 0), write=False)
    elif embedding == 'zeroshot':
        languages, lgs = pickle_rw(('zeroshot_languages', 0),
                                   ('zeroshot_lgs', 0), write=False)
    else:
        pass
    return languages, lgs


def gensim_vocab_vectors(lg):
    """Load embedding from file"""
    with open('../data/gensim/' + lg + '/' + lg + '.tsv') as f:
        f_string = f.read()
    f_list = f_string.split(']')[:-1]
    vocab = [_.split('[')[0].split('\t')[1] for _ in f_list]
    vectors_text = [_.split('[')[1].replace('\n', '').split(' ')
                    for _ in f_list]
    vectors_text = [[a for a in b if a != ''] for b in vectors_text]
    vectors = np.asarray(vectors_text, dtype=np.float32)
    return vocab, vectors


def polyglot_vocab_vectors(lg):
    """Load embedding from file"""
    with open('../data/polyglot/' + lg + '.pkl', 'rb') as f:
        vocab, vectors = pickle.load(f, encoding='bytes')
    return vocab, vectors


def fasttext_vocab_vectors(lg):
    """Load embedding from file"""
    firstline = True
    vocab = []
    vectors = []
    with open('../data/fasttext/wiki.' + lg + '.vec') as f:
        while f.readline():
            if firstline:
                firstline = False
            else:
                f_line = f.readline()
                if f_line != '':
                    f_list = f_line.split(' ')
                    word, vec = f_list[0], np.asarray(f_list[1:301],
                                                      dtype=np.float32)
                    vocab.append(word)
                    vectors.append(vec)
    vectors = np.asarray(vectors)
    return vocab, vectors


def zeroshot_vocab_vectors(lg):
    """Load embedding from file"""
    with open('../data/zeroshot/transmat/data/' + lg +
              '.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt') as f:
        f_string = f.read()
    f_list = f_string.split('\n')[1:-1]
    vocab = [_.split(' ')[0] for _ in f_list]
    vectors_text = [_.split(' ')[1:] for _ in f_list]
    vectors = np.asarray(vectors_text, dtype=np.float32)
    return vocab, vectors


def pick_vocab_vectors(embedding, lg):
    """Load vocab and vectors for lg/embedding"""
    if embedding == 'gensim':
        vocab, vectors = gensim_vocab_vectors(lg)
    elif embedding == 'polyglot':
        vocab, vectors = polyglot_vocab_vectors(lg)
    elif embedding == 'fasttext':
        vocab, vectors = fasttext_vocab_vectors(lg)
    elif embedding == 'zeroshot':
        vocab, vectors = zeroshot_vocab_vectors(lg)
    else:
        pass
    return vocab, vectors


if __name__ == "__main__":
    # List embeddings
    embeddings = ['gensim', 'polyglot', 'fasttext', 'zeroshot']

    # For each embedding
    for embedding in embeddings:
        # Load languages and lgs lists for embedding
        languages, lgs = embedding_languages_lgs(embedding)

        # For each language
        for lg in lgs:
            # Load vocab and vectors for embedding/lg
            vocab, vectors = pick_vocab_vectors(embedding, lg)

            # Pickle the vocab and vector objects
            pickle_rw((lg + '_' + embedding + '_vocab', vocab),
                      (lg + '_' + embedding + '_vectors', vectors))
