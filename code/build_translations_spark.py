from pyspark import SparkContext
from build_translations import translate_text
from gensim_download import pickle_rw


def translate_word_spark(word, lg_from, lg_to):
    """Dictionary of translated vocab"""
    t = translate_text(word, lg_from, lg_to)
    return (word, t)


if __name__ == '__main__':
    # Instantiate Spark Context
    sc = SparkContext()
    sc.setLogLevel("ERROR")

    # Set embedding and translation languages
    embedding = 'fasttext'
    translation = ('ru', 'en')

    # Load vocab and dictionary
    lg_from, lg_to = translation
    vocab = pickle_rw((lg_from + '_' + embedding + '_vocab', 0),
                      write=False)
    d = pickle_rw((lg_from + '_' + lg_to, 0), write=False)
    vocab_new = list(set(vocab).difference(set(d.keys())))

    counter = 0
    while len(vocab_new) != 0:
        # Parallelize and translate
        vocabRDD = sc.parallelize(vocab_new[:10000])
        translateRDD = vocabRDD.map(lambda x: translate_word_spark(
            x, lg_from, lg_to))
        translated = translateRDD.collect()

        # Add Translated to dictionary
        for k, v in translated:
            d[k] = v

        # Pickle
        pickle_rw((lg_from + '_' + lg_to, d))

        # Print Counter and update vocab_new
        counter += 10000
        print(counter)
        vocab_new = list(set(vocab).difference(set(d.keys())))
