from gensim_download import pickle_rw


if __name__ == "__main__":
    # Pickle lgs and languages lists
    fasttext_lgs = ['de', 'en', 'ru','zh_yue','es']
    fasttext_languages = ['German', 'English', 'Russian','Chinese','Spanish']
    pickle_rw(('fasttext_lgs', fasttext_lgs))
    pickle_rw(('fasttext_languages', fasttext_languages))
