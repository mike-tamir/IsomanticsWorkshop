from gensim_download import pickle_rw


if __name__ == "__main__":
    # Pickle lgs and languages lists
    fasttext_lgs = ['de', 'en', 'ru']
    fasttext_languages = ['German', 'English', 'Russian']
    pickle_rw(('fasttext_lgs', fasttext_lgs))
    pickle_rw(('fasttext_languages', fasttext_languages))
