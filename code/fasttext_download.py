from gensim_download import pickle_rw


if __name__ == "__main__":
    # Pickle lgs and languages lists
    #fasttext_lgs = ['de', 'en', 'ru','zh_yue','es','fr','it','ja']
    fasttext_lgs = ['fr','it','ja']
    fasttext_languages = ['French','Italian','Japanese']
    pickle_rw(('fasttext_lgs', fasttext_lgs))
    pickle_rw(('fasttext_languages', fasttext_languages))
