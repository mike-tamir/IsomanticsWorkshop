from gensim_download import pickle_rw


if __name__ == "__main__":
    # Pickle lgs and languages lists
    zeroshot_lgs = ['en', 'it']
    zeroshot_languages = ['English', 'Italian']
    pickle_rw(('zeroshot_lgs', zeroshot_lgs))
    pickle_rw(('zeroshot_languages', zeroshot_languages))
