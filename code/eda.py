import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from scipy.stats import anderson, kstest, norm, shapiro
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from gensim_download import pickle_rw


def lg_load(embedding):
    """Load languages and lgs lists for embedding"""
    if embedding == 'gensim':
        languages, lgs = pickle_rw(('gensim_languages', 0),
                                   ('gensim_lgs', 0), write=False)
    elif embedding == 'polyglot':
        # polyglot doesn't have languages, so use lgs for both
        languages, lgs = pickle_rw(('polyglot_lgs', 0),
                                   ('polyglot_lgs', 0), write=False)
    else:
        pass
    return languages, lgs


def gensim_load(lg):
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


def polyglot_load(lg):
    with open('../data/polyglot/' + lg + '.pkl', 'rb') as f:
        vocab, vectors = pickle.load(f, encoding='bytes')
    return vocab, vectors


def vocab_vectors_load(lg, embedding):
    """Load vocab and vectors for lg/embedding"""
    if embedding == 'gensim':
        vocab, vectors = gensim_load(lg)
    elif embedding == 'polyglot':
        vocab, vectors = polyglot_load(lg)
    else:
        pass
    return vocab, vectors


def norm_EDA(vectors, lg, embedding):
    """EDA on the norm of the vectors
    vectors = word embedding vectors
    lg = language
    embedding = gensim, polyglot, etc."""
    # L2 norm of vectors, then normalize distribution of L2 norms
    vectors_norm = np.linalg.norm(vectors, axis=1)
    vectors_norm_normalized = (vectors_norm - vectors_norm.mean()) \
        / vectors_norm.std()

    # Histogram compared to normal dist
    plt.figure(figsize=(10, 6))
    plt.xlim((-3, 5))
    plt.hist(vectors_norm_normalized, bins=100, normed=True)
    x = np.linspace(-3, 3, 100)
    plt.plot(x, norm.pdf(x, 0, 1), color='r', linewidth=3)
    plt.savefig('../images/' + lg + '_' + embedding + '_norm.png')
    plt.close('all')

    # Anderson Darling
    # If test stat is greater than crit val, reject ho=normal
    # crit_val_1 is critical value for p-value of 1%
    ad = anderson(vectors_norm_normalized, 'norm')
    ad_test_stat = ad.statistic
    ad_crit_val_1 = ad.critical_values[-1]
    ad_result = 'Reject' if ad_test_stat > ad_crit_val_1 else 'Fail to Reject'

    # Kolmogorov-Smirnov
    ks_p_val = kstest(vectors_norm_normalized, 'norm')[1]
    ks_result = 'Reject' if ks_p_val < .01 else 'Fail to Reject'

    # Shapiro
    sh_p_val = shapiro(vectors_norm_normalized)[1]
    sh_result = 'Reject' if sh_p_val < .01 else 'Fail to Reject'

    result = (ad_test_stat, ad_crit_val_1, ad_result,
              ks_p_val, ks_result,
              sh_p_val, sh_result)
    return result


def pca_EDA(vectors, lg, embedding):
    """PCA on vectors
    vectors = word embedding vectors
    lg = language
    embedding = gensim, polyglot, etc."""
    vectors_ss = StandardScaler().fit_transform(vectors)
    pca = PCA().fit(vectors_ss)
    n = pca.n_components_

    plt.figure(figsize=(10, 6))
    plt.xlim((0, n))
    plt.ylim((0, 1))
    plt.plot(range(n + 1), [0] + np.cumsum(pca.explained_variance_ratio_).
             tolist())
    plt.plot(range(n + 1), np.asarray(range(n + 1)) / n)
    plt.xlabel('Number of Eigenvectors')
    plt.ylabel('Explained Variance')
    plt.savefig('../images/' + lg + '_' + embedding + '_isotropy.png')
    plt.close('all')

    vocabulary_size, vector_length = vectors.shape
    isotropy = (1 - sum(np.cumsum(pca.explained_variance_ratio_) * 1 / n)) / .5
    return (vocabulary_size, vector_length, isotropy)


def csv_EDA(lgs, embedding):
    """Send norm and pca EDA results to csv
    lgs = list of languages
    embedding = gensim, polyglot, etc."""
    norm_EDA_cols = ['AD_test_stat', 'AD_crit_val_1', 'AD_result',
                     'KS_p_val', 'KS_result', 'SH_p_val', 'SH_result']
    pca_EDA_cols = ['vocabulary_size', 'vector_length', 'isotropy']
    EDA_ind = lgs[0: len(pca_EDA_results)]
    EDA_df = pd.DataFrame(norm_EDA_results, columns=norm_EDA_cols,
                          index=EDA_ind).\
        join(pd.DataFrame(pca_EDA_results, columns=pca_EDA_cols,
                          index=EDA_ind))
    EDA_df.index.name = 'lg'
    EDA_df.to_csv('../data/' + embedding + '_eda.csv')
    return


def report_EDA(lgs, languages, embedding):
    """Create and save markdown report of EDA results
    lgs = list of language abbreviations
    languages = list of languages
    embedding = gensim, polyglot, etc."""
    md = '# ' + embedding.title() + ' EDA  \n'
    for i in range(len(norm_EDA_results)):
        lg = lgs[i]
        md += '## ' + languages[i] + '  \n'
        md += '- Vocabulary Size = ' + '{:,.0f}'.format(
            pca_EDA_results[i][0]) + '  \n'
        md += '- Embedding Length = ' + str(pca_EDA_results[i][1]) + '  \n'

        md += '#### Embedding L2 Norms  \n'
        md += '![](../images/' + lg + '_' + embedding + '_norm.png)  \n'
        md += '- Anderson-Darling Test Statistic: ' +\
            str(norm_EDA_results[i][0].round(2)) + '  \n'
        md += '- Anderson-Darling Critical Value (1%): ' +\
            str(norm_EDA_results[i][1].round(2)) + '  \n'
        md += '- Anderson-Darling Test Result: ' +\
            norm_EDA_results[i][2] + '  \n\n'
        md += '- Kolmogorov-Smirnov p-value: ' +\
            str(norm_EDA_results[i][3].round(2)) + '  \n'
        md += '- Kolmogorov-Smirnov Test Result: ' +\
            norm_EDA_results[i][4] + '  \n\n'
        md += '- Shapiro-Wilk p-value: ' +\
            str(round(norm_EDA_results[i][5], 2)) + '  \n'
        md += '- Shapiro-Wilk Test Result: ' +\
            norm_EDA_results[i][6] + '  \n\n'

        md += '#### Embedding Isotropy  \n'
        md += '![](../images/' + lg + '_' + embedding + '_isotropy.png)  \n'
        md += '- Isotropy: ' + str(pca_EDA_results[i][2].round(2)) + '  \n\n'

    with open('../reports/' + embedding + '_EDA.md', mode='w') as f:
        f.write(md)
    return


if __name__ == "__main__":
    # For each embedding
    embeddings = ['gensim', 'polyglot']
    for embedding in embeddings:
        # Load languages and lgs lists for embedding
        languages, lgs = lg_load(embedding)

        # Results lists
        norm_EDA_results = []
        pca_EDA_results = []

        # For each language
        for lg in lgs:
            # Load vocab and vectors for lg/embedding
            vocab, vectors = vocab_vectors_load(lg, embedding)

            # EDA on the norm of the embedding vectors
            norm_EDA_results.append(norm_EDA(vectors, lg, embedding))

            # PCA and isotropy of the embedding vectors
            pca_EDA_results.append(pca_EDA(vectors, lg, embedding))

            # Pickle the vocab and vector objects
            pickle_rw((lg + '_' + embedding + '_vocab', vocab),
                      (lg + '_' + embedding + '_vectors', vectors))

        # Save norm and pca EDA results
        csv_EDA(lgs, embedding)

        # Create markdown report
        report_EDA(lgs, languages, embedding)
