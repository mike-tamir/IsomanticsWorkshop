import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import anderson, kstest, norm, shapiro
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from gensim_download import pickle_rw


def load_embedding(filepath):
    """Load embedding from file"""
    with open(filepath) as f:
        f_string = f.read()
    f_list = f_string.split(']')[:-1]
    vocab = [_.split('[')[0].split('\t')[1] for _ in f_list]
    vectors_text = [_.split('[')[1].replace('\n', '').split(' ')
                    for _ in f_list]
    vectors_text = [[a for a in b if a != ''] for b in vectors_text]
    vectors = np.asarray(vectors_text, dtype=np.float64)
    return vocab, vectors


def norm_EDA(vectors, path, lg):
    """EDA on the norm of the vectors
    vectors = word embedding vectors
    path = location of images folder
    lg = language"""
    # L2 norm of vectors, then normalize distribution of L2 norms
    vectors_norm = np.linalg.norm(vectors, axis=1)
    vectors_norm_normalized = (vectors_norm - vectors_norm.mean()) \
        / vectors_norm.std()

    # Histogram compared to normal dist
    plt.figure(figsize=(10, 6))
    plt.hist(vectors_norm_normalized, bins=100, normed=True)
    x = np.linspace(-3, 3, 100)
    plt.plot(x, norm.pdf(x, 0, 1), color='r', linewidth=3)
    plt.savefig(path + lg + '_gensim_norm.png')
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


def pca_EDA(vectors, path, lg):
    """PCA on vectors
    vectors = word embedding vectors
    path = location of images folder
    lg = language"""
    vectors_ss = StandardScaler().fit_transform(vectors)
    pca = PCA().fit(vectors_ss)

    plt.figure(figsize=(10, 6))
    n = pca.n_components_
    plt.plot(range(1, n + 1), np.cumsum(pca.explained_variance_ratio_))
    plt.plot(range(1, n + 1), np.asarray(range(1, n + 1)) / 300)
    plt.xlabel('Number of Eigenvectors')
    plt.ylabel('Explained Variance')
    plt.ylim((0, 1))
    plt.savefig(path + lg + '_gensim_isotropy.png')
    plt.close('all')

    isotropy = (1 - sum(np.cumsum(pca.explained_variance_ratio_) * 1 / n)) / .5
    return isotropy


def csv_EDA(path):
    """Send norm and pca EDA results to csv"""
    EDA_cols = ['AD_test_stat', 'AD_crit_val_1', 'AD_result',
                'KS_p_val', 'KS_result', 'SH_p_val', 'SH_result']
    EDA_ind = gensim_lgs[0: len(pca_EDA_results)]
    EDA_df = pd.DataFrame(norm_EDA_results, columns=EDA_cols, index=EDA_ind)\
        .join(pd.DataFrame(pca_EDA_results, columns=['isotropy'],
                           index=EDA_ind))
    EDA_df.index.name = 'lg'
    EDA_df.to_csv(path + 'gensim_eda.csv')
    return


def report_EDA(path):
    """Create and save markdown report of EDA results"""
    md = '# Gensim EDA  \n'
    for i in range(len(norm_EDA_results)):
        lg = gensim_lgs[i]
        md += '## ' + gensim_languages[i] + '  \n'

        md += '#### Embedding L2 Norms  \n'
        md += '![](../images/' + lg + '_gensim_norm.png)  \n'
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
        md += '![](../images/' + lg + '_gensim_isotropy.png)  \n'
        md += '- Isotropy: ' + str(pca_EDA_results[i].round(2)) + '  \n\n'

    with open(path + 'gensim_EDA.md', mode='w') as f:
        f.write(md)
    return


if __name__ == "__main__":
    # Load lists from pickle
    gensim_languages, gensim_lgs = pickle_rw('../pickle/',
                                             ('gensim_languages', 0),
                                             ('gensim_lgs', 0), write=False)

    # Results lists
    norm_EDA_results = []
    pca_EDA_results = []

    # For each language
    for lg in gensim_lgs[0:2]:
        # Load embedding from tsv file
        vocab, vectors = load_embedding('../data/gensim/' + lg + '/' +
                                        lg + '.tsv')

        # EDA on the norm of the embedding vectors
        norm_EDA_results.append(norm_EDA(vectors, '../images/', lg))

        # PCA and isotropy of the embedding vectors
        pca_EDA_results.append(pca_EDA(vectors, '../images/', lg))

        # Pickle the vocab and vector objects
        pickle_rw('../pickle/', (lg + '_gensim_vocab', vocab),
                  (lg + '_gensim_vectors', vectors))

    # Save norm and pca EDA results
    csv_EDA('../data/')

    # Create markdown report
    report_EDA('../reports/')
