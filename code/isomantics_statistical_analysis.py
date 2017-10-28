from ismtools import parse_arguments
from ismtools import pickle_rw
from ismtools import make_dict
from ismtools import make_df
from ismtools import stat_calc
from ismtools import SVD
from ismtools import log
from ismtools import calc_fro
from ismtools import convert_df_to_mat
from ismtools import translation_results
from ismtools import translation_matrix
from ismtools import load_csv


if __name__ == '__main__':
    # Manually set list of translations (embedding, lg1, lg2)
    
    
    parser = parse_arguments()
    parser.add_argument("--a", help="compute accuracy along with other statistics")
    args = parser.parse_args()
    
    if args.a:
        stats = ['min','max','mean','median','std','fro','acc']
    else:
        stats = ['min','max','mean','median','std','fro']
    
    svds = ['s','s1']
    languages = ['en','ru','de','es','fr','it', 'zh-CN']
    
    
    translations=[]
    
    for lang1 in languages:
        for lang2 in languages:
            translations.append(('fasttext_top',lang1, lang2))
    
    svd = {}
    svd1 = {}
    
    print('Creating Translation Matrices for....:\n')
    
    for translation in translations:
        embedding, lg1, lg2 = translation
        
        T = load_csv('../Spectral_Decomposition_Experiments/test_l3_l2_T/T/{}_{}_T.csv'.format(lg1,lg2))   
        
        #T = convert_df_to_mat(T)
        
        fro = calc_fro(T)
        
        if args.a:
            results_df = translation_results(X_test, y_test, vocab_test, T,
                                         lg2_vectors, lg2_vocab)
            acc = T_norm_EDA(results_df)
        else:
            acc = 0.0
            
        U,s,Vh = SVD(T)
        
        s1 = log(s)
    
        for stat in stats:
            svd[stat,translation[1],translation[2]] = stat_calc(stat, s, fro, acc)
            svd1[stat,translation[1],translation[2]] = stat_calc(stat, s1, fro, acc)
            
        
    #Exporting DataFrames for SVD Heatmaps
    
    print('Exporting DataFrames for SVD Heatmaps as .CSV files')
    
    s_df = make_df(languages,languages)
    s1_df = make_df(languages,languages)
    

    for stat in stats:
        for lang1 in languages:
            for lang2 in languages:
                s_df.set_value(lang1,lang2,svd[stat,lang1,lang2])
                s1_df.set_value(lang1,lang2,svd1[stat,lang1,lang2])

        s_df.to_csv('../HeatmapData/T/s_{}.csv'.format(stat),columns = languages, index= False)
        s1_df.to_csv('../HeatmapData/T/s1_{}.csv'.format(stat),columns = languages, index = False)
