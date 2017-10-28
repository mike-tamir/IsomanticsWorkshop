from ismtools import parse_arguments
from ismtools import pickle_rw
from ismtools import make_dict
from ismtools import path_exists
from ismtools import vocab_train_test
from ismtools import vectors_train_test
from ismtools import translation_matrix
from ismtools import convert_mat_to_df
from ismtools import path_exists
from ismtools import make_dir

if __name__ == '__main__':
    # Manually set list of translations (embedding, lg1, lg2)
    
    parser = parse_arguments()
    #parser.add_argument("--a", help="compute accuracy along with other statistics")
    parser.add_argument("--e", help="specify name for the experiment folder")
    parser.add_argument("--m", help="specify model for the experiment")
    args = parser.parse_args()
    
    mode = args.m
    
    directory = ('../Spectral_Decomposition_Experiments/'+args.e+'/T/')
    
    if not path_exists(directory):
        make_dir(directory) 
    
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
        # Vocab/Vectors/Dicts
        lg1_vocab, lg1_vectors, lg2_vocab, lg2_vectors = \
            pickle_rw((lg1 + '_' + embedding.split('_')[0] + '_vocab', 0),
                      (lg1 + '_' + embedding.split('_')[0] + '_vectors', 0),
                      (lg2 + '_' + embedding.split('_')[0] + '_vocab', 0),
                      (lg2 + '_' + embedding.split('_')[0] + '_vectors', 0),
                      write=False)
        lg1_dict = make_dict(lg1_vocab, lg1_vectors)
        lg2_dict = make_dict(lg2_vocab, lg2_vectors)

        print(lg1+'->'+lg2+'\n')

        # Train/Test Vocab/Vectors
        vocab_train, vocab_test = vocab_train_test(embedding, lg1, lg2, lg1_vocab)
        X_train, X_test, y_train, y_test = vectors_train_test(vocab_train,
                                                              vocab_test,lg1_dict,lg2_dict)
 
        
        # Fit tranlation matrix to training data
        model, history, T, tf,I, M, fro = translation_matrix(X_train, y_train, mode)
        
        #covariance_T = T.dot(T.T) 
        
        T = convert_mat_to_df(T)
        
        T.to_csv(directory+'/{}_{}_T.csv'.format(lg1,lg2), header=False, index= False)
        
      