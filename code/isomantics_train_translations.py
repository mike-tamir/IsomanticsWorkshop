import sys
import json
from ismtools import create_model_parameters_json, parse_arguments, pickle_rw, make_dict, path_exists, vocab_train_test, vectors_train_test, translation_matrix, translation_results, T_norm_EDA, convert_mat_to_df, path_exists, make_dir

if __name__ == '__main__':
    # Manually set list of translations (embedding, lg1, lg2)
    
    parser = parse_arguments()
    #parser.add_argument("--a", help="compute accuracy along with other statistics")
    parser.add_argument("--exp_name", help="specify name for the experiment folder")
    parser.add_argument("--reg_name", help="specify regularizer for the model")
    parser.add_argument("--loss_func", help="specify loss function for the model")
    parser.add_argument("--dim", help="specify no. of dimensions for the model")
    parser.add_argument("--l2_lambda", help="specify lambda value for Frobenius L2 norm in the regularizer of the model")
    parser.add_argument("--normality_lambda", help="specify lambda value for normalizer of the model")
    parser.add_argument("--orthonormality_lambda", help="specify lambda value for normalizer of the model")

    args = parser.parse_args()
    
    experiment = args.exp_name
    regularizer = args.reg_name
    loss_function = args.loss_func
    dimensions = int(args.dim)
    l2_lambda = float(args.l2_lambda)
    normality_lambda = float(args.normality_lambda)
    orthonormality_lambda = float(args.orthonormality_lambda)
    
    directory = ('../data/'+experiment+'/T_matrices/')
    
    out_json_path = "../data/"+experiment+"/model_parameters.json"
    
    if not path_exists(directory):
        make_dir(directory) 
    
    svds = ['s','s1']
    #languages = ['en','ru','de','es','fr','it', 'zh-CN']
    languages = ['en','la','ru','de','es','fr','it','hi','bn','zh-CN']
 
    
    translations=[]
    
    for lang1 in languages:
        for lang2 in languages:
            translations.append(('fasttext_top',lang1, lang2))
    
    svd = {}
    svd1 = {}
    
    # Create dict for storing model parameters.
        
    out_dict = create_model_parameters_json(experiment,
                                     regularizer,
                                     loss_function,
                                     dimensions,
                                     l2_lambda,
                                     normality_lambda,
                                     orthonormality_lambda)

    # Write dict out as json

    with open(out_json_path,"w") as jpath:
        json.dump(out_dict, jpath)

    jpath.close()
    
    
    print('Creating Translation Matrices for....:\n')
    
    acc_dict = {}
    
    
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
        
        #acc_dict[lg1+'_'+lg2] = {}

        print(lg1+'->'+lg2+'\n')

        # Train/Test Vocab/Vectors
        vocab_train, vocab_test = vocab_train_test(embedding, lg1, lg2, lg1_vocab)
        X_train, X_test, y_train, y_test = vectors_train_test(vocab_train,
                                                              vocab_test,lg1_dict,lg2_dict)
 
        
        print(y_train.shape)
        # Fit tranlation matrix to training data
        model, history, T, tf,I, M, fro = translation_matrix(X_train, 
                                                             y_train, 
                                                             dimensions,
                                                             loss_function, 
                                                             regularizer,
                                                             l2_lambda,                        
                                                             orthonormality_lambda)
        
        results_df = translation_results(X_test, y_test, vocab_test, T,
                                         lg2_vectors, lg2_vocab)
       
        test_accuracy = T_norm_EDA(results_df)
        
        acc = test_accuracy
        
        print('Accuracy:'+str(acc)+'\n')
        
        T = convert_mat_to_df(T)
        
        T.to_csv(directory+'/{}_{}_T.csv'.format(lg1,lg2), header=False, index= False)
        
        #acc_dict = {}
        
        acc_dict[lg1+'_'+lg2] = acc
    
    out_acc_json_path = "../data/"+experiment+"/acc_dict.json"
    with open(out_acc_json_path,"w") as jpath:
            json.dump(acc_dict, jpath)

    jpath.close()
        
   