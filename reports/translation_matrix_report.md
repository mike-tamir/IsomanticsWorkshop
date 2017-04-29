# Translation Matrix Results  
## En to Ru Fasttext_Random  
- En Vocabulary Size = 1,259,685  
- En Embedding Length = 300  
- Ru Vocabulary Size = 944,211  
- Ru Embedding Length = 300  
- Train Size = 5,000  
- Test Size = 1,500  
- <b>Test Accuracy = 3.9%</b>  

#### Test L2 Norms  
- X_norm: L2 norms for En test vectors  
- y_norm: L2 norms for Ru test vectors  
- yhat_norm: L2 norms for X.dot(T) test vectors (T = translation matrix)  
- yhat_neighbor norm: L2 norms for nearest neighborto X.dot(T) in y test vectors  
![](../images/en_ru_fasttext_random_T_norm.png)  

#### Translation Matrix Isotropy  
- Isotropy = 32.3%  
![](../images/en_ru_fasttext_random_T_isotropy.png)  

## En to Ru Fasttext_Top  
- En Vocabulary Size = 1,259,685  
- En Embedding Length = 300  
- Ru Vocabulary Size = 944,211  
- Ru Embedding Length = 300  
- Train Size = 5,000  
- Test Size = 1,500  
- <b>Test Accuracy = 46.3%</b>  

#### Test L2 Norms  
- X_norm: L2 norms for En test vectors  
- y_norm: L2 norms for Ru test vectors  
- yhat_norm: L2 norms for X.dot(T) test vectors (T = translation matrix)  
- yhat_neighbor norm: L2 norms for nearest neighborto X.dot(T) in y test vectors  
![](../images/en_ru_fasttext_top_T_norm.png)  

#### Translation Matrix Isotropy  
- Isotropy = 38.2%  
![](../images/en_ru_fasttext_top_T_isotropy.png)  

## En to De Fasttext_Random  
- En Vocabulary Size = 1,259,685  
- En Embedding Length = 300  
- De Vocabulary Size = 1,137,616  
- De Embedding Length = 300  
- Train Size = 5,000  
- Test Size = 1,500  
- <b>Test Accuracy = 21.9%</b>  

#### Test L2 Norms  
- X_norm: L2 norms for En test vectors  
- y_norm: L2 norms for De test vectors  
- yhat_norm: L2 norms for X.dot(T) test vectors (T = translation matrix)  
- yhat_neighbor norm: L2 norms for nearest neighborto X.dot(T) in y test vectors  
![](../images/en_de_fasttext_random_T_norm.png)  

#### Translation Matrix Isotropy  
- Isotropy = 35.6%  
![](../images/en_de_fasttext_random_T_isotropy.png)  

## En to De Fasttext_Top  
- En Vocabulary Size = 1,259,685  
- En Embedding Length = 300  
- De Vocabulary Size = 1,137,616  
- De Embedding Length = 300  
- Train Size = 5,000  
- Test Size = 1,500  
- <b>Test Accuracy = 63.6%</b>  

#### Test L2 Norms  
- X_norm: L2 norms for En test vectors  
- y_norm: L2 norms for De test vectors  
- yhat_norm: L2 norms for X.dot(T) test vectors (T = translation matrix)  
- yhat_neighbor norm: L2 norms for nearest neighborto X.dot(T) in y test vectors  
![](../images/en_de_fasttext_top_T_norm.png)  

#### Translation Matrix Isotropy  
- Isotropy = 43.4%  
![](../images/en_de_fasttext_top_T_isotropy.png)  

## En to It Zeroshot  
- En Vocabulary Size = 200,000  
- En Embedding Length = 300  
- It Vocabulary Size = 200,000  
- It Embedding Length = 300  
- Train Size = 5,000  
- Test Size = 1,869  
- <b>Test Accuracy = 27.9%</b>  

#### Test L2 Norms  
- X_norm: L2 norms for En test vectors  
- y_norm: L2 norms for It test vectors  
- yhat_norm: L2 norms for X.dot(T) test vectors (T = translation matrix)  
- yhat_neighbor norm: L2 norms for nearest neighborto X.dot(T) in y test vectors  
![](../images/en_it_zeroshot_T_norm.png)  

#### Translation Matrix Isotropy  
- Isotropy = 46.6%  
![](../images/en_it_zeroshot_T_isotropy.png)  

