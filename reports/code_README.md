# Repository Overview

## Landing Page
Please find the following 2 important documents directly on the repository landing page:
- Practical observations in applying neural word embeddings to machine translation
  - This paper describes this project in detail
- Presentation
  - These slides were used in a presentation of this project

## Code Folder
This folder contains the following Python scripts:
- gauth.py
  - Updates and saves google drive credentials
- gensim_download.py
  - Contains a list of google file ids for each of the embeddings
  - Retrieves each embedding file from google drive and saves locally to the data/gensim subfolder
  - Contains a list of gensim_languages and gensim_lgs.  Pickles these lists locally to the pickle subfolder
- polyglot_download.py
  - Retrieves each polyglot embedding file from the web address or google drive file and saves locally to the data/polyglot subfolder
  - Pickles polyglot_lgs list locally to the pickle subfolder
- fasttext_download.py
  - Contains lists of fasttext_languages and fasttext_lgs.  Pickles these lists locally to the pickle subfolder
  - The language embeddings are large and are manually retrieved one by one (not with this script)
- zeroshot_download.py
  - Contains lists of zeroshot_languages and zeroshot_lgs.  Pickles these lists locally to the pickle subfolder
  - The language embeddings are manually retrieved from the web-site (not with this script)
- vocab_vectors.py
  - For each type of embedding and each language:
  - Loads the embedding from file
  - Creates the vocab list and the vectors array
  - Pickles the vocab and vectors locally to the pickle subfolder
- eda.py
  - Performs exploratory data analysis on the embedding vectors
  - Plots and saves images locally to the images subfolder
  - Create reports and saves locally to the reports subfolder
  - Saves CSV of results locally to the data subfolder
- build_translations.py
  - Creates bilingual translation dictionary between 2 languages using google translate
  - Note: It takes several days for 1 language-to-language mapping due to google translate throttle
- build_translations_spark.py
  - Parallelizes the language-to-language translation, but still impacted by throttle and costs much more if using with Elastic Map Reduce on Amazon Web Services
- translate.py
  - Creates the training and test data
  - Trains the translation matrix T
  - Calculates the test performance of the translation
  - Creates the results report and saves locally to the reports subfolder

## Images Folder
This folder stores all the images used in the reports

## Reports Folder
This folder contains the following reports:
 - Exploratory data analysis output for each of the 4 embedding sources
 - The translation matrix output containing plots and details for each of the 5 experiments
