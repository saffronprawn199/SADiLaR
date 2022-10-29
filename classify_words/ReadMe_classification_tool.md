# Classify pre-trained word embeddings using logistic classification

for `classify_word_vectors.py`

### Dependencies:
1. gensim
2. pandas
3. NLTK
4. sklearn

**Note:** These dependencies can be installed using pip python package manager.

#### Example of how program is used:

Note: Make sure the input files are in a **CSV format** with column named '**Sentence**' and '**Tag**' with a sentence string and tag in each row. Example of input file is given in `classification_dataset_example.csv`

## Example of how to use:


#### This wil classify the pre-trained word vectors and a baseline classifier:
* ` python classify_word_vectors.py --inFilePath English_tagged.csv --inModelPath ./word2vec_models/word2vec_train_epochs_trained_10_English_CBOW_embedding_size_300 --outFileName word2vec_train_epochs_trained_10_English_CBOW_embedding_size_300_classification.txt --BL 1 -tags news,sport,weather,advertisement,traffic`

#### This wil classify only the pre-trained word vectors:

* `python classify_word_vectors.py --inFilePath English_tagged.csv --inModelPath ./word2vec_models/word2vec_train_epochs_trained_10_English_CBOW_embedding_size_300 --outFileName word2vec_train_epochs_trained_10_English_CBOW_embedding_size_300_classification.txt -tags news,sport,weather,advertisement,traffic
`

#### Output files:

Results of classified models can be found in the following directory:
* `./Classified_models`
