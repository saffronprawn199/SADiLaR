Train word embeddings using fastText and word2vec
for train_word_vectors.py

Dependiencies:
1)gensim
2)pandas

These dependiecies can be installed using pip python package manager.

Example of how program is used:

To findout what kind of arguments the program takes type:
- python train_word_vectors.py --help

Note: Make sure the input files are in a CSV format with column named 'Sentences' with a sentence string in each row.

Example of how to use:


This wil train word vectors for both fastText and word2vec using CBOW and SG
- python train_word_vectors.py --inFilePath English_all.csv --outFileName English --CBOW 1 --SG 1 --typeEmbedding word2vec fastText

This wil train word vectors for fastText using SG
- python train_word_vectors.py --inFilePath English_all.csv --outFileName English --SG 1 --typeEmbedding fastText
or
- python train_word_vectors.py --inFilePath English_all.csv --outFileName English --SG 1 --CBOW 0 --typeEmbedding fastText
Zulu_output_clean
This wil train word vectors for fastText using CBOW
- python train_word_vectors.py --inFilePath English_all.csv --outFileName English --CBOW 1 --SG 0 --typeEmbedding word2vec 
or 
- python train_word_vectors.py --inFilePath English_all.csv --outFileName English --CBOW 1 --typeEmbedding word2vec

Also use to show more options:
- python train_word_vectors.py --help

Outputfiles:

Models that were trained can be found in the following directories:
- ./fastText_models
- ./word2vec_models


Example of how such an output might look:
- ./fastText_models/fastText_train_epochs_trained_10_Afrikaans_SG_embedding_size_300
- ./fastText_models/fastText_train_epochs_trained_10_Afrikaans_CBOW_embedding_size_300


Note the "--outFileName Afrikaans" is contained inside the filename, with a short description of training details.
