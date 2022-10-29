Visualise pre-trained word embeddings using t-SNE
for plot_word_vectors.py

Dependiencies:
1)gensim
2)matplotlib
3)sklearn

These dependiecies can be installed using pip python package manager.

Example of how program is used:

To findout what kind of arguments the program takes type:
- python plot_word_vectors.py --help

Example of how to use:

This wil give a 2D and 3D t_SNE plot of the pre-trained word vectors
- python plot_word_vectors.py --inModelPath ./fastText_models/fastText_train_epochs_trained_10_English_SG_embedding_size_300 --outFileName fastText_train_epochs_trained_10_English_SG_embedding_size_300
or
- python plot_word_vectors.py --inModelPath ./fastText_models/fastText_train_epochs_trained_10_English_SG_embedding_size_300 --outFileName fastText_train_epochs_trained_10_English_SG_embedding_size_300 -2D 1 -3D 1

This wil give a 2D t_SNE plot of the pre-trained word vectors
- python plot_word_vectors.py --inModelPath ./fastText_models/fastText_train_epochs_trained_10_English_SG_embedding_size_300 --outFileName fastText_train_epochs_trained_10_English_SG_embedding_size_300 -2D 1 -3D 0
or
- python plot_word_vectors.py --inModelPath ./fastText_models/fastText_train_epochs_trained_10_English_SG_embedding_size_300 --outFileName fastText_train_epochs_trained_10_English_SG_embedding_size_300 -2D 1

This wil give a 3D t_SNE plot of the pre-trained word vectors
- python plot_word_vectors.py --inModelPath ./fastText_models/fastText_train_epochs_trained_10_English_SG_embedding_size_300 --outFileName fastText_train_epochs_trained_10_English_SG_embedding_size_300 -2D 0 -3D 1
or
- python plot_word_vectors.py --inModelPath ./fastText_models/fastText_train_epochs_trained_10_English_SG_embedding_size_300 --outFileName fastText_train_epochs_trained_10_English_SG_embedding_size_300 -3D 1


Outputfiles:

Plots of embedding models can be found in the following directory:
- ./Visualisation
