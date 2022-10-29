# **Visualise pre-trained word embeddings using t-SNE**

for plot_word_vectors.py

#### **Dependencies:**
* gensim
* matplotlib
* sklearn

**Note:** These dependencies can be installed using pip python package manager.

## Example of how program is used:

#### To find what kind of arguments the program takes type:
* `python plot_word_vectors.py --help`

## Example of how to use:

### This wil give a 2D and 3D t_SNE plot of the pre-trained word vectors:

* `python plot_word_vectors.py --inModelPath ./fastText_models/fastText_train_epochs_trained_10_English_SG_embedding_size_300 --outFileName fastText_train_epochs_trained_10_English_SG_embedding_size_300
`

    **OR**

* `python plot_word_vectors.py --inModelPath ./fastText_models/fastText_train_epochs_trained_10_English_SG_embedding_size_300 --outFileName fastText_train_epochs_trained_10_English_SG_embedding_size_300 -2D 1 -3D 1
`

### This wil give a 2D t_SNE plot of the pre-trained word vectors:

* `python plot_word_vectors.py --inModelPath ./fastText_models/fastText_train_epochs_trained_10_English_SG_embedding_size_300 --outFileName fastText_train_epochs_trained_10_English_SG_embedding_size_300 -2D 1 -3D 0
`

    **OR**

* `python plot_word_vectors.py --inModelPath ./fastText_models/fastText_train_epochs_trained_10_English_SG_embedding_size_300 --outFileName fastText_train_epochs_trained_10_English_SG_embedding_size_300 -2D 1
`

### This wil give a 3D t_SNE plot of the pre-trained word vectors: 

* `python plot_word_vectors.py --inModelPath ./fastText_models/fastText_train_epochs_trained_10_English_SG_embedding_size_300 --outFileName fastText_train_epochs_trained_10_English_SG_embedding_size_300 -2D 0 -3D 1
`

    **OR**

* `python plot_word_vectors.py --inModelPath ./fastText_models/fastText_train_epochs_trained_10_English_SG_embedding_size_300 --outFileName fastText_train_epochs_trained_10_English_SG_embedding_size_300 -3D 1
`

## Output files:

#### Plots of embedding models can be found in the following directory:
* `./Visualisation`
