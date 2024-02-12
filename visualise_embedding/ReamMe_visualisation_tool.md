# **Visualise pre-trained word embeddings using t-SNE**

for visualise_embedding.py

#### **Dependencies:**
* gensim
* matplotlib
* sklearn

**Note:** These dependencies can be installed using pip python package manager.

## Example of how program is used:

#### To find what kind of arguments the program takes type:
* `python visualise_embedding.py --help`

## Example of how to use:

### This will give a 2D and 3D t_SNE plot of the pre-trained word vectors:

* `python visualise_embedding.py --inModelPath ./fastText_models/fastText_train_epochs_trained_10_English_SG_embedding_size_300 --outFileName fastText_train_epochs_trained_10_English_SG_embedding_size_300
`

    **OR**

* `python visualise_embedding.py --inModelPath ./fastText_models/fastText_train_epochs_trained_10_English_SG_embedding_size_300 --outFileName fastText_train_epochs_trained_10_English_SG_embedding_size_300 -2D 1 -3D 1
`

### This will give a 2D t_SNE plot of the pre-trained word vectors:

* `python visualise_embedding.py --inModelPath ./fastText_models/fastText_train_epochs_trained_10_English_SG_embedding_size_300 --outFileName fastText_train_epochs_trained_10_English_SG_embedding_size_300 -2D 1 -3D 0
`

    **OR**

* `python visualise_embedding.py --inModelPath ./fastText_models/fastText_train_epochs_trained_10_English_SG_embedding_size_300 --outFileName fastText_train_epochs_trained_10_English_SG_embedding_size_300 -2D 1
`

### This will give a 3D t_SNE plot of the pre-trained word vectors: 

* `python visualise_embedding.py --inModelPath ./fastText_models/fastText_train_epochs_trained_10_English_SG_embedding_size_300 --outFileName fastText_train_epochs_trained_10_English_SG_embedding_size_300 -2D 0 -3D 1
`

    **OR**

* `python visualise_embedding.py --inModelPath ./fastText_models/fastText_train_epochs_trained_10_English_SG_embedding_size_300 --outFileName fastText_train_epochs_trained_10_English_SG_embedding_size_300 -3D 1
`

## Output files:

#### Plots of embedding models can be found in the following directory:
* `./Visualisation`
