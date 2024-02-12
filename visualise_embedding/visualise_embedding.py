import gensim
from gensim.models import Word2Vec
import os
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import islice
import numpy as np
from matplotlib.lines import Line2D
from itertools import cycle
import seaborn as sns
import pandas as pd

sns.set_style("darkgrid")

plt.style.use('ggplot')

# random.seed(10)

# def tsnescatterplot(word_embedding_model_path, word_t, word_f):
#     """ Plot in seaborn the results from the t-SNE dimensionality reduction algorithm of the vectors of a query word,
#     , word_vectors is the gensdim model. This is for word vectors
#     """
#     wordvec_model = gensim.models.Word2Vec.load(word_embedding_model_path)
#     # wordvec_model.init_sims(replace=True)
#     wv = wordvec_model.wv
#
#     # arrays = np.empty((0, embedding_dimension), dtype='f')
#     word_labels =[word_t, word_f]
#     color_list = ['red', 'red']
#     list_of_vectors = []
#
#     # adds the vector of the query word
#     # arrays = np.append(arrays, word_vectors.__getitem__([word_t]), axis=0)
#     array_word_one = np.array(wv.get_vector(word_t, norm=True))#, axis=0)
#     list_of_vectors.append(array_word_one)
#
#     # adds the vector of the query word
#     # arrays = np.append(arrays, word_vectors.__getitem__([word_f]), axis=0)
#     array_word_two = np.array(wv.get_vector(word_f, norm=True))#, axis=0)
#     list_of_vectors.append(array_word_two)
#
#     # gets list of most similar words from query word
#     close_words = [i[0] for i in wv.most_similar(positive=[word_t], topn=50)]
#     list_names = [i[0] for i in wv.most_similar(positive=[word_f], topn=50)]
#     print(len(close_words))
#     print(len(list_names))
#
#     for i in close_words:
#         if i in list_names:
#             close_words.remove(i)
#
#     # close_words.remove(word_f)
#     # list_names.remove(word_t)
#     # adds the vector for each of the closest words to the array
#     for wrd in close_words:
#         wrd_vector = wv.get_vector(wrd, norm=True)
#         word_labels.append(wrd)
#         color_list.append('blue')
#         list_of_vectors.append(np.array(wrd_vector))
#         # arrays = np.append(arrays, wrd_vector, axis=0)
#
#
#     # adds the vector for each of the words from list_names to the array
#     for wrd in list_names:
#         wrd_vector = wv.get_vector(wrd, norm=True)
#         word_labels.append(wrd)
#         color_list.append('orange')
#         # arrays = np.append(arrays, wrd_vector, axis=0)
#         list_of_vectors.append(np.array(wrd_vector))
#
#     # Reduces the dimensionality from 300 to 2 dimensions with t-SNE
#     # Finds t-SNE coordinates for 2 dimensions
#     np.set_printoptions(suppress=True)
#
#     array_of_list_of_vectors = np.array(list_of_vectors)
#     print(array_of_list_of_vectors)
#     Y = TSNE(n_components=2, random_state=5, perplexity=35, n_iter=4000).fit_transform(array_of_list_of_vectors)#50
#
#     # Sets everything up to plot
#     df = pd.DataFrame({'x': [x for x in Y[:, 0]],
#                        'y': [y for y in Y[:, 1]],
#                        'words': word_labels,
#                        'color': color_list})
#
#     fig, _ = plt.subplots()
#     fig.set_size_inches(22, 12)
#
#     # Basic plot
#     p1 = sns.scatterplot(x="x", y="y", hue="color", s=1, legend=None,data=df) # scatter_kws={'s': 1}
#
#     custom = [Line2D([], [], marker='.', color='blue', linestyle='None',markersize=8),
#               Line2D([], [], marker='.', color='orange', linestyle='None',markersize=8)]
#
#     legend=plt.legend(custom, ['True Detection', 'False Detection'], loc='upper right', title="Detection type", fontsize=15)
#     plt.setp(legend.get_title(), fontsize=24)
#     # Adds annotations one by one with a loop
#     for line in range(0, df.shape[0]):
#         p1.text(df["x"][line],
#                 df['y'][line],
#                 '  ' + df["words"][line].title(),
#                 horizontalalignment='left',
#                 verticalalignment='bottom',
#                 color=df['color'][line],
#                 fontsize=1,
#                 weight='normal'
#                 ).set_size(10)
#
#     plt.xlim(Y[:, 0].min(), Y[:, 0].max())
#     plt.ylim(Y[:, 1].min(), Y[:, 1].max())
#
#     plt.title('Visualizing Word Embeddings using t-SNE fastText SG',fontsize=24)
#     # lgnd=plt.legend(fontsize=50)
#     # lgnd.legendHandles[0]._legmarker.set_markersize(6)
#     # plt.setp(p1._legend.get_texts(), fontsize=16)
#     plt.tick_params(labelsize=20)
#     plt.xlabel('tsne-one', fontsize=24)
#     plt.ylabel('tsne-two', fontsize=24)
#
#     # plt.title('t-SNE visualization for {}'.format(word.title()))
#     plt.savefig('word_scatter.png')
#     plt.savefig('word_scatter.eps', format='eps')

def get_word_vectors(word_vectors, word_list):
    return [np.array(word_vectors.get_vector(word, norm=True)) for word in word_list]


def get_similar_words(word_vectors, word, topn=10):
    return [word for word, _ in word_vectors.most_similar(positive=[word], topn=topn)]


def tsnescatterplot(word_embedding_model_path, outputfile_path, words):
    wordvec_model = gensim.models.Word2Vec.load(word_embedding_model_path)
    wv = wordvec_model.wv

    # Prepare data
    word_labels = words
    color_list = ['red'] * len(word_labels)
    list_of_vectors = get_word_vectors(wv, word_labels)

    set_of_similar_words = dict()

    color_pallet = ['blue', 'orange', 'green', 'pink', 'black', 'navy', 'indigo', 'gold']
    color_pallet_cycle = cycle(color_pallet)
    color_list_all_words = []
    # Get similar words and add them to the set
    for word in word_labels:
        similar_words = get_similar_words(wv, word)
        tuple_similar_words = [(k, None) for k in similar_words]
        set_of_similar_words.update(tuple_similar_words)  # Use update instead of add
        color_list_all_words.extend(len(tuple_similar_words) * [next(color_pallet_cycle)])

    # Update labels and vectors
    word_labels_extended = word_labels + list(set_of_similar_words)
    list_of_vectors.extend(get_word_vectors(wv, list(set_of_similar_words)))

    color_list.extend(color_list_all_words)
    print(color_list)

    # t-SNE reduction
    np.set_printoptions(suppress=True)
    array_of_list_of_vectors = np.array(list_of_vectors)
    Y = TSNE(n_components=2, random_state=43, perplexity=20, n_iter=15000).fit_transform(array_of_list_of_vectors)

    print(f"Length of Y[:, 0]: {len(Y[:, 0])}")
    print(f"Length of Y[:, 1]: {len(Y[:, 1])}")
    print(f"Length of word_labels: {len(word_labels_extended)}")
    print(f"Length of color_list: {len(color_list)}")


    # DataFrame for plotting
    df = pd.DataFrame({'x': Y[:, 0], 'y': Y[:, 1], 'words': word_labels_extended, 'color': color_list[:-4]})

    # Plot setup
    fig, _ = plt.subplots()
    fig.set_size_inches(22, 12)
    p1 = sns.scatterplot(x="x", y="y", hue="color", data=df, legend=None, s=1)

    # Custom legend
    # custom = [Line2D([], [], marker='.', color='blue', linestyle='None', markersize=8),
    #           Line2D([], [], marker='.', color='orange', linestyle='None', markersize=8)]
    custom = []
    color_pallet_cyclic_iterator = cycle(color_pallet)

    for i in range(len(word_labels)):
        custom.append(Line2D([], [], marker='.', color=next(color_pallet_cyclic_iterator), linestyle='None', markersize=8))


    legend = plt.legend(custom, word_labels, loc='lower left', title="Words", fontsize=12)

    plt.setp(legend.get_title(), fontsize=24)

    # Annotations
    for line in range(df.shape[0]):
        p1.text(df["x"][line], df['y'][line], ' ' + df["words"][line].title(),
                horizontalalignment='left', verticalalignment='bottom', color=df['color'][line], fontsize=10)

    # Axis and title setup
    plt.xlim(Y[:, 0].min(), Y[:, 0].max())
    plt.ylim(Y[:, 1].min(), Y[:, 1].max()) #continuous bag of words
    plt.title('Visualising south african word embeddings using t-SNE fastText skip gram', fontsize=24)
    plt.tick_params(labelsize=20)
    plt.xlabel('tsne-one', fontsize=24)
    plt.ylabel('tsne-two', fontsize=24)

    # Save figures
    plt.savefig(outputfile_path)
    # plt.savefig('word_scatter.eps', format='eps')

    plt.show()


def tsne_plot_2d(filename, label, title, embeddings_wv_2d, words):
    x = []
    y = []
    for value in embeddings_wv_2d:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i], label=label)
        plt.annotate(words[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom',
                     size=15)

    plt.title(title)
    plt.grid(True)
    plt.savefig(os.path.join(filename), format='png', dpi=150, bbox_inches='tight')
    plt.show()


def tsne_plot_3d(filename, title, label, embeddings):
    fig = plt.figure()
    ax = Axes3D(fig)
    plt.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], label=label)
    plt.legend(loc=4)
    plt.title(title)
    plt.savefig(os.path.join(filename), format='png', dpi=150, bbox_inches='tight')
    plt.show()


def get_values(nw, pretrained_model):
    wordvec_model = gensim.models.Word2Vec.load(pretrained_model)
    # wordvec_model.init_sims(replace=True)
    wv = wordvec_model.wv
    # del wordvec_model
    # list(islice(wordvec_model.wv, 13030, 13050))

    words = []
    embeddings = []
    i = 0
    for word in wv.key_to_index:

        i += 1
        embeddings.append(wv[word])
        words.append(word)
        if i == nw:
            break

    return embeddings, words


def transform_2d(embeddings, words, out_file, random_state):
    tsne_wv_2d = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=random_state)
    embeddings_array = np.array(embeddings)
    embeddings_wv_2d = tsne_wv_2d.fit_transform(embeddings_array)

    title = 'Visualizing 2D Embedding using t-SNE'
    filename = out_file + "_tsne_wv_2d.png"
    label = 'Word2Vec embeddings'
    tsne_plot_2d(filename, label, title, embeddings_wv_2d, words)


def transform_3d(embeddings, out_file, random_state):
    # tsne_wv_3d = TSNE(perplexity=30, n_components=3, init='pca', n_iter=3500, random_state=random_state)
    # # print(embeddings)
    #
    # # Assuming 'embeddings' is your pre-processed data
    # embeddings_array = np.array(embeddings)
    # embeddings_wv_3d = tsne_wv_3d.fit_transform(embeddings_array)
    #
    # title = 'Visualizing 3D Embedding using t-SNE'
    # label = 'Embedding model'
    # filename = out_file + "_tsne_wv_3d.png"

    # tsne_plot_3d(filename, title, label, embeddings_wv_3d)

    try:
        # Set up t-SNE for 3D
        tsne = TSNE(n_components=3, random_state=random_state)

        embeddings_array = np.array(embeddings)

        embeddings_wv_3d = tsne.fit_transform(embeddings_array)

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embeddings_wv_3d[:, 0], embeddings_wv_3d[:, 1], embeddings_wv_3d[:, 2])

        filename = out_file + "_tsne_wv_3d.png"

        # Save the plot
        plt.savefig(os.path.join(filename), format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualise pre-trained word embedding models using t-SNE plots.')
    requiredNamed = parser.add_argument_group('Required named arguments')
    requiredNamed.add_argument('--inModelPath',
                               action="store",
                               metavar='string',
                               type=str,
                               dest='inModelPath',
                               help="specifies word embedding input file name",
                               required=True)
    requiredNamed.add_argument('--outFileName',
                               action="store",
                               metavar='string',
                               type=str,
                               dest='outFileName',
                               help="specifies t-SNE output file name",
                               required=True)
    parser.add_argument('-2D',
                        '--two_D',
                        metavar='bool',
                        type=bool,
                        default=False,
                        help="specifies if 2D model should be plot")
    parser.add_argument('-nw',
                        '--number_words',
                        metavar='int',
                        type=int,
                        default=100,
                        help="number of words plotted in t-SNE plot")
    parser.add_argument('-rs',
                    '--random_state',
                    metavar='int',
                    type=int,
                    default=40,
                    help="t-SNE is a stochastic dimensionality reduction method- \
                    it might require you to work through different random \
                     seeds to get meaningful results")
    parser.add_argument('-3D',
                        '--three_D',
                        metavar='bool',
                        type=bool,
                        default=False,
                        help="specifies if 3D model should be plot")

    args = parser.parse_args()

    model = args.inModelPath
    emb, wrds = get_values(args.number_words, model)

    visualisation_path = './Visualisation/'
    os.makedirs(visualisation_path, exist_ok=True)
    out = visualisation_path + args.outFileName

    if args.two_D and not args.three_D:
        # transform_2d(emb, wrds, out, args.random_state) #['pyn', 'politiek', 'verslawing', 'nuus', 'kar', 'kultuur', 'bakkie', 'sport', 'pasta', 'kos', 'emosies'])
        tsnescatterplot(model, args.outFileName, ['pain', 'politics', 'addiction', 'news', 'car', 'culture', 'bakkie', 'sport', 'pasta', 'food', 'emotions', 'disney','south_africa','radio'])#['ubuhlungu', 'ipolitiki', 'umlutha', 'izindaba','imoto', 'isiko', 'iveni', 'ezemidlalo', 'ukudla', 'imizwa'])#['pain', 'politics', 'addiction', 'news', 'car', 'culture', 'bakkie', 'sport', 'pasta', 'food', 'emotions'])#['ubuhlungu', 'ipolitiki', 'umlutha', 'izindaba','imoto', 'isiko', 'iveni', 'ezemidlalo','i-pasta', 'ukudla', 'imizwa'])#['ubuhlungu', 'ipolitiki', 'umlutha', 'izindaba','imoto', 'isiko', 'iveni', 'ezemidlalo', 'ukudla', 'imizwa'])#['pain', 'politics', 'addiction', 'news', 'car', 'culture', 'bakkie', 'sport', 'pasta', 'food', 'emotions'])#['ubuhlungu', 'ipolitiki', 'umlutha', 'izindaba','imoto', 'isiko', 'iveni', 'ezemidlalo','i-pasta', 'ukudla', 'imizwa'])#['pyn', 'politiek', 'verslawing', 'nuus', 'kar', 'kultuur', 'bakkie', 'sport', 'pasta', 'kos', 'emosies'])


    elif args.three_D and not args.two_D:
        transform_3d(emb, out, args.random_state)
    elif args.two_D and args.three_D:
        transform_2d(emb, wrds, out, args.random_state)
        transform_3d(emb, out, args.random_state)
    elif args.two_D is False and (args.three_D is False):
        parser.error("Select either 2D or 3D to be \"True\".")
