import gensim
import os
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import islice
plt.style.use('ggplot')


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
    wordvec_model = gensim.models.FastText.load(pretrained_model)
    wordvec_model.init_sims(replace=True)
    wv = wordvec_model.wv
    del wordvec_model
    list(islice(wv.vocab, 13030, 13050))

    words = []
    embeddings = []
    i = 0
    for word in wv.vocab:
        i+=1
        embeddings.append(wv[word])
        words.append(word)
        if i == nw:
            break

    return embeddings, words


def transform_2d(embeddings, words, out_file, random_state):
    tsne_wv_2d = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=random_state)
    embeddings_wv_2d = tsne_wv_2d.fit_transform(embeddings)

    title = 'Visualizing 2D Embedding using t-SNE'
    filename = out_file + "_tsne_wv_2d.png"
    label = 'Word2Vec embeddigns'
    tsne_plot_2d(filename, label, title, embeddings_wv_2d, words)


def transform_3d(embeddings, out_file, random_state):
    tsne_wv_3d = TSNE(perplexity=30, n_components=3, init='pca', n_iter=3500, random_state=random_state)
    embeddings_wv_3d = tsne_wv_3d.fit_transform(embeddings)

    title = 'Visualizing 3D Embedding using t-SNE'
    label = 'Embedding model'
    filename = out_file + "_tsne_wv_3d.png"
    tsne_plot_3d(filename, title, label, embeddings_wv_3d)


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
        transform_2d(emb, wrds, out, args.random_state)
    elif args.three_D and not args.two_D:
        transform_3d(emb, out, args.random_state)
    elif args.two_D and args.three_D:
        transform_2d(emb, wrds, out, args.random_state)
        transform_3d(emb, out, args.random_state)
    elif args.two_D is False and (args.three_D is False):
        parser.error("Select either 2D or 3D to be \"True\".")
