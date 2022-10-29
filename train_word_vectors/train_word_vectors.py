#!/usr/bin/env python
from time import time
import pandas as pd
import os
import re
from gensim.models import Word2Vec
from gensim.models.fasttext import FastText as FT_gensim
from optparse import OptionParser
import logging
import multiprocessing
import argparse

os.system("taskset -p 0xff %d" % os.getpid())
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                    datefmt = '%H:%M:%S',
                    level=logging.INFO)

cores = multiprocessing.cpu_count()
'''Use this when you have to little RAM to store all your data'''


class SentencesIterator(object):

    def __init__(self, df):
        self.df = df

    def __iter__(self):
        self.df['Sentences'] = self.df['Sentences'].dropna()
        for row in self.df['Sentences']:
            try:
                sentence_stream = row.split()
                sentence_stream = \
                list(filter((' ').__ne__, sentence_stream))
            except Exception:
                print('No data data in row')
            yield sentence_stream


class TrainWordVectors(object):

    def __init__(
                 self, 
                 file_location,
                 name,
                 use_iterator):

        self.name = name
        self.use_iterator = use_iterator

        '''Embedding hyperparameters'''
        self.typeEmbedding = None
        self.sg = None
        self.window = None
        self.embeddingDimension = None
        self.epochs = None
        self.randomState = None
        self.min_count = None
        self.sample = None
        self.alpha = None
        self.min_alpha = None
        self.negative = None
        self.trainedModel = None
        self.workers = cores - 1
        '''Load data'''
        self.df = pd.read_csv(file_location)
        
    def trainWordVector(
                        self, typeEmbedding,
                        embeddingDimension, embMethod,
                        epochs, sg,
                        window_size, min_count,
                        sub_sampling, alpha,
                        min_alpha, negative_sampling,
                        randomState, word_vector):

        word2vecModelPath = './word2vec_models/'
        fastTextModelPath = './fastText_models/'

        os.makedirs(word2vecModelPath, exist_ok=True)
        os.makedirs(fastTextModelPath, exist_ok=True)

        self.typeEmbedding = word_vector
        self.sg = sg
        self.window = window_size
        self.embeddingDimension = embeddingDimension
        self.epochs = epochs
        self.randomState = randomState
        self.min_count = min_count
        self.sample = sub_sampling
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.negative = negative_sampling

        if self.typeEmbedding in 'fastText':
            modelPath = fastTextModelPath

            '''Directory paths and file-names'''
            wordvecModelName = 'fastText_train' + "_epochs_trained_" \
                               + str(self.epochs)
            nameEmbedding = wordvecModelName + '_' \
                            + self.name + '_'\
                            + embMethod + '_embedding_size_' \
                            + str(self.embeddingDimension)

            '''Create directory'''
            if not os.path.isdir(modelPath):
                os.mkdir(modelPath)
                print("Directory ", modelPath, "Created ")
            else:
                print("Directory ", modelPath, "Already Exists ")

            self.trainFastText(modelPath, nameEmbedding)
        elif self.typeEmbedding in 'word2vec':
            modelPath = word2vecModelPath

            '''Directory paths and file-names'''
            wordvecModelName = 'word2vec_train' + "_epochs_trained_" \
                                + str(self.epochs)
            nameEmbedding = wordvecModelName + '_' \
                            + self.name + '_' \
                            + embMethod + '_embedding_size_' \
                            + str(self.embeddingDimension)

            '''Create directory'''
            if not os.path.isdir(modelPath):
                os.mkdir(modelPath)
                print("Directory ", modelPath, "Created ")
            else:
                print("Directory ", modelPath, "Already Exists ")

            self.trainWord2vec(modelPath, nameEmbedding)

        else:
            print("Error Occurred model not available")

    '''Use if you have enough memory to load the data'''
    @staticmethod
    def sentencesFunc(df_train):
        trainText = df_train['Sentences'].dropna().tolist()
        trainText = [sent.split() for sent in trainText]
        trainText = list(filter((' ').__ne__, trainText))
        return trainText

    def trainWord2vec(self, wordvecModelPath, nameEmbedding):

        print("Begin word2vec!")
        w2vModel = Word2Vec(min_count=self.min_count,
                            window=self.window,
                            size=self.embeddingDimension,
                            sg=self.sg,
                            sample=self.sample,
                            seed=self.randomState,
                            alpha=self.alpha,
                            min_alpha=self.min_alpha,
                            negative=self.negative,
                            workers=cores - 1)
        if not self.use_iterator:
            sentences = self.sentencesFunc(self.df)
        else:
            sentences = SentencesIterator(self.df)

        t = time()
        print("Build vocabulary")
        '''Build the vocabulary'''
        w2vModel.build_vocab(sentences, progress_per=10000)
        print('Vocabulary done!')

        '''Train the word embedding model'''
        w2vModel.train(sentences, 
                       total_examples=w2vModel.corpus_count,
                       epochs=self.epochs,
                       total_words=w2vModel.corpus_total_words,
                       report_delay=2)

        w2vModel.save(wordvecModelPath + nameEmbedding)

        print('Time to train the model: {} mins'
              .format(round((time() - t) / 60, 2)))

    def trainFastText(self, wordvecModelPath, nameEmbedding):

        print("Begin FastText!")
        ftModel = FT_gensim(min_count=self.min_count,
                            window=self.window,
                            size=self.embeddingDimension,
                            sg=self.sg,
                            sample=self.sample,
                            seed=self.randomState,
                            alpha=self.alpha,
                            min_alpha=self.min_alpha,
                            negative=self.negative,
                            workers=cores - 1)
        if not self.use_iterator:
            sentences = self.sentencesFunc(self.df)
        else:
            sentences = SentencesIterator(self.df)
        t = time()
        print("Build vocabulary")
        '''Build the vocabulary'''
        ftModel.build_vocab(sentences, progress_per=10000)
        print('Vocabulary done!')
        
        '''Train the word embedding model'''
        ftModel.train(sentences,
                      epochs=self.epochs,
                      total_examples=ftModel.corpus_count,
                      total_words=ftModel.corpus_total_words)
        savepath = os.path.join(wordvecModelPath, "{}"
                                .format(nameEmbedding))
        ftModel.save(savepath)

        print("Epoch saved: {}".format(self.epochs),
              "Start next epoch ... ", sep="\n")
        print('Time to train the model: {} mins'
              .format(round((time() - t) / 60, 2)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train word2vec \
                                                  and fastText embeddings.')
    requiredNamed = parser.add_argument_group('Required named arguments')
    requiredNamed.add_argument('--inFilePath',
                               action="store", 
                               metavar='string',
                               type=str,
                               dest='inFilePath',
                               help="defines the input csv file path",
                               required=True)
    requiredNamed.add_argument('--outFileName',
                               action="store", 
                               metavar='string',
                               type=str,
                               dest='outFileName',
                               help="specifies word embedding output file name",
                               required=True)
    parser.add_argument('--typeEmbedding',
                        metavar='string',
                        type=str, 
                        nargs="*",
                        default='word2vec',
                        help="defines what type of word embedding should \
                              be trained specify either \"fastText\" \
                              or \"word2vec\" or select both")
    parser.add_argument('--embeddingDimension',
                        metavar='int',
                        type=int, 
                        default=300,
                        help="specifies input word embedding \
                              dimension - default dimension is 300")
    parser.add_argument('--CBOW',
                        metavar='bool',
                        type=bool, 
                        default=False, 
                        help="specifies that the model should be trained \
                         using continuous bag-of-words ")
    parser.add_argument('--SG',
                        metavar='bool',
                        type=bool, 
                        default=True, 
                        help="specifies that the model should be trained using skip-gram \
                        - default dimension is skip-gram")
    parser.add_argument('--Epochs',
                        metavar='int',
                        type=int, 
                        default=10, 
                        help="specifies number of epochs word embeddings are \
                         trained for - default is 10 epochs")
    parser.add_argument('--randomState',
                        metavar='int',
                        type=int, 
                        default=40, 
                        help="specifies random seed used to initialise word embedding model, \
                        for reproducibility - default value is 40")
    parser.add_argument('--minimumCount',
                        metavar='int',
                        type=int, 
                        default=2, 
                        help="ignores all words with total frequency lower than this \
                         - default value is 2")
    parser.add_argument('--subSampleFactor',
                        metavar='float',
                        type=float, 
                        default=1e-5, 
                        help="threshold for configuring which higher-frequency words \
                        are randomly downsampled - default value is 1e-5")
    parser.add_argument('--learningRate',
                        metavar='float',
                        type=float, 
                        default=0.025, 
                        help="specifies the initial learning rate \
                        - default value is 0.025")
    parser.add_argument('--minLearningRate',
                        metavar='float',
                        type=float, 
                        default=0.0001, 
                        help="specifies minimum learning rate the initial learning rate will \
                        linearly drop to as training progresses \
                        - default value is 0.0001")
    parser.add_argument('--negativeSampling',
                        metavar='int',
                        type=int, 
                        default=5, 
                        help="specifies how many “noise words” should be drawn (usually between 5-20) \
                         - default value is 5")
    parser.add_argument('--windowSize',
                        metavar='int',
                        type=int, 
                        default=5, 
                        help="maximum distance between the current and predicted word within a sentence \
                        - default value is 5")
    parser.add_argument('--useIterator',
                        metavar='bool',
                        type=bool, 
                        default=False, 
                        help="use this option if you have limited RAM to load sentences \
                        - default value is \"False\"")

    args = parser.parse_args()
    skipGram = {}
    if args.CBOW and not args.SG:
        skipGram['CBOW'] = 0
    elif args.SG and not args.CBOW:
        skipGram['SG'] = 1
    elif args.CBOW and args.SG:
        skipGram['CBOW'] = 0
        skipGram['SG'] = 1
    elif args.CBOW is False and (args.SG is False):
        parser.error("Select either SG or CBOW to be \"True\".")

    word_vectors = TrainWordVectors(file_location=args.inFilePath,
                                    name=args.outFileName,
                                    use_iterator=args.useIterator)
    for key, value in skipGram.items():
        for typeEmbed in args.typeEmbedding:
            word_vectors.trainWordVector(typeEmbedding=typeEmbed,
                                         embeddingDimension=args.embeddingDimension,
                                         epochs=args.Epochs, 
                                         sg=value,
                                         embMethod=key,
                                         window_size=args.windowSize, 
                                         min_count=args.minimumCount,
                                         sub_sampling=args.subSampleFactor,
                                         alpha=args.learningRate,
                                         min_alpha=args.minLearningRate,
                                         negative_sampling=args.negativeSampling,
                                         randomState=args.randomState, 
                                         word_vector=typeEmbed)
