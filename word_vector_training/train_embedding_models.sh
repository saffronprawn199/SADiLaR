#!/bin/bash

# Define the input CSV files
#INPUT_FILES=("/Users/rhynostrydom/PycharmProjects/SADiLaR/Data/english_training_data.csv" "/Users/rhynostrydom/PycharmProjects/SADiLaR/Data/afrikaans_training_data.csv" "/Users/rhynostrydom/PycharmProjects/SADiLaR/Data/zulu_training_data.csv")

INPUT_FILES=("/Users/rhynostrydom/PycharmProjects/SADiLaR/Data/false_positive_classifier_radio_broadcast_news.csv")

#"/Users/rhynostrydom/PycharmProjects/SADiLaR/Data/english_training_data.csv")
# Function to train models
train_models() {
    local infile=$1
    local outfile=$2

    # Train for fastText and Word2Vec using both CBOW and SG
    python train_word_vectors.py --inFilePath "$infile" --outFileName "$outfile" --CBOW 1 --SG 1 --typeEmbedding word2vec fastText

    # Train for fastText using SG
#    python train_word_vectors.py --inFilePath "$infile" --outFileName "$outfile" --SG 1 --typeEmbedding fastText

    # Train for fastText using CBOW
#    python train_word_vectors.py --inFilePath "$infile" --outFileName "$outfile" --CBOW 1 --typeEmbedding word2vec
}

# Iterate over input files and train models
for FILE in "${INPUT_FILES[@]}"; do
    OUT_NAME=$(basename "$FILE" .csv)  # Extract base name from file for output
    train_models "$FILE" "$OUT_NAME"
done

# Optional: List or move output files
echo "Models trained. Check the ./fastText_models and ./word2vec_models directories for output."



