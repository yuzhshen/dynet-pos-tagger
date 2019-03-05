## DyNet POS Tagger

Bidirectional LSTM for POS tagging. 

- To train, modify the following constants defined in `train.py`.
    - `GLOVE_PATH`: Set to the GloVe TXT file, e.g. `'/Users/zhao/dev/#resources/glove.6B/glove.6B.50d.txt'`
    - `INPUT_DIM`: Set to the dimension of the GloVe embeddings you are using, e.g. `50`
    - `POS_TREEBANK_PATH`: Set to the directory for POS tags (inside this folder there should be numbered folders which contain the actual WSJ .pos files), e.g. `./penn-treebank-pos`

- Project and README are WIP.