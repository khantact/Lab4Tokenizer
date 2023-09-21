import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tokenizer as T
import subprocess

def testCreateVocab():

    import json
    with open('./tests/goldTrainDict.json', 'r') as f:
        gold_word2idx = json.load(f)

    tokenizer = T.Tokenizer()
    fname = './tests/train.txt'

    try:
        tokenizer.create_vocab(fname, freqThreshold=100) 
    except NotImplementedError:
        assert 0, 'You need to implement the create_vocab function'

    for word in gold_word2idx:
        if word == '' or word == ' ':
            continue
        assert word in tokenizer.word2idx, f"You are missing '{word}' in your word2idx when reading tests/train.text with freqThreshold=100"

    for idx, word in enumerate(tokenizer.idx2word):
        assert idx == tokenizer.word2idx[word], f"There is a mismatch between your idx2word and word2idx for '{word}'"

