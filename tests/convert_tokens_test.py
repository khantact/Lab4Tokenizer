import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tokenizer as T
import subprocess

def testConvertTokens2IDs():

    idx2word = ['is', 'the', 'man', 'who', 'tall', 'happy', '?', '<s>', '</s>', '<unk>', '<pad>'] 

    word2idx = {'is':0, 'the':1, 'man':2, 'who':3, 'tall':4, 
                     'happy': 5, '?': 6, '<s>': 7, '</s>': 8, '<unk>': 9,
                     '<pad>': 10}

    tokenizer = T.Tokenizer(maxSequenceLength=10)
    tokenizer.word2idx = word2idx
    tokenizer.idx2word = idx2word

    try: 
        tokenizer.convert_tokens_to_ids('the')
    except NotImplementedError:
        assert 0, 'You need to implement the convert_tokens_to_ids function'

    assert tokenizer.convert_tokens_to_ids("tall") == [4]

    assert tokenizer.convert_tokens_to_ids(["is", "the", "man", "who", "is", "tall", "happy", "?"]) == [0, 1, 2, 3, 0, 4, 5, 6]

    assert tokenizer.convert_tokens_to_ids(["<s>", "what", "the", "man", "is", "<pad>"]) == [7, 9, 1, 2, 0, 10]
