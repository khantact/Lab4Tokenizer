import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tokenizer as T
import subprocess

def testEncode():

    idx2word = ['is', 'the', 'man', 'who', 'tall', 'happy', '?', '<s>', '</s>', '<unk>', '<pad>'] 
    word2idx = {'is':0, 'the':1, 'man':2, 'who':3, 'tall':4, 
                     'happy': 5, '?': 6, '<s>': 7, '</s>': 8, '<unk>': 9,
                     '<pad>': 10}

    tokenizer = T.Tokenizer(maxSequenceLength=10)
    tokenizer.word2idx = word2idx
    tokenizer.idx2word = idx2word

    try:
        tokenizer.encode("the")
    except NotImplementedError:
        assert 0, "You need to implement the encode function"

    assert tokenizer.encode('<unk>') == [9], "Simple check failed"

    text = 'Is the man tall?'
    assert tokenizer.encode(text, add_special_tokens=True) == [7, 0, 1, 2, 4, 6, 8], "Checking add_special_tokens=True failed"

    text = ['Is the man tall?', 'Or is the man who is tall tall?']
    studentTensor = tokenizer.encode(text, padding=True)
    goldTensor = [[0, 1, 2, 4, 6, 10, 10, 10, 10], [9, 0, 1, 2, 3, 0, 4, 4, 6]]
    assert studentTensor == goldTensor, "Checking padding=True failed"

    text = 'Is the man who is tall also quite tall though?'
    assert tokenizer.encode(text, truncate=True, 
                            add_special_tokens=True) == [7, 0, 1, 2, 3, 0, 4, 9, 9, 8], "Checking truncate=True (with maxSequenceLength as 10) and add_special_tokens=True failed"
