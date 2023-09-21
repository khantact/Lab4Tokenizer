import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tokenizer as T
import subprocess

def testPreprocess():

    tokenizer = T.Tokenizer()

    try:
        tokenizer.preprocess('the')
    except NotImplementedError:
        assert 0, "You need to implement the preprocess function"

    text = "      Well, I thought; Oh, I guess) I     hate it!?\n"
    assert tokenizer.preprocess(text) == 'well , i thought ; oh , i guess ) i hate it ! ?'

    text = "<s> She loves <unk>, cat </s> <pad> <pad> <pad>"
    assert tokenizer.preprocess(text) == '<s> she loves <unk> , cat </s> <pad> <pad> <pad>'

    tokenizer = T.Tokenizer(lower=False)

    text = "Noam Chomsky!"
    assert tokenizer.preprocess(text) == 'Noam Chomsky !', "Make sure to lowercase text only when self.lower is True"

