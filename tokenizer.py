import re  # Python regular expressions (may be useful)
import string  # Python string library
from typing import Union, Dict, List, Tuple


class Tokenizer:
    """
    A class for tokenzing text for input to a model. 

    Attributes:
        word2idx (dict): A dictionary mapping tokens (string) to id.
        idx2word (dict): A list which associates ids with tokens 
                         (e.g., idx2word[9] returns token with id 9). 

        maxSequenceLength (int): Maximum length of text. Default is 1024.

        unk_token (str): Token for unknown tokens. Default is <unk>.
        unk_token_id (int): Token id for unknown tokens.

        bos_token (str): Beginning of sequence token. Default is <s>.
        bos_token_id (int): Token id for bos token.

        eos_token (str): End of sequence token. Default is </s>.
        eos_token_id (int): Token id for eos token.

        pad_token (str): Token for padding. Default is <pad>.
        pad_token_id (int): Token id for mask token.

        add_special_tokens (bool): Whether to add bos token to start 
                                   and eos token to end of sequence when __call__. 
                                   Default is True.

        lower (bool): Whether to lowercase text

        padding (bool): Whether to pad (when applicable) when __call__.
                        Default is True.
        truncate (bool): Whether to right truncate when sequence length
                         is greater than maxSequenceLength when __call__.
                         Default is True.

    """

    def __init__(self,
                 maxSequenceLength=1024,
                 unk_token="<unk>",
                 bos_token='<s>',
                 eos_token='</s>',
                 pad_token='<pad>',
                 add_special_tokens=True,
                 lower=True,
                 padding=True,
                 truncate=True,
                 ):

        self.word2idx = dict()
        self.idx2word = []

        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token

        self.maxSequenceLength = maxSequenceLength

        self.add_special_tokens = add_special_tokens
        self.lower = lower
        self.padding = padding
        self.truncate = truncate

    def __len__(self):
        """
        Sets len operator for the class to number of tokens in vocab.
        """
        return len(self.idx2word)

    def __call__(self,
                 text: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Sets call operator for the class to be encode. Thus, 
        tokenizer(text, ...) is equivalent to tokenizer.encode(text, ...)
        """

        return self.encode(text,
                           add_special_tokens=self.add_special_tokens,
                           padding=self.padding,
                           truncate=self.truncate)

    @property
    def unk_token_id(self):
        """
        Fixes self.unk_token_id
        """
        if self.unk_token not in self.word2idx:
            return None
        return self.word2idx[self.unk_token]

    @property
    def bos_token_id(self):
        """
        Fixes self.bos_token_id
        """
        if self.bos_token not in self.word2idx:
            return None
        return self.word2idx[self.bos_token]

    @property
    def eos_token_id(self):
        """
        Fixes self.eos_token_id
        """
        if self.eos_token not in self.word2idx:
            return None
        return self.word2idx[self.eos_token]

    @property
    def pad_token_id(self):
        """
        Fixes self.pad_token_id
        """
        if self.pad_token not in self.word2idx:
            return None
        return self.word2idx[self.pad_token]

    def save_tokenizer(self, outname: str):
        """
        Save the tokenizer to a plain txt file named outname. 
        Format the file such that each line has one token and 
        the line number corresponds to the index of that token.

        For example, 
            assuming we have self.idx2word = ['the', 'cat']
            outname should contain:
            the
            cat
        """
        with open(outname, 'w') as f:
            for word in self.idx2word:
                f.write(word+'\n')

    def load_tokenizer(self, vocabfname: str):
        """
        Load a tokenizer from a plain txt file name vocabfname
        (i.e. the output of save_tokenizer). 
        That is, we assume the format is such 
        that each line has one token with the line in the file 
        the index of the token in the vocabulary (counting from 0). 
        You should update both word2idx and idx2word!

        Args:
            vocabfname (str): Name of vocab file

        For example, 
            >>> from tokenizer import Tokenizer
            >>> tokenizer = Tokenizer()
            >>> tokenizer.load_tokenizer('ToyVocab.txt')
            >>> tokenizer.word2idx
            >>> {'the': 0, 'a': 1, 'cat': 2, 'loves': 3, 'eats': 4, 'food': 5, 
            ...      '.': 6, '!': 7, '<unk>': 8, 
            ...      '<pad>': 9, '<s>': 10, '</s>': 11}

            >>> tokenizer.idx2word
            >>> ['the', 'a', 'cat', 'loves', 'eats', 'food', '.', '!',  
            ... '<unk>', '<pad>', '<s>', '</s>']

        """
        with open(vocabfname, 'r') as f:
            for line in f:
                line = line.strip()
                if line not in self.word2idx:
                    self.word2idx[line] = len(self.idx2word)
                    self.idx2word.append(line)

    def preprocess(self, text: str) -> str:
        """
        Preprocess the text for use by tokenizer. It should 
        separate punctuation (!"#$%&'()*+,-./\\:;=?@[]^_`{|}~) 
        from words, lowercase the input (if specified with 
        self.lower), and remove any newline 
        characters, trailing spaces, or extra spaces. 
        Take care that the punctuation does not include < or >, 
        so that the special tokens (e.g., <unk>)
        are not modified and with / in </s>.

        Args:
            text (str): Text to be input to the tokenizer.

        Returns:
            str: Preprocessed text

        For example, 
            >>> from tokenizer import Tokenizer
            >>> tokenizer = Tokenizer()
            >>> tokenizer.preprocess(" the man, who is tall, is happy!\n")
            >>> "the man , who is tall , is happy !"
            >>> tokenizer.preprocess(" The <unk> cat loves to buy food; I love him? \n\n")
            >>> "the <unk> cat loves to buy food ; i love him ?"

        Hints: 
            For handling punctuation, you may consider using 
            regular expressions using pythons re package or
            the string functions translate and maketrans. 
            Consider, for example, wanting to replace B with ABC: 
            >>> text = 'CD B EF'
            >>> re.sub('B', r'ABC', text)
            >>> 'CD ABC EF'
            ...
            >>> text = 'CD B EF'
            >>> table = {'B': "ABC"}
            >>> text.translate(str.maketrans(table))
            >>> 'CD ABC EF'
        """
        toRet = ""
        newS = re.findall(r'[\w]+|[<>!\"#$%&\'()*+,-./\\:;=?@\[\]\^\_\`\{\|\}\~\)\(]', text)
        for word in newS:
            if self.lower:
                word = word.lower()
            if (re.search(r'[</]',word)):
                toRet += word
            elif (re.search(r'>', word)):
                toRet = toRet.rstrip()
                toRet += word + " "
            else: 
                toRet += word + " "
        return toRet.strip()

    def tokenize(self, text: str) -> List[str]:
        """
        Takes a string a returns a list of tokens. Tokens are defined
        as space delineated characters. 

        Args: 
           text (str): text to be input to tokenizer.

        Returns:
            List[str]: A list of strings (words).

        For example, 
            >>> from tokenizer import Tokenizer
            >>> tokenizer = Tokenizer()
            >>> tokenizer.tokenize("the man , who is tall , is happy !")
            >>> ["the", "man", ",", "who", "is", "tall", ",", "is", "happy", "!"]
        """
        return text.split(' ')

    def word_tokenize(self, text: str) -> List[str]:
        """
        Takes a string and returns a list of tokens. 

        Args: 
            text (str): input string

        Returns:
            List[str]: A list of tokens (words).

        For example, 
            >>> from tokenizer import Tokenizer
            >>> tokenizer = Tokenizer()
            >>> tokenizer.word_tokenize("the man, who is tall, is happy!")
            >>> ["the", "man", ",", "who", "is", "tall", ",", "is", "happy", "!"]
        """
        return self.tokenize(self.preprocess(text))

    def convert_tokens_to_ids(self,
                              tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Takes a string (a token) or a list of strings 
        (tokens; output of tokenize) and returns the id(s) of the token(s). 
        If the token is not in the vocabulary return the unk token id.

        Args:
            tokens (str, List[str]): Token or list of tokens to be converted to ids. 

        Returns:
            int | List[int]: The id of the token or a list of the ids of the tokens.  

        For example, 
            assuming that self.word2idx = {'the': 0, 'cat': 1, '<unk>': 2}
            >>> tokenizer.convert_tokens_to_ids("the")
            >>> [0]
            >>> tokenizer.convert_tokens_to_ids(["the", "cat"])
            >>> [0, 1]
            >>> tokenizer.convert_tokens_to_ids(["the", "cat", "sleeps"])
            >>> [0, 1, 2]

        Hint: 
            Consider using the built-in function type()
            Use self.word2idx

        """
        unknown = self.word2idx['<unk>']
        keys = self.word2idx.keys()
        tokenIDs = []
        if (type(tokens) == str):
            if (tokens in keys):
                return [self.word2idx[tokens]]
            else:
                return unknown
        else:
            for token in tokens:
                if (token in keys):
                    tokenIDs.append(self.word2idx[token])
                else:
                    tokenIDs.append(unknown)
        return tokenIDs

    # TODO
    def encode(self, text: Union[str, List[str]],
               add_special_tokens: bool = False,
               padding: bool = False,
               truncate: bool = False) -> Union[List[int], List[List[int]]]:
        """
        Take input text, preprocess it, tokenize it, add special 
        tokens (if specified), pad to the maximum length in the batch 
        (if a batch and specified), truncate if longer than maximum length
        (if specified), and return the ids in the tokenizer. 

        Args:
            text (str, List[str]): Input text or batch of texts. 
            add_special_tokens (bool): Whether to add eos and bos to text.
                                       Default is False
            padding (bool): Whether to pad to the maximum length 
                            in the batch. Default is False.
            truncate (bool): Whether to truncate input if it exceeds 
                             maxSequenceLength (truncate from the right). 
                             Default is False.

        Returns:
            List[int] | List[List[int]]: The encoding of the text or 
                                            batch of texts by the tokenizer.

        For example, 
            assuming word2idx = {'the':0, 'cat':1, 'eats':2, '<pad>':3,
                                '</s>':4, '<s>': 5}
            and maxSequenceLength = 5

            >>> text = "the"
            >>> tokenizer.encode(text)
            >>> [0]
            >>> text = "the cat"
            >>> tokenizer.encode(text)
            >>> [0, 1]
            >>> tokenizer.encode(text, add_special_tokens=True)
            >>> [5, 0, 1, 4]
            >>> text = ['the cat', 'the cat eats']
            >>> tokenizer.encode(text)
            >>> [[0, 1], [0, 1, 2]]
            >>> tokenizer.encode(text, padding=True)
            >>> [[0, 1, 3], [0, 1, 2]]
            >>> tokenizer.encode(text, add_special_tokens=True, padding=True)
            >>> [[5, 0, 1, 4, 3], [5, 0, 1, 2, 4]]
            >>> text = 'the cat eats the cat'
            >>> tokenizer.encode(text, truncate=True, add_special_tokens=True)
            >>> [5, 0, 1, 2, 4]

        Hint: 

            Notice the ordering of pad, truncate, and the special tokens
        """

        # TODO: Your code goes here

        # Delete the following line when implementing your function
        raise NotImplementedError

    def convert_ids_to_tokens(self,
                              ids: Union[int, List[int]]) -> Union[str, List[str]]:
        """
        Takes an id or a list of ids and returns the token(s) 
        corresponding to those ids. 

        Args:
            ids (int, List[int]): id or list of ids

        Returns:
            str | List[str]: The token or list of tokens corresponding to the ids.

        For example, 
            assuming that self.word2idx = {'the': 0, 'cat': 1, '<unk>': 2}
            >>> tokenizer.convert_ids_to_tokens(0)
            >>> "the"
            >>> tokenizer.convert_ids_to_tokens([0, 1])
            >>> ["the", "cat"]
            >>> tokenizer.convert_ids_to_tokens([0, 1, 2])
            >>> ["the", "cat", "<unk>"]
        """
        if type(ids) == int:
            return self.idx2word[ids]
        tokens = []
        for elem in ids:
            tokens.append(self.idx2word[elem])
        return tokens

    def decode(self,
               ids: Union[List[int], List[List[int]]]) -> Union[List[str], List[List[str]]]:
        """
        Takes ids (a list of ids or a batch of ids) and returns 
        the tokens corresponding to those ids (as a list of a batch). 

        Args:
            ids (List[int] | List[List[int]]): The ids as a list, batch, or tensor.

        Returns:
            List[str] | List[List[str]]: The tokens or batch of tokens. 

        For example, 
            assuming word2idx = {'the':0, 'cat':1, 'eats':2, '<pad>':3}

            >>> ids = [0, 1]
            >>> tokenizer.decode(ids)
            >>> ['the', 'cat']
            >>> ids = [[0, 1, 2], [0, 0, 0]]
            >>> tokenizer.decode(ids)
            >>> [['the', 'cat', 'eats'], ['the', 'the', 'the']]
        """
        if type(ids[0]) == int:
            return self.convert_ids_to_tokens(ids)
        words = []
        for i in ids:
            words.append(self.convert_ids_to_tokens(i))
        return words

    # TODO
    def create_vocab(self, fname: str,
                     freqThreshold: int = 30,
                     addSpecialTokens: bool = True):
        """
        Create a vocabulary for the tokenizer from a file name 
        (called fname). Only keep words that occur more than 
        freqThreshold and, if specified, add special tokens 
        (<unk>, <pad>, </s>, <s>). Make sure you use 
        the same preprocessing and tokenization scheme
        (e.g., how should you treat "this?"). Words
        should be added both to word2idx and idx2word.  

        Args:
            fname (str): Name of file to build vocabulary from.
            freqThreshold (int): Threshold of frequency for 
                                 inclusion in vocabulary. 
                                 Default is 30.
            addSpecialTokens (bool): Whether to add special tokens to 
                                     the vocabulary. Default is True.

        For example, suppose we have a file, called "cat.txt",
        with the following text (the specific ids can vary depending 
        on how you do this):

        The cat jumps over the other cat. The other cat was
        unhappy, and as we know, an unhappy cat is one 
        that never jumps over anything again.

        >>> tokenizer = Tokenizer()
        >>> tokenizer.create_vocab("cat.txt", freqThreshold=1, addSpecialTokens=False)
        >>> tokenizer.word2idx
        >>> {'the': 0, "cat": 1, "jumps": 2, "over": 3, 
        ...     "other": 4, ".": 5, "unhappy": 6, ",": 7}

        """
        # Reset (do not remove this)
        self.word2idx = {}
        self.idx2word = []

        # TODO: Your code goes here

        # Delete the following line when implementing your function
        raise NotImplementedError


if __name__ == "__main__":

    tokenizer = Tokenizer(maxSequenceLength=5)
    ##Try out your tokenizer below
    tokenizer.load_tokenizer('ToyVocab.txt')



