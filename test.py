from tests.create_test import testCreateVocab
from tests.convert_tokens_test import testConvertTokens2IDs
from tests.encode_test import testEncode
from tests.preprocess_test import testPreprocess
import argparse

parser = argparse.ArgumentParser(prog='test.py', 
                                 description='Test your tokenizer')


parser.add_argument('--test',
                    default='all',
                    nargs='?',
                    choices=['create', 'convert', 'encode', 
                            'preprocess', 'all'],
                    help='run test of create, convert, encode, '\
                    'preprocess, or all (default: all)')

args = parser.parse_args()

# if args.test == 'create' or args.test == 'all':
#     print('Testing create_vocab()...')
#     testCreateVocab()
#     print('Passed!')

# if args.test == 'convert' or args.test == 'all':
#     print('Testing convert_tokens_to_ids()...')
#     testConvertTokens2IDs()
#     print('Passed!')

# if args.test == 'encode' or args.test == 'all':
#     print('Testing encode()...')
#     testEncode()
#     print('Passed!')

# if args.test == 'preprocess' or args.test == 'all':
#     print('Testing preprocess()...')
#     testPreprocess()
#     print('Passed!')
