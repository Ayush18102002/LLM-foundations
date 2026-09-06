"""
Tokenizers
tokenizers are ont of the core components of the NLP pipeline
it translate text into data that can be processed by the model
model can only processed numbers and tokenizers need to convert our text into numerical data
"""

tokenized_text = "Jim bought 300 shares of Acme Corp. in 2006"
print(tokenized_text.split())

# output - ['Jim', 'bought', '300', 'shares', 'of', 'Acme', 'Corp.', 'in', '2006']

# One way to reduce the amount of unknown tokens is to go one level deeper, using a character-based tokenizer.


"""
Character-based tokenizers split the text into characters, rather than words. This has two primary benefits:

The vocabulary is much smaller.
There are much fewer out-of-vocabulary (unknown) tokens, since every word can be built from characters.
But here too some questions arise concerning spaces and punctuation:
"""

"""
subword -based tokenization
Subword tokenization algorithms rely on the principle that frequently used words should not be split into smaller subwords, but rare words should be decomposed into meaningful subwords.

"""
# loading and saving 
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

#  Similar to AutoModel, the AutoTokenizer class will grab the proper tokenizer class in the library based on the checkpoint name, and can be used directly with any checkpoint:

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

tokenizer("Using a Transformer network is simple")

"""
output: 
{'input_ids': [101, 7993, 170, 11303, 1200, 2443, 1110, 3014, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
"""
tokenizer.save_pretrained("directory_on_my_computer")

"""
Translating text to numbers is known as encoding. Encoding is done in a two-step process: the tokenization, followed by the conversion to input IDs.

As we’ve seen, the first step is to split the text into words (or parts of words, punctuation symbols, etc.), usually called tokens. There are multiple rules that can govern that process, which is why we need to instantiate the tokenizer using the name of the model, to make sure we use the same rules that were used when the model was pretrained.

The second step is to convert those tokens into numbers, so we can build a tensor out of them and feed them to the model. To do this, the tokenizer has a vocabulary, which is the part we download when we instantiate it with the from_pretrained() method. Again, we need to use the same vocabulary used when the model was pretrained.


"""
# The tokenization process is done by the tokenize() method of the tokenizer:

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
sequence = " using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)
print(tokens)

# output - ['using', 'a', 'Trans', '##former', 'network', 'is', 'simple']


# This tokenizer is a subword tokenizer: it splits the words until it obtains tokens that can be represented by its vocabulary. That’s the case here with transformer, which is split into two tokens: trans and ##former.

# from token to input IDs
# The conversion to input IDs is handled by the convert_tokens_to_ids() tokenizer method:

ids = tokenizer.converts_token_to_ids(tokens)
print(ids)

# output - [1606, 170, 13809, 23763, 2443, 1110, 3014]

# decoding 

# decoding is going the other way around : from vocabulary indices, we want to get a string . thhis can be done with the decode() method as follows:

decode_string = tokenizer.decode([1606, 170, 13809, 23763, 2443, 1110, 3014])
print(decode_string)