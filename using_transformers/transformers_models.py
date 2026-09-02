# creating a transformer model using the transformers library
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-cased")

"""
Similar to the tokenizer, the from_pretrained() method will download and cache the model data from the Hugging Face Hub. As mentioned previously, the checkpoint name corresponds to a specific model architecture and weights, in this case a BERT model with a basic architecture (12 layers, 768 hidden size, 12 attention heads) and cased inputs (meaning that the uppercase/lowercase distinction is important). There are many checkpoints available on the Hub — you can explore them here.

The AutoModel class and its associates are actually simple wrappers designed to fetch the appropriate model architecture for a given checkpoint. It’s an “auto” class meaning it will guess the appropriate model architecture for you and instantiate the correct model class. However, if you know the type of model you want to use, you can use the class that defines its architecture directly:


"""

from transformers import BertModel
model = BertModel.from_pretrained("bert-base-cased")


# Loading and saving a model

model.save_pretrained("directorry_on_my_computer")


# Encoding Text
"""
Transformer models handle text by turning the inputs into numbers. Here we will look at exactly what happens when your text is processed by the tokenizer. We’ve already seen in Chapter 1 that tokenizers split the text into tokens and then convert these tokens into numbers. We can see this conversion through a simple tokenizer:
"""


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

encoded_input = tokenizer("Hello, my dog is cute")
print(encoded_input)

# {'input_ids': [101, 8667, 117, 1139, 3676, 1110, 10509, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}

"""
We get a dictionary with the following fields:

input_ids: numerical representations of your tokens
token_type_ids: these tell the model which part of the input is sentence A and which is sentence B (discussed more in the next section)
attention_mask: this indicates which tokens should be attended to and which should not (discussed more in a bit)
We can decode the input IDs to get back the original text:
"""

tokenizer.decode(encoded_input["input_ids"])


# You’ll notice that the tokenizer has added special tokens — [CLS] and [SEP] — required by the model. Not all models need special tokens; they’re utilized when a model was pretrained with them, in which case the tokenizer needs to add them as that model expects these tokens.

# You can encode multiple sentences at once, either by batching them together (we’ll discuss this soon) or by passing a list:

encoded_input = tokenizer(["Hello, my dog is cute", "Hello, my cat is amazing"])
print(encoded_input)

# {'input_ids': [[101, 8667, 117, 1139, 3676, 1110, 10509, 102], [101, 8667, 117, 1139, 5855, 1110, 6929, 102]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]]}

# we can also ask the tokenizer to retun tensor directly from pytorch or numpy by using the return_tensors argument:

encoded_input = tokenizer(["Hello, my dog is cute", "Hello, my cat is amazing"], return_tensors="pt")
print(encoded_input)

"""
{'input_ids': tensor([[  101,  8667,   117,  1139,  3676,  1110, 10509,   102],
        [  101,  8667,   117,  1139,  5855,  1110,  6929,   102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1]])}
"""

"""
But there’s a problem: the two lists don’t have the same length! Arrays and tensors need to be rectangular, so we can’t simply convert these lists to a PyTorch tensor (or NumPy array). The tokenizer provides an option for that: padding.

Padding inputs
If we ask the tokenizer to pad the inputs, it will make all sentences the same length by adding a special padding token to the sentences that are shorter than the longest one:


"""
encoded_input = tokenizer(
    ["How are you?", "I'm fine, thank you!"], padding=True, return_tensors="pt"
)
print(encoded_input)
