"""
Datasets has been designed to overcome these limitations. It frees you from memory management problems by treating datasets as memory-mapped files, and from hard drive limits by streaming the entries in a corpus.

Datasets has been designed to overcome these limitations. It frees you from memory management problems by treating datasets as memory-mapped files, and from hard drive limits by streaming the entries in a corpus.
"""




# Hugging Face's datasets library solves this in two important ways:

# 1. Memory mapping
# 2. Streaming

"""
1. 📚 Memory Mapping — "Don't load everything into RAM"

Think about a huge book.

You don't need to put the entire book on your desk to read page 500.

You can keep the book on the shelf and take out only the page you need.

That's roughly what memory mapping does.

Instead of:

20 GB dataset
       ↓
20 GB RAM ❌

it can work more like:

20 GB dataset on disk
       ↓
Only the pieces you need
       ↓
RAM

So even though the dataset is almost 20 GB, Hugging Face can access it without loading all 20 GB into RAM. The lesson demonstrates this with a PubMed dataset containing about 15.5 million rows



"""

# pile :-

# the pile is an english text corpuus that use it for training large-scale language model

# pip install azstandard

# load the dataset using the remote files 

from datasets import load_dataset
data_files = "https://the-eye.eu/public/AI/pile_preliminary_components/PUBMED_title_abstracts_2019_baseline.jsonl.zst"
pubmed_dataset = load_dataset("json", data_files=data_files, split="train")
print(pubmed_dataset)

"""

What is the Pile?

The lesson uses something called The Pile as an example.

The Pile is a huge collection of text used for training language models. It contains things like:

📄 Scientific articles
💻 GitHub code
🌐 Web text
⚖️ Legal documents
etc.

The lesson says the full Pile is around 825 GB.

Obviously, you don't want to download 825 GB just to experiment with it.

So you can stream it.


🔄 Processing streamed data

Here's a particularly useful idea.

Suppose you're training BERT and need to tokenize the text.

Normally:

Dataset
   ↓
Load everything
   ↓
Tokenize everything
   ↓
Train

With streaming:

Dataset
   ↓
Get some text
   ↓
Tokenize
   ↓
Train
   ↓
Get more text
   ↓
Tokenize
   ↓
Train
   ↓
...

For example:

tokenized_dataset = dataset.map(
    lambda x: tokenizer(x["text"])
)

So you can process huge datasets while they are being streamed.

6. 🎲 What about shuffling?

Normally, you might do:

dataset.shuffle()

But if your dataset is enormous and you're streaming it, you can't easily shuffle the entire dataset because you don't have the entire dataset in memory.

Instead, Hugging Face uses a buffer.

For example:

dataset.shuffle(buffer_size=10_000)

Think of it like this:

First 10,000 examples
        ↓
   🎲 Shuffle
        ↓
Pick one
        ↓
Bring in next example
        ↓
🎲 Shuffle again

So you're approximately shuffling the data without keeping the whole dataset in memory.

7. 🔗 Combining datasets

The lesson also shows that you can combine streamed datasets.

For example:

Medical dataset
      +
Legal dataset
      ↓
Combined dataset

Using:

interleave_datasets([
    medical_dataset,
    legal_dataset
])

You might get:

Medical
Legal
Medical
Legal
Medical
Legal
...

This is useful when you want to train a model on different kinds of data



             HUGE DATASET
                  │
          ┌───────┴────────┐
          │                │
     Fits on disk?     Too big for disk?
          │                │
          ▼                ▼
   MEMORY MAPPING       STREAMING
          │                │
          ▼                ▼
  Don't load all       Don't download all
  into RAM             at once
          │                │
          └───────┬────────┘
                  ▼
             Process data
                  │
                  ▼
              Train LLM 🤖


"""