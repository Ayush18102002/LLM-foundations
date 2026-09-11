# Slice and Dice

"""
Most of the time, the data you work with won’t be perfectly prepared for training models. In this section we’ll explore the various features that 🤗 Datasets provides to clean up your datasets.


Slicing and dicing our data
Similar to Pandas, 🤗 Datasets provides several functions to manipulate the contents of Dataset and DatasetDict objects. We already encountered the Dataset.map() method in Chapter 3, and in this section we’ll explore some of the other functions at our disposal.

For this example we’ll use the Drug Review Dataset that’s hosted on the UC Irvine Machine Learning Repository, which contains patient reviews on various drugs, along with the condition being treated and a 10-star rating of the patient’s satisfaction.

First we need to download and extract the data, which can be done with the wget and unzip commands:
"""

# wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip"
# unzip drugsCom_raw.zip
# Since TSV is just a variant of CSV that uses tabs instead of commas as the separator, we can load these files by using the csv loading script and specifying the delimiter argument in the load_dataset() function as follows:

from datasets import load_dataset

data_files = {"train": "drugsComTrain_raw.tsv", "test": "drugsComTest_raw.tsv"}
# \t is the tab character in Python
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
#A good practice when doing any sort of data analysis is to grab a small random sample to get a quick feel for the type of data you’re working with. In 🤗 Datasets, we can create a random sample by chaining the Dataset.shuffle() and Dataset.select() functions together:

drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
# Peek at the first few examples
drug_sample[:3]
"""
{'Unnamed: 0': [87571, 178045, 80482],
 'drugName': ['Naproxen', 'Duloxetine', 'Mobic'],
 'condition': ['Gout, Acute', 'ibromyalgia', 'Inflammatory Conditions'],
 'review': ['"like the previous person mention, I&#039;m a strong believer of aleve, it works faster for my gout than the prescription meds I take. No more going to the doctor for refills.....Aleve works!"',
  '"I have taken Cymbalta for about a year and a half for fibromyalgia pain. It is great\r\nas a pain reducer and an anti-depressant, however, the side effects outweighed \r\nany benefit I got from it. I had trouble with restlessness, being tired constantly,\r\ndizziness, dry mouth, numbness and tingling in my feet, and horrible sweating. I am\r\nbeing weaned off of it now. Went from 60 mg to 30mg and now to 15 mg. I will be\r\noff completely in about a week. The fibro pain is coming back, but I would rather deal with it than the side effects."',
  '"I have been taking Mobic for over a year with no side effects other than an elevated blood pressure.  I had severe knee and ankle pain which completely went away after taking Mobic.  I attempted to stop the medication however pain returned after a few days."'],
 'rating': [9.0, 3.0, 10.0],
 'date': ['September 2, 2015', 'November 7, 2011', 'June 5, 2013'],
 'usefulCount': [36, 13, 128]}
Note that we’ve fixed the seed in Dataset.shuffle() for reproducibility purposes. Dataset.select() expects an iterable of indices, so we’ve passed range(1000) to grab the first 1,000 examples from the shuffled dataset. From this sample we can already see a few quirks in our dataset:

The Unnamed: 0 column looks suspiciously like an anonymized ID for each patient.
The condition column includes a mix of uppercase and lowercase labels.
The reviews are of varying length and contain a mix of Python line separators (\r\n) as well as HTML character codes like &039;.
Let’s see how we can use 🤗 Datasets to deal with each of these issues. To test the patient ID hypothesis for the Unnamed: 0 column, we can use the Dataset.unique() function to verify that the number of IDs matches the number of rows in each split:
"""

for split in drug_dataset.keys():
    assert len(drug_dataset[split]) == len(drug_dataset[split].unique("Unnamed: 0"))
 # This seems to confirm our hypothesis, so let’s clean up the dataset a bit by renaming the Unnamed: 0 column to something a bit more interpretable. We can use the DatasetDict.rename_column() function to rename the column across both splits in one go:

drug_dataset = drug_dataset.rename_column(
    original_column_name="Unnamed: 0", new_column_name="patient_id"
)
print(drug_dataset)
"""
DatasetDict({
    train: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount'],
        num_rows: 161297
    })
    test: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount'],
        num_rows: 53766
    })
})
"""
# Try it out! Use the Dataset.unique() function to find the number of unique drugs and conditions in the training and test sets.

# Next, let’s normalize all the condition labels using Dataset.map(). As we did with tokenization in Chapter 3, we can define a simple function that can be applied across all the rows of each split in drug_dataset:


def lowercase_condition(example):
    return {"condition": example["condition"].lower()}


drug_dataset.map(lowercase_condition)
"""
AttributeError: 'NoneType' object has no attribute 'lower'
Oh no, we’ve run into a problem with our map function! From the error we can infer that some of the entries in the condition column are None, which cannot be lowercased as they’re not strings. Let’s drop these rows using Dataset.filter(), which works in a similar way to Dataset.map() and expects a function that receives a single example of the dataset. Instead of writing an explicit function like:
"""

def filter_nones(x):
    return x["condition"] is not None
#and then running drug_dataset.filter(filter_nones), we can do this in one line using a lambda function. In Python, lambda functions are small functions that you can define without explicitly naming them. They take the general form:


# lambda <arguments> : <expression>
# where lambda is one of Python’s special keywords, <arguments> is a list/set of comma-separated values that define the inputs to the function, and <expression> represents the operations you wish to execute. For example, we can define a simple lambda function that squares a number as follows:


lambda x : x * x
#To apply this function to an input, we need to wrap it and the input in parentheses:


(lambda x: x * x)(3)

#9
#Similarly, we can define lambda functions with multiple arguments by separating them with commas. For example, we can compute the area of a triangle as follows:


(lambda base, height: 0.5 * base * height)(4, 8)

# 16.0
# Lambda functions are handy when you want to define small, single-use functions (for more information about them, we recommend reading the excellent Real Python tutorial by Andre Burgaud). In the 🤗 Datasets context, we can use lambda functions to define simple map and filter operations, so let’s use this trick to eliminate the None entries in our dataset:


drug_dataset = drug_dataset.filter(lambda x: x["condition"] is not None)
# With the None entries removed, we can normalize our condition column:

drug_dataset = drug_dataset.map(lowercase_condition)
# Check that lowercasing worked
drug_dataset["train"]["condition"][:3]

# ['left ventricular dysfunction', 'adhd', 'birth control']
# It works! Now that we’ve cleaned up the labels, let’s take a look at cleaning up the reviews themselves.

#  Creating new columns
# Whenever you’re dealing with customer reviews, a good practice is to check the number of words in each review. A review might be just a single word like “Great!” or a full-blown essay with thousands of words, and depending on the use case you’ll need to handle these extremes differently. To compute the number of words in each review, we’ll use a rough heuristic based on splitting each text by whitespace.

# Let’s define a simple function that counts the number of words in each review:


def compute_review_length(example):
    return {"review_length": len(example["review"].split())}
# nlike our lowercase_condition() function, compute_review_length() returns a dictionary whose key does not correspond to one of the column names in the dataset. In this case, when compute_review_length() is passed to Dataset.map(), it will be applied to all the rows in the dataset to create a new review_length column:


drug_dataset = drug_dataset.map(compute_review_length)
# Inspect the first training example
drug_dataset["train"][0]
"""
{'patient_id': 206461,
 'drugName': 'Valsartan',
 'condition': 'left ventricular dysfunction',
 'review': '"It has no side effect, I take it in combination of Bystolic 5 Mg and Fish Oil"',
 'rating': 9.0,
 'date': 'May 20, 2012',
 'usefulCount': 27,
 'review_length': 17}
 """
# As expected, we can see a review_length column has been added to our training set. We can sort this new column with Dataset.sort() to see what the extreme values look like:

drug_dataset["train"].sort("review_length")[:3]
"""
{'patient_id': [103488, 23627, 20558],
 'drugName': ['Loestrin 21 1 / 20', 'Chlorzoxazone', 'Nucynta'],
 'condition': ['birth control', 'muscle spasm', 'pain'],
 'review': ['"Excellent."', '"useless"', '"ok"'],
 'rating': [10.0, 1.0, 6.0],
 'date': ['November 4, 2008', 'March 24, 2017', 'August 20, 2016'],
 'usefulCount': [5, 2, 10],
 'review_length': [1, 1, 1]}
 """
# As we suspected, some reviews contain just a single word, which, although it may be okay for sentiment analysis, would not be informative if we want to predict the condition.

# An alternative way to add new columns to a dataset is with the Dataset.add_column() function. This allows you to provide the column as a Python list or NumPy array and can be handy in situations where Dataset.map() is not well suited for your analysis.

# Let’s use the Dataset.filter() function to remove reviews that contain fewer than 30 words. Similarly to what we did with the condition column, we can filter out the very short reviews by requiring that the reviews have a length above this threshold:


drug_dataset = drug_dataset.filter(lambda x: x["review_length"] > 30)
print(drug_dataset.num_rows)

 # {'train': 138514, 'test': 46108}
# As you can see, this has removed around 15% of the reviews from our original training and test sets.

# Try it out! Use the Dataset.sort() function to inspect the reviews with the largest numbers of words. See the documentation to see which argument you need to use sort the reviews by length in descending order.

# he last thing we need to deal with is the presence of HTML character codes in our reviews. We can use Python’s html module to unescape these characters, like so:

import html

text = "I&#039;m a transformer called BERT"
html.unescape(text)

"I'm a transformer called BERT"
# We’ll use Dataset.map() to unescape all the HTML characters in our corpus:


drug_dataset = drug_dataset.map(lambda x: {"review": html.unescape(x["review"])})
# As you can see, the Dataset.map() method is quite useful for processing data — and we haven’t even scratched the surface of everything it can do!

# the Map() method's superpowers

# the Dataset.map() method takes a batched argument that if set to true, cause it to send a batch 
# the mao function at once ( the batch size is configurable but default to 1,000).
# the previous map functionthat unescaped all the HTML took a bit of time to run.

new_drug_dataset = drug_dataset.map(
    lambda x : {"review": [html.unescape(o) for o in x["review"]]}, batched = True
)

# Using Dataset.map() with batched=True will be essential to unlock the speed of the “fast” tokenizers that we’ll encounter

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(example):
    return tokenizer(example["review"],truncation=True)


%time tokenized_dataset = drug_dataset.map(tokenize_function, batched=True)

"""
This means that using a fast tokenizer with the batched=True option is 30 times faster than its slow counterpart with no batching — this is truly amazing! That’s the main reason why fast tokenizers are the default when using AutoTokenizer (and why they are called “fast”). They’re able to achieve such a speedup because behind the scenes the tokenization code is executed in Rust, which is a language that makes it easy to parallelize code execution.

Parallelization is also the reason for the nearly 6x speedup the fast tokenizer achieves with batching: you can’t parallelize a single tokenization operation, but when you want to tokenize lots of texts at the same time you can just split the execution across several processes, each responsible for its own texts.

Dataset.map() also has some parallelization capabilities of its own. Since they are not backed by Rust, they won’t let a slow tokenizer catch up with a fast one, but they can still be helpful (especially if you’re using a tokenizer that doesn’t have a fast version). To enable multiprocessing, use the num_proc argument and specify the number of processes to use in your call to Dataset.map():

"""

slow_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=False)


def slow_tokenize_function(examples):
    return slow_tokenizer(examples["review"], truncation=True)


tokenized_dataset = drug_dataset.map(slow_tokenize_function, batched=True, num_proc=8)


