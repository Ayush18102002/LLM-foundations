from transformers import pipeline

"""
classifier = pipeline("sentiment-analysis")

#result = classifier("I've been waiting for a HuggingFace course my whole life.")
#print(result)

# as per the result is says [{'label': 'POSITIVE', 'score': 0.9598049521446228}]
# the above sentence os positive 


# same as we can pass several  sentences 

result1 = classifier([
    "I've been waiting for a HuggingFace course my whole life.",
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!"
])
print(result1)


# [{'label': 'POSITIVE', 'score': 0.9598049521446228}, {'label': 'POSITIVE', 'score': 0.9598049521446228}, {'label': 'NEGATIVE', 'score': 0.9994558691978455}]
"""

"""
classifier2 = pipeline("zero-shot-classification")

result2 = classifier2(
    " this is a course abput the Transformers libraray",
    candidate_labels = ["education", "politics","business"]
)
print(result2)

# output
#{'sequence': ' this is a course abput the Transformers libraray', 
# 'labels': ['education', 'business', 'politics'], 
# 'scores': [0.5757914185523987, 0.32652145624160767, 0.09768706560134888]}
"""

# text generation

generator = pipeline("text-generation")

result3 = generator("In this course, we will teach you how to")

print(result3)
"""
generator2 = pipeline("text-generation",model="orcarouter/Qwen3.8-27B-Uncensored-MLX")
result4 = generator2(
        " In this course, we will teach you how to",
        max_length = 30,
        num_return_sequence=2,
)
print(result4)
"""

 # MASK-FILLING

# The idea of this tasks is to fill in the blanks in a given text.

unmasker = pipeline("fill-mask")
return2 = unmasker("this course will teach you all about <mask> models.", top_k=2) # top_k=2 this agumentations controls how many possibilities you want to be displayed
print(return2)

# NAMED ENTITY RECOGNITION

# NER is a task where the model has to find which part of the input text correspond to entities such as persons,       locations, or organizations

# ner = pipeline("ner", grouped_entities=True) # grouped_entitiesb= True regroup the part of the sentence that correspond to the same entity

# result3 = ner("my name is sylvain and i work at google in new jersey")
print(result3)


# Encoder-Decoder architecture

# transformers are nothing but a language model. that trained on Raw Text this is also called 
# casual langiage modeling because the output depends on the past and present input
# but not the future ones. 

# Transformers are big models we can improve the accuracy of the model through increase the model's size as well as the ampunt of data they are pretrained on.

# Encoder(left) -  the encoder receives an input and build a representation of it(its features). this means model is optimized to acquire understanding from the input

# Decoder (right): The decoder uses the encoder’s representation (features) along with other inputs to generate a target sequence. This means that the model is optimized for generating outputs.

# Encoder-only models: Good for tasks that require understanding of the input, such as sentence classification and named entity recognition.

# Decoder-only models: Good for generative tasks such as text generation.

# Encoder-decoder models or sequence-to-sequence models: Good for generative tasks that require an input, such as translation or summarization.

# ATTENTION LAYERS

# A key feature of Transformer models is that they are built with special layers called attention layers. In fact, the title of the paper introducing the Transformer architecture was “Attention Is All You Need”! We will explore the details of attention layers later in the course

# There are two main approaches for training a transformer model:

# Masked language modeling (MLM): Used by encoder models like BERT, this approach randomly masks some tokens in the input and trains the model to predict the original tokens based on the surrounding context. This allows the model to learn bidirectional context (looking at words both before and after the masked word).

# Causal language modeling (CLM): Used by decoder models like GPT, this approach predicts the next token based on all previous tokens in the sequence. The model can only use context from the left (previous tokens) to predict the next token.

# transcribe the video 
from transformers import pipeline

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
result5 = transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
print(result5)

# output - {'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}

