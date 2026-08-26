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

