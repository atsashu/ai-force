from transformers import pipeline

classifier = pipeline("sentiment-analysis")

res= classifier("I am not happy today")

print(res)