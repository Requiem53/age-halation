import pickle

with open("vit_age_classifier.pkl", "rb") as f:
    header = f.read(1024)
print(header[:100])
