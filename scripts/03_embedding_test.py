import numpy as np
from sentence_transformers import SentenceTransformer

# This loads the brain that knows how to map meaning.
model = SentenceTransformer('all-MiniLM-L6-v2')

s1 = "The Thinkpad is a powerful laptop."
s2 = "The Computer is very fast."
s3 = "I like eating apples."
s4 = "I like using my computer to eat apples."

#Encode them into math
#This results a list containing two vectors
vectors = model.encode([s1, s2, s3, s4])

A = vectors[0]
B = vectors[1]
C = vectors[2]
D = vectors[3]

score_s1_s2 = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
score_s1_s3 = np.dot(A, C) / (np.linalg.norm(A) * np.linalg.norm(C))
score_s1_s4 = np.dot(A, D) / (np.linalg.norm(A) * np.linalg.norm(D))
print(score_s1_s2)
print(score_s1_s3)
print(score_s1_s4)
print(f"s1 vs s4 (The Hybrid): {score_s1_s4}")