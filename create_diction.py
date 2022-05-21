import os
import torch
import numpy as np
from os.path import join
import re
from tqdm import tqdm


# embeddings_dict = {}
# with open("../.vector_cache/glove.6B.300d.txt",'r',encoding="utf-8") as f:
#     for line in f:
#         values = line.split()
#         word = values[0]
#         vector = np.asarray(values[1:], "float32")
#         embeddings_dict[word] = vector
#
# np.save('text_dict', np.array(embeddings_dict))

final_vector = []
text_dict = np.load('text_dict.npy',allow_pickle=True)
text_word = np.load('vocab_dict1.npy',allow_pickle=True)
vocab_text = text_word.item()
for word,value in vocab_text.items():
    final_vector.append(text_dict.item()[word])
print(len(final_vector))
np.save('vocab_vector',np.array(final_vector))







