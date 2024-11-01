import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math, string, re

from sklearn.svm import LinearSVC, SVC
from string import punctuation
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix
from time import time
import pickle

nltk.download('stopwords')
SW = stopwords.words("english")
PUNCT = list(punctuation)
D = 7

# POS tag encoding as per the ConLL2003 dataset
pos_tag_mapping = {'"' : 0, '''''' : 1, '#' : 2, '$' : 3, '(' : 4, ')' : 5, ',' : 6, '.' : 7, ':' : 8, '``' : 9, 'CC' : 10, 'CD' : 11, 'DT' : 12,
                   'EX' : 13, 'FW' : 14, 'IN' : 15, 'JJ' : 16, 'JJR' : 17, 'JJS' : 18, 'LS' : 19, 'MD' : 20, 'NN' : 21, 'NNP' : 22, 'NNPS' : 23,
                   'NNS' : 24, 'NN|SYM' : 25, 'PDT' : 26, 'POS' : 27, 'PRP' : 28, 'PRP$' : 29, 'RB' : 30, 'RBR' : 31, 'RBS' : 32, 'RP' : 33,
                   'SYM' : 34, 'TO' : 35, 'UH' : 36, 'VB' : 37, 'VBD' : 38, 'VBG' : 39, 'VBN' : 40, 'VBP' : 41, 'VBZ' : 42, 'WDT' : 43,
                   'WP' : 44, 'WP$' : 45, 'WRB' : 46}

# Function to get the POS tag of the given word as per the ConLL2003 Dataset
def get_pos_tag(word):
    # Tokenize the word
    tokens = word_tokenize(word)

    # Get the POS tag
    pos_tags = pos_tag(tokens)

    # Return the POS tag of the first token
    return pos_tags[0][1] if pos_tags else None

#Given a word, the function extracts the necessary features.
def getWordVectors(w, pos_tag, position):
    res = [0 for _ in range(D)]

    # If the word starts with a caps letter
    if w[0].isupper():
        res[0] = 1
    else:
        res[0] = 0

    # If the word is completely caps
    if w.isupper():
        res[1] = 1
    else:
        res[1] = 0

    # Length of the word
    res[2] = len(w)

    # If the word is a stop word
    if w.lower() in SW:
        res[3] = 1
    else:
        res[3] = 0

    # If the word is a punctuation
    if w in PUNCT:
        res[4] = 1
    else:
        res[4] = 0

    # The POS tag of the word
    res[5] = pos_tag

    # The relative position of the word in the sentence
    res[6] = position

    #Converting to a float array
    res = np.asarray(res, dtype = np.float32)
    return res

# This function gives the feature vectors for the sentences of a dataset
def getVectors(s):
    words = [] # To store the tokens of the sentence
    nei_tags = [] # To store their NEI tags
    vectors = [] # To store the features of the words

    for d in data[s]:
        sen = d["tokens"] # The sentence
        l = len(sen)

        for i in range(l):
            words.append(sen[i]) # Appending the word to the words array
            vecs = getWordVectors(sen[i], d["pos_tags"][i], float(i/l)) # Getting the features of the word
            vectors.append(vecs) # Appending the word features to the vectors array

            # Getting the NEI tags and appending it to the nei_tags array
            n = d["ner_tags"][i]
            if (n == 0):
                nei_tags.append(0)
            else:
                nei_tags.append(1)

    return words, vectors, nei_tags

# This function deals with the period in the sentence
def removePeriod(word):
    # Exception words are the list of words for which we don't want to remove the period from the root word
    exception_words = ["mr.", "mrs.", "ms.", "dr."]
    if word.lower() in exception_words:
        return word

    # Otherwise, separate the word and the punctuation
    res = ""
    for i in range(len(word)):
        if (word[i] != '.'):
            res += word[i]
    return res    

# Perform tokenization and pre-processing
def preprocess_sent(sent):
    words = sent.split()
    if (words[-1][-1] == '.'):
        words[-1] = words[-1][:-1]
        words.append('.')
    proc_words = []
    exception_punc = ['\'', '-', '`', '/', '\\', '$', '.']
    n = len(words)
    for i in range(n):
        w = words[i]
        if (w == " "):
            continue
        else: 
            # Replace formatted numbers with the same number but without commas
            w = re.sub(r'(\d{1,3})(,\d{3})+', lambda m: m.group(0).replace(',', ''), w)
            if (w[0] == "$"):
                w = w[1:]
            w = removePeriod(w)
            s = 0
            n = len(w)
            for i in range(n):
                if (w[i].isalnum()):
                    continue
                elif (w[i] not in exception_punc):
                    proc_words.append(w[s:i])
                    proc_words.append(w[i])
                    s = i + 1
            if (s < n):
                proc_words.append(w[s:])
    
    if (proc_words and proc_words[-1].isalnum()):
        proc_words.append(".")
    if proc_words:
        proc_words[0] = proc_words[0].capitalize()
    
    # Remove empty strings from the list
    proc_words = [ele for ele in proc_words if ele != '']
    
    return proc_words

svm_model = pickle.load(open('nei_svm.pkl','rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


# Streamlit app
# Set page config with title and layout
st.set_page_config(page_title="NEI SVM")
st.title("Named Entity Identifier")


sentence = st.text_input("Enter the sentence:")
if st.button("Analyze"):
    if sentence:
        words = preprocess_sent(sentence)
        l = len(words)
        f = []
        
        for i in range(l):
            pos = pos_tag_mapping[get_pos_tag(words[i])]  # Get the POS tag encoding
            f.append(getWordVectors(words[i], pos, float(i / l)))  # Get the word vector
        
        f_scaled = scaler.transform(f)
        nei_tags = svm_model.predict(f_scaled)

        # Modify NEI tags based on conditions
        for i in range(len(words)):
            if (i >= 1 and words[i] == 'of' and nei_tags[i - 1] == 1):
                nei_tags[i] = 1
        
        output = " ".join(f"{words[i]}_{int(nei_tags[i])}" for i in range(len(words)))
        st.write("Output:")
        st.write(output)
    else:
        st.write("Please enter a sentence.")
