#!/usr/bin/env python
# coding: utf-8

# In[2]:


import nltk
import random
import numpy as np
from bs4 import BeautifulSoup


# In[3]:


positive_reviews = BeautifulSoup(open('electronics/positive.review').read(), features="lxml")
positive_reviews = positive_reviews.findAll('review_text')


# In[13]:


trigrams = {} # ex: {('i', 'this'): ['purchased', 'bought', 'recommend', 'use'], (x, y):[abc, def], ...}
for review in positive_reviews:
    s = review.text.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    for i in range(len(tokens) - 2):
        k = (tokens[i], tokens[i+2])
        if k not in trigrams:
            trigrams[k] = []
        trigrams[k].append(tokens[i+1]) # key = tuple of before-word and after-word, val = list of between words


# In[38]:


print(len(tokens), len(positive_reviews))
print(len(trigrams))
# trigrams


# In[19]:


trigrams_probabilities = {} # key = before & after word tuple, val = dict of middlewords:probabilities
# ex: {('i', 'this'): {'purchased': 0.12422360248447205, 'bought': 0.3105590062111801},...}
for k, words in trigrams.items():
    if len(set(words)) > 1:
        d = {}
        n = 0
        for w in words:
            if w not in d:
                d[w] = 0
            d[w] += 1
            n += 1
        for w, c in d.items():
            d[w] = float(c) / n
        trigrams_probabilities[k] = d


# In[23]:


# trigrams_probabilities


# In[25]:


def random_sample(d):
    r = random.random()
    cumulative = 0
    for w, p in d.items():
        cumulative += p
        if r < cumulative:
            return w


# In[31]:


# test the random_sample() function.  select a review, make substitution, print both for comparison
def test_spinner():
    review = random.choice(positive_reviews)
    s = review.text.lower()
    print('Original: ', s)
    tokens = nltk.tokenize.word_tokenize(s)
    for i in range(len(tokens) -2):
        if random.random() < 0.2:
            k = (tokens[i], tokens[i+2])
            if k in trigrams_probabilities:
                w = random_sample(trigrams_probabilities[k])
                tokens[i+1] = w
    print('Spun: ')
    print(' '.join(tokens).replace(' .','.').replace(" ''", "''").replace(' ,',',').replace('$ ','$').replace(' !','!'))


# In[36]:


test_spinner()


# In[ ]:




