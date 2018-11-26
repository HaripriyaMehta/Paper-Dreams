import spacy
from collections import Counter 
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim import corpora, models, similarities
import nltk
from nltk.corpus import gutenberg
from urllib.request import urlopen
from nltk.tokenize import sent_tokenize
from spacy.lang.en.stop_words import STOP_WORDS
import enchant
import math
import pandas as pd 
nlp = spacy.load('en_vectors_web_lg')
#ids = ["35997", "18735", "582", "517", "20437", "18546", "28", "35688", "22816", "36039", "19998"]
d = enchant.Dict("en_US")
new_list = []
for i in range(20000, 21000):
    try:
        print(i)
        url = "https://www.gutenberg.org/cache/epub/" + str(i) +"/pg" + str(i) + ".txt" 
        raw = urlopen(url).read().decode('utf-8')
        text = nltk.Text(nltk.word_tokenize(raw))
        raw = raw.split("*** START: FULL LICENSE ***", 1)[0]
        sent_tokenize_list = [word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(raw)) if pos[0] == 'N']
        new_list.extend(sent_tokenize_list)
    except:
        pass

alloccur = []
for i in new_list:
    if i not in STOP_WORDS and i!= "" and d.check(i):
        alloccur.append(i.strip())


Counter = Counter(alloccur) 
most_occur = Counter.most_common(100) 

alloccur = []
for i in most_occur:
    alloccur.append(i[0])

col_names =  ['token1', 'similarity', 'token2']
df = pd.DataFrame(columns = col_names)
tokens = nlp(" ".join(alloccur))
for token1 in tokens:
    for token2 in tokens:
        simil = round(token1.similarity(token2),1)*10
        if simil>0.0 and simil!=1.0:
            df.loc[len(df)] = [token1.text, int(simil),token2.text]

df.to_csv("conceptnet.csv")
df2 =pd.read_csv("conceptnet.csv")
df1 = set(df2.loc[(df2['token1'] == "feet") & (df2['similarity'] == 3), 'token2'].tolist())
print(df1)
