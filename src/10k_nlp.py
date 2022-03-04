#Ref #https://statsmaths.github.io/stat289-f18/solutions/tutorial19-gensim.html

import pandas as pd
import gensim
from gensim import corpora, matutils, models, similarities
from gensim.similarities.docsim import MatrixSimilarity
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import re
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# nltk.download('stopwords')


# nltk.download('wordnet')
# nltk.download('omw-1.4')

df = pd.read_excel('./data/sp500_fi_ratio_full_.xlsx', engine='openpyxl')
df = df.drop(['Unnamed: 0', 'id'], axis=1)
df = df.fillna("0")

data_text = df[['Text']]
data_text['index'] = data_text.index
documents = data_text

print(len(documents))
print(documents[:5])

stemmer = SnowballStemmer(language="english")

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

doc_sample = documents[documents['index'] == 310].values[0][0]
print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))

processed_docs = documents['Text'].map(preprocess)
processed_docs[:10]


dictionary = gensim.corpora.Dictionary(processed_docs)
count = 0

for k, v in dictionary.iteritems():
    # print(k, v)
    count += 1
    if count > 10:
        break

dictionary.filter_extremes(no_below=20, no_above=0.5, keep_n=100000)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
len(bow_corpus)


######################################################

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
# corpus_tfidf = tfidf[bow_corpus[0]]

from pprint import pprint
for doc in corpus_tfidf:
    pprint(doc)
    break




######################################################


tf_array = matutils.corpus2dense(bow_corpus, num_terms=len(dictionary.token2id))
tf_array = np.transpose(tf_array)
features = [dictionary[vocab] for vocab in dictionary]
df_tf = pd.DataFrame(tf_array, columns=features)
print(df_tf.shape)

tfidf_array = matutils.corpus2dense(corpus_tfidf, num_terms=len(dictionary.token2id))
tfidf_array = np.transpose(tfidf_array)
features = [dictionary[vocab] for vocab in dictionary]
df_tfidf = pd.DataFrame(tfidf_array, columns=features)
# print(df_tfidf.shape)

######################################################
######################################################

#
# # bow_corpus[310]
# #
# bow_doc_4310 = bow_corpus[310]
# for i in range(len(bow_doc_4310)):
#     print("Word {} (\"{}\") appears {} time.".format(bow_doc_4310[i][0],
#                                                dictionary[bow_doc_4310[i][0]],
#                                                      bow_doc_4310[i][1]))
#
#
#




# lda_model = gensim.models.LdaModel(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
lda_model = gensim.models.LdaModel(bow_corpus, num_topics=10, id2word=dictionary, passes=2)
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

# print(lda_model.print_topics())

lda_model_tfidf = gensim.models.LdaModel(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2)
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))


# print(lda_model_tfidf.print_topics())

##########################################################################################################

cm = CoherenceModel(model=lda_model, texts=processed_docs, coherence='u_mass')
coherence = cm.get_coherence()
print("Coherence:",coherence)
# coherencesT.append(coherence)
print('Perplexity: ', lda_model.log_perplexity(bow_corpus),'\n\n')
# perplexitiesT.append(lda4.log_perplexity(corpus))


li_topic =[]
topics = lda_model.print_topics(num_words=10)
for topic in topics:
    print(topic)
#     li_topic.append(topic)
#



print(lda_model.print_topics())


##########################################################################################################

num_topics = 10
texts = processed_docs


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    print(compute_coherence_values)
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        print("Processing...: ", num_topics, "/", limit)
        model=gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=10)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='u_mass')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

start=2; limit=40; step=6;

# model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus_tfidf, texts=texts, start=start, limit=limit, step=step)



# # Show graph
# import matplotlib.pyplot as plt
#
# x = range(start, limit, step)
# plt.plot(x, coherence_values)
# plt.xlabel("Num Topics")
# plt.ylabel("Coherence score")
# plt.legend(("coherence_values"), loc='best')
# plt.show()



##########################################################################################################

df

df_tf
df_tfidf


data2 = pd.read_excel("./data/topic_li.xlsx")
print(data2)
topic_cal = data2.columns.tolist()




for cal in topic_cal:
    topic = cal
    wordlist = data2[topic].dropna().tolist()
    cols = df_tfidf.columns.tolist()
    intersect = []

    for c in cols:
        if c in wordlist:
            intersect.append(c)

    df_cp = df_tfidf[intersect]
    df_= df_cp.sum(axis=1)
    df[topic] = df_


df__ = df.drop(['Text'], axis = 1)

df.shape

df__.to_excel('./data/fi_bertopic.xlsx')

##############################
########## BERTopic ##########
##############################


from bertopic import BERTopic


stop_words = nltk.corpus.stopwords.words('english')

#item_report_factors_risk
new_words=('item', 'report', 'factors')

for i in new_words:
    stop_words.append(i)


bert_df = documents

def clean_text(x):
  x = str(x)
  x = x.lower()
  x = re.sub(r'#[A-Za-z0-9]*', ' ', x)
  x = re.sub(r'https*://.*', ' ', x)
  x = re.sub(r'@[A-Za-z0-9]+', ' ', x)
  tokens = word_tokenize(x)
  x = ' '.join([w for w in tokens if not w.lower() in stop_words])
  x = re.sub(r'[%s]' % re.escape('!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~“…”’'), ' ', x)
  x = re.sub(r'\d+', ' ', x)
  x = re.sub(r'\n+', ' ', x)
  x = re.sub(r'\s{2,}', ' ', x)
  return x


bert_df = bert_df.Text.apply(clean_text)




topic_model = BERTopic()
# topics, _ = topic_model.fit_transform(documents['Text'].tolist())
topics, _ = topic_model.fit_transform(bert_df.tolist())

new_topics, new_probs = topic_model.reduce_topics(bert_df.tolist(), topics, _, nr_topics="auto")
prob_df = pd.DataFrame(new_probs, columns=['probs'])



# prob_df.to_csv("topic propotion per document.csv")


freq =  topic_model.get_topic_info()
freq

topic_li = []
for i in range(0, 29, 1):
    topic_model.get_topic(1)
    topic_li.append(topic_model.get_topic(i))

topic_li_df = pd.DataFrame(topic_li)
topic_li_df.to_excel("./data/topic_li.xlsx")

ddd = topic_model.get_topic(1)
ddd
topic_model.visualize_barchart(n_words=10)