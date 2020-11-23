import numpy as np
import pandas as pd
import re
from collections import Counter
import sys

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from joblib import dump, load

def clean_dataframe(df):
    df.drop('id', axis = 1, inplace = True)
    df.dropna(subset = ['text'], inplace = True)
    df['author'].fillna('author unknown', inplace = True)
    df['title'].fillna('no title', inplace = True)
    df.reset_index(drop = True, inplace = True)
    return

def regex_clean(x):
    if pd.isna(x):
        return np.nan
    else:
        return re.sub('[^a-zA-Z0-9.]', ' ', x)
    
def high_level_text_summary(df, col):
    df['_'.join([col, 'clean'])] = df[col].apply(regex_clean)
    df['_'.join([col, 'clean'])] = df['_'.join([col, 'clean'])].str.lower()
    df['_'.join([col, 'tot_chars'])] = df['_'.join([col, 'clean'])].str.len()
    df['_'.join([col, 'tot_words'])] = df['_'.join([col, 'clean'])].str.split().str.len()
    df['_'.join([col, 'tot_sentences'])] = df['_'.join([col, 'clean'])].str.split('.').str.len()
    df['_'.join([col, 'tot_line_breaks'])] = df['_'.join([col, 'clean'])].str.split('\n').str.len()
    return df 

def spacy_POS(df, col, nlp):

    n_batch = 10
    pos_summary = []
    # Where texts is an iterator over your documents (unicode)
    for docs in nlp.pipe(df['_'.join([col, 'clean'])], batch_size=n_batch, n_threads=3):
        pos = [docs[i].pos_ for i in np.arange(len(docs))]
        pos_summary.append(dict(Counter(pos)))
        
    df_orig = pd.DataFrame([list(pos_summary[0].values())], columns = list(pos_summary[0].keys()))
    for idx in np.arange(1, len(pos_summary)):
        df_join = pd.DataFrame([list(pos_summary[idx].values())], columns = list(pos_summary[idx].keys()))
        df_orig = pd.concat([df_orig, df_join], ignore_index = True)
    df_orig.fillna(0, inplace = True)
    df_pos_norm = df_orig.div(df_orig.sum(axis=1), axis=0)
    df_pos_norm = df_pos_norm.add_prefix(col + '_')
    return df_pos_norm

def spacy_NER(df, col, nlp):

    n_batch = 10
    ner_summary = []
    # Where texts is an iterator over your documents (unicode)
    for docs in nlp.pipe(df['_'.join([col, 'clean'])], batch_size=n_batch, n_threads=3):
        if len(docs) == 0:
            list_ner = [1, 1]
            ner_summary.append(dict(Counter(list_ner)))
        else:
            list_ner = [docs.ents[j].label_ for j in np.arange(len(docs.ents))]
            ner_summary.append(dict(Counter(list_ner)))

    df_orig = pd.DataFrame([list(ner_summary[0].values())], columns = list(ner_summary[0].keys()))
    for idx in np.arange(1, len(ner_summary)):
        df_join = pd.DataFrame([list(ner_summary[idx].values())], columns = list(ner_summary[idx].keys()))
        df_orig = pd.concat([df_orig, df_join], ignore_index = True)
    df_orig.fillna(0, inplace = True)
    df_ner_norm = df_orig.div(df_orig.sum(axis=1), axis=0)
    df_ner_norm.fillna(0, inplace = True)
    df_ner_norm = df_ner_norm.add_prefix(col + '_')
    return df_ner_norm

def sentiment_analysis(df, col, analyser):
    score = []
    for sentence in df[col].values:
        output = analyser.polarity_scores(sentence)
        output = list(output.values())[-1]
        score.append(output)
    df['_'.join([col, 'sentiment'])] = score
    return df

def lda_analysis(df, col, data_nature):
    if data_nature == 'train':
        bow = CountVectorizer(max_df = 0.9, min_df = 0.1)
        text_bow = bow.fit_transform(df['_'.join(['clean'])])
        lda = LatentDirichletAllocation()
        res = lda.fit_transform(text_bow)
        dump(bow, './param_store/count_vec.joblib')
        dump(lda, './param_store/lda.joblib')
    else:
        bow = load('./param_store/count_vec.joblib')
        lda = load('./param_store/lda.joblib')
        text_bow = bow.transform(df['_'.join(['clean'])])
        res = lda.transform(text_bow)
    df_lda = pd.DataFrame(res, columns = ['_'.join(['lda_component',str(i)]) for i in np.arange(10)])
    return df_lda

if __name__ == '__main__':
    data_file = sys.argv[1]
    data_nature = sys.argv[2]
    data = pd.read_csv(data_file)
    clean_dataframe(data)
    
    for col in ['text', 'title']:
        data = high_level_text_summary(data, col)

    if data_nature == 'train':
        tf = TfidfVectorizer(max_features = 1000, ngram_range = (1, 2), stop_words = 'english')
        tf.fit(data['text_clean'].dropna())
        dump(tf, './param_store/tfidf_model.joblib')
    else:
        tf = load('./param_store/tfidf_model.joblib')
    res = tf.transform(data['text_clean'].dropna())
    tfidf_df = pd.DataFrame(res.toarray(), columns = tf.get_feature_names())

    nlp = spacy.load('en_core_web_sm', disable = ['ner', 'parser'])
    pos_title = spacy_POS(data, 'title', nlp)
    pos_text = spacy_POS(data, 'text', nlp)
    
    nlp = spacy.load('en_core_web_sm', disable = ['tagger', 'parser'])
    ner_title = spacy_POS(data, 'title', nlp)
    ner_text = spacy_POS(data, 'text', nlp)
    
    analyser = SentimentIntensityAnalyzer()
    for col in ['text', 'title']:
        data = sentiment_analysis(data, col, analyser)
        
    lda_text = lda_analysis(data, 'text', data_nature)
    
    features = pd.concat([data[['text_tot_chars', 
                          'text_tot_words', 
                          'text_tot_sentences',
                          'title_tot_chars', 
                          'title_tot_words',
                          'title_sentiment',
                          'text_sentiment']], 
                          tfidf_df, 
                          pos_title, 
                          pos_text,
                          ner_title,
                          ner_text,
                          lda_text,
                          data[['label']]], axis = 1)
    
    features.to_csv(''.join(['./data/', data_nature, '_engineered.csv']), index = False)