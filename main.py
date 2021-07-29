"""
Imports and Setup
"""

from bs4 import BeautifulSoup
from gensim.models import Word2Vec, word2vec
from lime import lime_tabular
from lime import lime_text
from lime.lime_text import LimeTextExplainer
from nltk.corpus import stopwords as sw
from pathlib import Path

from simpletransformers.classification import ClassificationModel, ClassificationArgs
from simpletransformers.model import TransformerModel

from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from tqdm import tqdm, tqdm_pandas, tqdm_notebook

import gensim
import lime
import logging
import math
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pandas as pd
import random
import re
import seaborn as sns
import sklearn
import spacy
import sqlite3
import string

tqdm.pandas()

# Creating a list of classifiers to be used for testing later
classifiers = []
classifiers.append(('LR', LogisticRegression(multi_class='ovr', max_iter=500, random_state=42)))
classifiers.append(('SGD', SGDClassifier(random_state=42)))
classifiers.append(('KNN', KNeighborsClassifier()))
classifiers.append(('CART', DecisionTreeClassifier(random_state=42)))
classifiers.append(('SVM', SVC(random_state=42)))
classifiers.append(('RF', RandomForestClassifier(random_state=42, n_estimators=100)))
classifiers.append(('AB', AdaBoostClassifier(random_state=42, n_estimators=100)))
classifiers.append(('GB', GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=1.0)))


def get_classification_results(models, classifiers, cm=False):
    classification_results = []

    for model_name, model in models:
        target = df_clean['type'].copy()
        target.columns = ["type", "fake"]
        target = target.astype(int)
        if 'fraudulent' in model.columns:
            features = model.drop(['fraudulent', 'type'], axis=1)
        else:
            features = model.drop(['type'], axis=1)
        X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(features, target, test_size=0.2,
                                                                            stratify=target, random_state=42)

        for name, classifier in classifiers:
            print('Fitting ', str(name), ' on ', str(model_name))
            model_fit = classifier.fit(X_train_clf, y_train_clf.values.ravel())
            predicted = model_fit.predict(X_test_clf)
            classification_results.append((model_name, name, pd.DataFrame(
                classification_report(y_test_clf, predicted, output_dict=True)).transpose(),
                                           confusion_matrix(y_test_clf, predicted),
                                           matthews_corrcoef(y_test_clf, predicted)))
            if cm == True:
                disp = plot_confusion_matrix(model_fit, X_test_clf, y_test_clf, cmap=plt.cm.Blues, values_format='')
                disp.ax_.set_title(str('Confusion Matrix for ' + str(model_name) + ' ' + str(name)))
                plt.show()

    return (classification_results)


"""
Data Processing
"""

# Load annotated data from CSV
combined_jobs_upsampled = pd.read_csv('emscad_types_anon_upsampled_1.csv')
combined_jobs_upsampled['type'].value_counts()

# Load NLTK and spaCy libraries
nltk.download('stopwords')
nltk.download('words')

nlp = spacy.load("en_core_web_lg")
stopwords = sw.words('english')
words = set(nltk.corpus.words.words())
punctuations = string.punctuation


# Function: Clean Text
def process_text(docs):
    texts = []
    for doc in tqdm(docs):
        doc = re.sub(r'\d+\n/gm', ' ', doc)
        doc = nlp(doc, disable=['parser', 'ner'])
        tokens = [tok.lemma_.lower().strip() for tok in doc]
        tokens = [tok for tok in tokens if
                  tok in words and tok not in stopwords and tok not in punctuations and tok.isnumeric() is False]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)


# Creating a dataframe of cleaned job advertisements
df_clean = combined_jobs_upsampled.copy()
df_clean['text'] = process_text(df_clean['text'])

"""
Bag of Words / TfIdf Feature Sets
"""

df_pre_bow = df_clean.copy()
df_pre_bow = df_pre_bow.reset_index(drop=True)

# Applying the CountVectorizer and TfidfVectorizer functions
count_vectorizer = CountVectorizer(stop_words="english")
tfidf_vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(stop_words="english", lowercase=False)

# Fitting on the cleaned text
vectorized_bow = count_vectorizer.fit_transform(df_pre_bow['text'])
vectorized_tfidf = tfidf_vectorizer.fit_transform(df_pre_bow['text'])

# Finalising the dataframe with tfidf and bow vectors
df_bow = pd.DataFrame(vectorized_bow.todense(), columns=count_vectorizer.get_feature_names())
df_bow = pd.concat([df_bow, df_clean['type']], axis=1, join="inner")

df_tfidf = pd.DataFrame(vectorized_tfidf.todense(), columns=tfidf_vectorizer.get_feature_names())
df_tfidf = pd.concat([df_tfidf, df_clean['type']], axis=1, join="inner")

models_class1 = []
models_class1.append(('1: BOW', df_bow))
models_class1.append(('1: TFIDF', df_tfidf))

# Test the classification of the BOW and TFIDF Models
get_classification_results(models_class1, classifiers)

"""
Ruleset Features
"""

# Creating a ruleset dataframe based on the unprocessed dataset
df_ruleset = combined_jobs_upsampled.copy()

## Setting up the functions for data processing and feature extraction

# Function: Presence of Word in String
def is_word_in_field(docs, word):
    word_in_string = []
    for doc in tqdm(docs):
        if word.lower() in doc.lower():
            word_in_string.append(1)
        else:
            word_in_string.append(0)
    return pd.Series(word_in_string)


# Function: Presence of Words in String
def are_words_in_field(docs, words):
    contains = []
    for doc in tqdm(docs):
        has_tokens = 0
        for word in words:
            if word.lower() in doc.lower():
                has_tokens = 1
        contains.append(has_tokens)
    return pd.Series(contains)


# Feature: Number of Consecutive Punctuation
def get_consecutive_punctuation(docs):
    counts = []
    for doc in tqdm(docs):
        count = 0
        doc = nlp(doc, disable=['parser', 'ner'])
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = list(zip(tokens[1:], tokens))
        for tok in tokens:
            if tok[0] in punctuations and tok[1] in punctuations:
                count += 1
        counts.append(count)
    return pd.Series(counts)


# Function: Get POS Tag
def get_pos_tags(text, tag):
    count = 0
    doc = nlp(text)
    for tok in doc:
        if tok.pos_ != tag:
            continue
        count += 1
    return count


# Extract Feature: No company profile
empty_profiles = []
for profile in df_ruleset['company_profile']:
    if profile == ' ':
        empty_profiles.append(1)
    else:
        empty_profiles.append(0)

df_ruleset['has_no_company_profile'] = pd.Series(empty_profiles)

# Extract Feature: Short Company Profile
short_profiles = []
for desc in tqdm(df_ruleset['company_profile']):
    if len(desc) < 10:
        short_profiles.append(1)
    else:
        short_profiles.append(0)

df_ruleset['has_short_company_profile'] = pd.Series(short_profiles)

# Extract Feature: Has No Long Company Profile
no_long_profiles = []
for desc in tqdm(df_ruleset['company_profile']):
    if 10 < len(desc) < 100:
        no_long_profiles.append(1)
    else:
        no_long_profiles.append(0)

df_ruleset['has_no_long_company_profile'] = pd.Series(no_long_profiles)

# Extract Feature: Has Short Description
short_descriptions = []
for desc in tqdm(df_ruleset['description']):
    if len(desc) < 10:
        short_descriptions.append(1)
    else:
        short_descriptions.append(0)

df_ruleset['has_short_description'] = pd.Series(short_descriptions)

# Extract Feature: Has Short Requirements
short_requirements = []
for desc in tqdm(df_ruleset['requirements']):
    if len(desc) < 10:
        short_requirements.append(1)
    else:
        short_requirements.append(0)

df_ruleset['has_short_requirements'] = pd.Series(short_requirements)

# Extract Feature: Contains Spam Word
df_ruleset['contains_spamword'] = are_words_in_field(df_ruleset['text'],
                                                     ['home', 'online', 'week', 'income', 'extra', 'cash'])

# Extract Feature: Consecutive Punctuation
df_ruleset['consecutive_punct'] = get_consecutive_punctuation(df_ruleset['text'])

# Extract Feature: Has Money Symbols in Title
df_ruleset['money_in_title'] = are_words_in_field(df_ruleset['title'],
                                                  ['$', '$$', '$$$', '€', '€€', '€€€', '£', '££', '£££'])

# Extract Feature: Has Money Symbols in Description
df_ruleset['money_in_description'] = are_words_in_field(df_ruleset['description'],
                                                        ['$', '$$', '$$$', '€', '€€', '€€€', '£', '££', '£££'])

# Extract Feature: Has URL in Text
df_ruleset['url_in_text'] = is_word_in_field(df_ruleset['text'], "url")

# Extract Feature: Prompts for External Application
df_ruleset['external_application'] = are_words_in_field(df_ruleset['text'],
                                                        ['apply at', 'send resume', 'send your resume', 'contact us',
                                                         'call me', 'call us'])

# Extract Feature: Addresses Lower Education
df_ruleset['addresses_lower_education'] = are_words_in_field(df_ruleset['text'], ['high school', 'no degree'])

# Extract Feature: Has Incomplete Extra Attributes
extra_cols = ['industry', 'function', 'employment_type', 'required_education']
df_ruleset['has_incomplete_extra_attributes'] = pd.Series(df_ruleset[extra_cols].isna().any(axis=1).astype(int))

# Extract Feature: POS Tags
df_pos = df_ruleset.copy()
df_pos['noun_count'] = df_pos['text'].progress_apply(lambda x: get_pos_tags(x, 'NOUN'))
df_pos['verb_count'] = df_pos['text'].progress_apply(lambda x: get_pos_tags(x, 'VERB'))
df_pos['adj_count'] = df_pos['text'].progress_apply(lambda x: get_pos_tags(x, 'ADJ'))
df_pos['adv_count'] = df_pos['text'].progress_apply(lambda x: get_pos_tags(x, 'ADV'))
df_pos['pron_count'] = df_pos['text'].progress_apply(lambda x: get_pos_tags(x, 'PRON'))

# Finalising the Ruleset Models
df_no_pos = df_ruleset.copy()
df_pos.drop(['description', 'company_profile', 'requirements', 'benefits', 'title', 'department', 'salary_range',
             'in_balanced_dataset', 'employment_type', 'required_experience', 'required_education', 'industry',
             'function', 'fraudulent', 'text'], axis=1, inplace=True)
df_no_pos.drop(['description', 'company_profile', 'requirements', 'benefits', 'title', 'department', 'salary_range',
                'in_balanced_dataset', 'employment_type', 'required_experience', 'required_education', 'industry',
                'function', 'fraudulent', 'text'], axis=1, inplace=True)

# Feature Correlations
df_no_pos_correlations = pd.concat([df_no_pos.drop(['type'], axis=1),
                                    pd.get_dummies(df_no_pos['type'])], axis=1)

plt.figure(figsize=(15, 12))
correlations = df_no_pos_correlations.corr()
sns.heatmap(correlations, annot=True, cmap=plt.cm.Blues)
plt.show()

for i in [0, 1, 2, 3]:
    cor_target = (correlations[i])
    print('Top feature correlations for class ', i)
    print(cor_target[abs(cor_target) > 0.05])
    print(' ')

# Dropping features with 1-1 correlations
df_no_pos.drop(['telecommuting', 'has_no_company_profile'], axis=1, inplace=True)
df_pos.drop(['telecommuting', 'has_no_company_profile'], axis=1, inplace=True)

# Combining Ruleset+POS with Tf-Idf / BoW Models
df_bow = df_bow.drop(['fraudulent', 'type'], axis=1)
df_tfidf = df_tfidf.drop(['fraudulent', 'type'], axis=1)

df_pos_bow = pd.concat([df_pos.reset_index(drop=True), df_bow.reset_index(drop=True)], axis=1)
df_pos_tfidf = pd.concat([df_pos.reset_index(drop=True), df_tfidf.reset_index(drop=True)], axis=1)

models_class2 = []
models_class2.append(('2: Ruleset', df_no_pos))
models_class2.append(('2: Ruleset+POS', df_pos))
models_class2.append(('2: Ruleset+POS+BOW', df_pos_bow))
models_class2.append(('2: Ruleset+POS+TFIDF', df_pos_tfidf))

get_classification_results(models_class2, classifiers)

# RF Feature Importance
target = df_clean['type'].copy()
target.columns = ["type", "fake"]
target = target.astype(int)
features = df_pos.drop(['type'], axis=1)

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(features, target, test_size=0.2,
                                                                stratify=target, random_state=42)
feature_names = X_train_rf.columns

clf_rf_model = RandomForestClassifier(random_state=42).fit(X_train_rf, y_train_rf.values.ravel())

result = permutation_importance(clf_rf_model, X_train_rf, y_train_rf, n_repeats=15, random_state=42, n_jobs=2)

rf_importance = pd.Series(result.importances_mean, index=feature_names).sort_values(ascending=False)

# Plotting Random Forest Feature Importance
fig, ax = plt.subplots()
rf_importance.plot.bar(ax=ax)
ax.set_title("Feature Importances using Permutation")
ax.set_ylabel("Mean accuracy decrease")
# fig.tight_layout()
plt.show()

rf_importance

"""
Word2Vec Embeddings
"""

logging.basicConfig(level=logging.INFO)
nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def sentence_to_words(text):
    snt_words = re.sub("[^a-zA-Z]", " ", text)
    snt_words_split = snt_words.lower().split()
    return snt_words_split


def doc_to_sentences(doc, tokenizer):
    sentences = []
    tokenized_sentences = tokenizer.tokenize(doc.strip())
    for sentence in tokenized_sentences:
        if len(sentence) > 0:
            sentences.append(sentence_to_words(sentence))
    return sentences


def gen_feature_vec(doc, embedding, num_features):
    vec = np.zeros(num_features, dtype="float32")
    word_count = 0

    word_index2word = set(embedding.wv.index2word)
    for word in doc:
        if word not in word_index2word:
            continue
        word_count += 1
        vec = np.add(vec, embedding[word])

    vec = np.divide(vec, word_count)
    return vec


def avg_feature_vec(docs, embeddings, num_features):
    feature_vecs = np.zeros((len(docs), num_features), dtype='float32')

    i = 0
    for doc in docs:
        feature_vecs[i] = gen_feature_vec(doc, embeddings, num_features)
        i += 1

    return feature_vecs

# Creating the training and testing sets for the embeddings
target_embed = combined_jobs_upsampled['type'].copy()
target_embed.columns = ["type", "fake"]
target_embed = target_embed.astype(int)
features_embed = combined_jobs_upsampled['text']
X_train_embed, X_test_embed, y_train_embed, y_test_embed = train_test_split(features_embed, target_embed, test_size=0.2,
                                                                            stratify=target_embed, random_state=42)

train_data_embed = pd.DataFrame()
test_data_embed = pd.DataFrame()

train_data_embed['text'] = X_train_embed
train_data_embed['labels'] = y_train_embed

test_data_embed['text'] = X_test_embed
test_data_embed['labels'] = y_test_embed

train_sentences = []  # Initialize an empty list of sentences
for doc in train_data_embed['text']:
    train_sentences += doc_to_sentences(doc, tokenizer)

# Training the word2vec model
w2v_model = word2vec.Word2Vec(train_sentences, workers=3,
                              size=300, min_count=40,
                              window=10, sample=1e-3)

train_words_processed = process_text(train_data_embed['text'])
test_words_processed = process_text(test_data_embed['text'])

cleaned_train_text = []
cleaned_test_text = []

for doc in train_words_processed:
    cleaned_train_text.append(sentence_to_words(doc))

for doc in test_words_processed:
    cleaned_test_text.append(sentence_to_words(doc))

df_embed_train = pd.DataFrame(avg_feature_vec(cleaned_train_text, w2v_model, 300)).fillna(0)
df_embed_test = pd.DataFrame(avg_feature_vec(cleaned_test_text, w2v_model, 300)).fillna(0)

df_embed_train.head

# Combining Ruleset with Embeddings
df_embed_ruleset_train = pd.concat([df_embed_train.reset_index(drop=True), X_train_rf.reset_index(drop=True)],
                                   axis=1).fillna(0)
df_embed_ruleset_test = pd.concat([df_embed_test.reset_index(drop=True), X_test_rf.reset_index(drop=True)],
                                  axis=1).fillna(0)

# Evaluating the Word2Vec and CombinedWord2vec Models
for name, model in classifiers:
    model_fit = model.fit(df_embed_train, y_train_embed.values.ravel())
    predicted = model_fit.predict(df_embed_test)
    print("Classification report of " + str(name) + " tested on Embeddings test set:")
    print(classification_report(y_test_embed, predicted, digits=3))
    print("")
    print(matthews_corrcoef(y_test_embed, predicted))

for name, model in classifiers:
    model_fit = model.fit(df_embed_ruleset_train, y_train_embed.values.ravel())
    predicted = model_fit.predict(df_embed_ruleset_test)
    print("Classification report of " + str(name) + " tested on Embeddings+Ruleset test set:")
    print(classification_report(y_test_embed, predicted, digits=6))
    print("")
    print(matthews_corrcoef(y_test_embed, predicted))

# Hyperparameter tuning
grid_param = {
    'n_estimators': [90, 110, 130, 150],
    'criterion': ['gini', 'entropy'],
    'max_depth': range(2, 20, 5),
    'min_samples_leaf': range(1, 10, 2),
    'min_samples_split': range(2, 10, 2),
    'max_features': ['auto', 'log2']
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=grid_param, cv=5, n_jobs=-1,
                           verbose=3)
grid_search.fit(df_embed_ruleset_train, y_train_embed.values.ravel())

grid_search.best_params_

# RF Hyperparameter Tuning: Ruleset + Embeddings
clf_rf_combined = RandomForestClassifier(random_state=42, criterion='entropy',
                                         max_depth=17, max_features='auto',
                                         min_samples_leaf=1, min_samples_split=2,
                                         n_estimators=90).fit(df_embed_ruleset_train, y_train_embed.values.ravel())

pred_clf_combined = clf_rf_combined.predict(df_embed_ruleset_test)
print(confusion_matrix(y_test_embed, pred_clf_combined))
print(classification_report(y_true=y_test_embed, y_pred=pred_clf_combined))

"""
Transformer-based Model
"""

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

data_tf = pd.DataFrame()
data_tf['text'] = combined_jobs_upsampled['text']
data_tf['value'] = combined_jobs_upsampled['type']

target_trans = data_tf['value'].copy()
features_trans = data_tf['text'].copy()
X_train_trans, X_test_trans, y_train_trans, y_test_trans = train_test_split(features_trans, target_trans, test_size=0.2,
                                                                            stratify=target_trans, random_state=42)

train_data_tf = pd.DataFrame()
test_data_tf = pd.DataFrame()

train_data_tf['text'] = X_train_trans
train_data_tf['labels'] = y_train_trans

test_data_tf['text'] = X_test_trans
test_data_tf['labels'] = y_test_trans

model_args = {
    'num_train_epochs': 5,
    'max_seq_length': 256,
    'train_batch_size': 16,
    'eval_batch_size': 64,
    'warmup_steps': 500,
    'weight_decay': 0.01,
    'logging_steps': 10,
    'learning_rate': 5e-5,
    'fp16': False,
    'no_cache': True,
    'overwrite_output_dir': True
}

# Training and Evaluating BERT

trans_model_bert = ClassificationModel('bert', 'bert-base-cased', num_labels=4, use_cuda=True, args=model_args)
trans_model_roberta = ClassificationModel('roberta', 'roberta-base', num_labels=4, use_cuda=True, args=model_args)
trans_model_xlnet = ClassificationModel('xlnet', 'xlnet-base-cased', num_labels=4, use_cuda=True, args=model_args)

trans_model_bert.train_model(train_data_tf)
result_bert, model_outputs_bert, wrong_predictions_bert = trans_model_bert.eval_model(test_data_tf,
                                                                                      acc=sklearn.metrics.accuracy_score,
                                                                                      cm=sklearn.metrics.confusion_matrix)

predictions_bert, raw_outputs_bert = trans_model_bert.predict(list(X_test_trans))
pd.DataFrame(classification_report(y_pred=predictions_bert, y_true=y_test_trans, output_dict=True)).transpose()
disp = ConfusionMatrixDisplay(confusion_matrix=result_bert['cm'],
                              display_labels=np.asarray([0, 1, 2, 3, 4]))

disp = disp.plot(cmap=plt.cm.Blues, values_format='')
plt.show()

# Training and Evaluating RoBERTa
trans_model_roberta.train_model(train_data_tf)
result_roberta, model_outputs_roberta, wrong_predictions_roberta = trans_model_roberta.eval_model(test_data_tf,
                                                                                                  acc=sklearn.metrics.accuracy_score,
                                                                                                  cm=sklearn.metrics.confusion_matrix)

predictions_roberta, raw_outputs_roberta = trans_model_roberta.predict(list(X_test_trans))
pd.DataFrame(classification_report(y_pred=predictions_roberta, y_true=y_test_trans, output_dict=True)).transpose()

disp = ConfusionMatrixDisplay(confusion_matrix=result_roberta['cm'],
                              display_labels=np.asarray([0, 1, 2, 3, 4]))

disp = disp.plot(cmap=plt.cm.Blues, values_format='')
plt.show()

# Training and Evaluating XLNet
trans_model_xlnet.train_model(train_data_tf)
result_xlnet, model_outputs_xlnet, wrong_predictions_xlnet = trans_model_xlnet.eval_model(test_data_tf,
                                                                                          acc=sklearn.metrics.accuracy_score,
                                                                                          cm=sklearn.metrics.confusion_matrix)

predictions_xlnet, raw_outputs_xlnet = trans_model_xlnet.predict(list(X_test_trans))
pd.DataFrame(classification_report(y_pred=predictions_xlnet, y_true=y_test_trans, output_dict=True)).transpose()

disp = ConfusionMatrixDisplay(confusion_matrix=result_xlnet['cm'],
                              display_labels=np.asarray([0, 1, 2, 3, 4]))

disp = disp.plot(cmap=plt.cm.Blues, values_format='')
plt.show()

print("Classification report for BERT:")
print(pd.DataFrame(classification_report(y_pred=predictions_bert, y_true=y_test_trans, output_dict=True)).transpose())
print(" ")
print("Classification report for RoBERTa:")
print(
    pd.DataFrame(classification_report(y_pred=predictions_roberta, y_true=y_test_trans, output_dict=True)).transpose())
print(" ")
print("Classification report for XLNet:")
print(pd.DataFrame(classification_report(y_pred=predictions_xlnet, y_true=y_test_trans, output_dict=True)).transpose())
