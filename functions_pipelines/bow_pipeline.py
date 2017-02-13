#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 13:33:16 2017

@author: marleneguraieb
"""

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline


# transformer that cleans text with spaCy


class CleanTextTransformer(TransformerMixin):
    """
    Convert text to cleaned text
    """

    def transform(self, X, **transform_params):
        return [cleanText(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# function that cleans text:
def cleanText(text):
    # get rid of newlines, and non alpha-numeric characters
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = re.sub(r'([^\s\w]|_)+', ' ', text)
    text = re.sub(' +',' ',text)   



    #parse it
    parsed_text = nlp(text)

    # get rid of proper nouns
    proper = []
    token_isoov = [token.is_oov for token in parsed_text]
    token_text = [token.orth_ for token in parsed_text]
    token_pos = [token.pos_ for token in parsed_text]
    for pos, word, oov in zip(token_pos,token_text,token_isoov):
        if pos == 'PROPN' and oov==True:
            proper.append(str(word))
    for pro in proper:
        text = text.replace(pro,' PROPN ')


    #recode entities
    ents = {}
    for num,entity in enumerate(parsed_text.ents):
        ents[entity.label_] = entity.orth_
    for code, entity in ents.items():
        text = text.replace(entity,str(' '+code+' '))
        
    
    # lowercase
    text = text.lower()
    text = re.sub(' +',' ',text)  
    
    return text

# A custom function to tokenize the text using spaCy
# and convert to lemmas
def tokenizeText(sample):

    # get the tokens using spaCy
    tokens = parser(sample)

    # lemmatize
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas

    # stoplist the tokens
    tokens = [tok for tok in tokens if tok not in STOPLIST]

    # stoplist symbols
    tokens = [tok for tok in tokens if tok not in SYMBOLS]

    # remove large strings of whitespace
    while "" in tokens:
        tokens.remove("")
    while " " in tokens:
        tokens.remove(" ")
    while "\n" in tokens:
        tokens.remove("\n")
    while "\n\n" in tokens:
        tokens.remove("\n\n")

    return tokens


#Visualization

## Print the top features for each class:
def print_top10(vectorizer, clf,N):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(clf.classes_):
        top = np.argsort(clf.coef_[i])[-N:]
        print("%s: %s" % (class_label,
              "-".join(feature_names[j] for j in top)))

## Create a dictionary with all the top features in the classes:
def top_feat_dict(vectorizer, clf,N):
    """Prints features with the highest coefficient values, per class"""
    top_feats = {}
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(clf.classes_):
        top = np.argsort(clf.coef_[i])[-N:]
        top_feats['Class_' + str(class_label)] = pd.DataFrame({'Feature': [feature_names[j] for j in top],
              'Coefficient': [clf.coef_[2][j] for j in top]})
    return top_feats

def top_verb_dict(vectorizer, clf,N):
    """Creates a dictionary with all the top verbs in the classes"""
    top_feats = {}
    feature_names = vectorizer.get_feature_names()
    verbs = [len([1 for token in nlp(feature) if token.pos_ == 'VERB']) for feature in feature_names] 
    verbs = [np.nan if x == 0 else 1 for x in verbs]
    for i, class_label in enumerate(clf.classes_):
        top = np.argsort([-a*b for a,b in zip(clf_bow.coef_[i],verbs)])[::-1][-N:]
        top_feats['Class_' + str(class_label)] = pd.DataFrame({'Verb': [feature_names[j] for j in top],
              'Coefficient': [clf.coef_[i][j] for j in top]})
    return top_feats

"""
===============
Function to create the word cloud background:
"""
    
def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)




