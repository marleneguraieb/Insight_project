#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 13:42:15 2017

@author: marleneguraieb
"""

# Extract all syntactic components from text and return a dictionary:
def syntaxFeat(text):
    # get rid of newlines, and non alpha-numeric characters
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = re.sub(r'([^\s\w]|_)+', ' ', text)
    text = text.replace(' +',' ')


    # parse it
    parsed_text = nlp(text)
    
    # get tokens, and length
    token_parsed = [token for token in parsed_text]
    len_token = len(token_parsed)
    
    # Walk up the syntactic tree, collecting the arc labels.
    len_dep = []
    for token in token_parsed:
        dep_labels = []
        while token.head is not token:
            dep_labels.append(token.dep_)
            token = token.head
        len_dep.append(len(dep_labels))
    #this is the longest subtree, one variable you need to return
    max_stree = max(len_dep)
    
    #count entities/length of sentence
    perc_ents = len(parsed_text.ents)/len_token
    
    #count proper nouns
    perc_propn = sum([token.pos_=='PROPN' for token in parsed_text])/len_token
    
    #count noun chunks
    perc_nounch = len([token for token in parsed_text.noun_chunks])/len_token
    
    #count words out of vocabulary
    perc_oov = sum([token.is_oov for token in parsed_text])/len_token
    
    #count numbers
    perc_num = sum([token.like_num for token in parsed_text])/len_token
    
    return {'len_token':len_token,'max_stree':max_stree,'perc_ents':perc_ents,'perc_propn':perc_propn,
            'perc_nounch':perc_nounch,'perc_oov':perc_oov,'perc_num':perc_num}
    

#Do syntax features for every row and return a dataframe
def syntaxFeatures(df):    
    syntax = pd.DataFrame()
    for i in df.index.values:
        temp = syntaxFeat(df.loc[i,'X'])
        syntax = syntax.append(temp,ignore_index=True)
    return syntax

#Get all the vocabulary in the corpus and its features:
def vocab_pl(data):
    #get dictionary data
    pl_data = pd.read_csv('pl_dict_clean.csv')
    pl_data = pl_data[['word','fam','conc','imag','kf_wf']]
    pl_data = pl_data.dropna(thresh=2).drop_duplicates()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pl_data = pl_data.groupby('word').agg({'fam': np.nanmean,'conc': np.nanmean,'imag': np.nanmean, 
                                               'kf_wf':np.nanmean}).reset_index() 

    vocab = []
    for i in data.index.values:
        # get rid of newlines, and non alpha-numeric characters
        text = data.loc[i,'X']
        text = text.strip().replace("\n", " ").replace("\r", " ")
        text = re.sub(r'([^\s\w]|_)+', ' ', text)
        text = text.replace(' +',' ')
        parsed_text = nlp(text)
        token_text = [token.orth_ for token in parsed_text]
        for token in token_text:
            vocab.append(token.upper())
    vocab = list(set(vocab))
    
    vocab_in_pl = [word for word in vocab if word in list(pl_data.word)]
    pl_data =  pl_data[pl_data['word'].isin(vocab_in_pl)]
    
    return pl_data

# Score each sentence
def pl_features(text,pl_data):
        
    # get rid of newlines, and non alpha-numeric characters
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = re.sub(r'([^\s\w]|_)+', ' ', text)
    text = text.replace(' +',' ')

    # parse it
    parsed_text = nlp(text)
    
    # get tokens
    token_text = [token.orth_ for token in parsed_text]

    kf_scores = []
    fam_scores = []
    conc_scores = []
    imag_scores = []
    for token in token_text:
        kf_scores.append(pl_data.loc[pl_data['word']==token.upper(),'kf_wf'].max())
        fam_scores.append(pl_data.loc[pl_data['word']==token.upper(),'fam'].max())
        conc_scores.append(pl_data.loc[pl_data['word']==token.upper(),'conc'].max())
        imag_scores.append(pl_data.loc[pl_data['word']==token.upper(),'imag'].max())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        kf =   np.nanmean(kf_scores)
        fam =  np.nanmean(fam_scores)
        conc = np.nanmean(conc_scores)
        imag = np.nanmean(imag_scores)
        perc_kf = 1-(np.isnan(kf_scores).sum()/len(kf_scores))
    
    return {'kf_score': kf,'fam_score': fam,'conc_score': conc,'imag_score': imag,'perc_kf': perc_kf}

#Run it for all the dataset:
def plFeatures(df,pl_data):
    #get dictionary data
#    pl_data = vocab_pl(df)

    pl = pd.DataFrame()
    for i in df.index.values:
        temp = pl_features(df.loc[i,'X'],pl_data)
        pl = pl.append(temp,ignore_index=True)
    return pl

