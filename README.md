# Write Right!
## Helping CrossLead identify successful  business objectives

This post describes my project as a Fellow in Insight Data Science, a program that helps academics make the transition to industry.

At Insight, I consulted with CrossLead, an enterprise SaaS platform that provides a leadership and management system to optimize organizational performance. They help companies organize their business strategies, principles, and goals into a hierarchical structure that helps them track progress and link objectives accross different teams and levels of their organization. 

CrossLead’s management model uses an Alignment Triangle, in which the company’s vision stands at the top, and then progressively more concrete ways of attaining it are stacked below it in a hierarchy. They currently provide a platform to construct this pyramid by linking the different teams in the company and nudging their clients towards a framework (within their platform) in which smaller, more concrete initiatives are linked to more comprehensive strategies and then to broader objectives and finally to the company’s mission and vision. They also provide a dashboard and some analytics that lets them track the progress of each goal (in all levels of the hierarchy) and link the team members responsible for it.

<iframe src="https://docs.google.com/presentation/d/1XBg5TCPeHghp39Idn7lJOb83cs8hqIpHW1fF7PeI4-o/embed?start=false&loop=false&delayms=3000" frameborder="0" width="960" height="569" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>

One of Crosslead’s own objectives is to expand their analytic capabilities to provide insights to their clients on how to optimize their strategic planning approach. In support of this, they asked me to develop a predictive model for objective completion using the information contained in the description of the objective itself. WHAT WILL it to for them 

Management best practices reccommend that companies state their goals in a way that is more specific, measurable and timed, but can we find this in CrossLead’s data? How much information can we extract from the way an objective is written that gives us generalizable insights about how other companies should structure their plans?

Quite a bit, it turns out…

## Defining the target

CrossLead has around 7,500 business objectives in their dataset. These objectives all have a description (like the mock one above). Around 1/3 of them have never been updated, roughly 1/3 have been updated, but there has been no progress, and the rest have a level of completion between 0 and 100%. A significant portion of the data accumulates on two values (not updated, and 0% completion), so I will treat this as a classification problem where my target variable is binned into four classes: the ones that are never updated, updated but zero progress, progress up to 50% and progress up to 100%. Doing this I turn to the feature engineering process. 

## First pass: Tf-Idf and Linear SVM

Stating business objectives usually involves answers to three of the five W questions: **WHO**, **WHAT**, and **WHEN**. Intuitively good goals will find a way to succinctly answer those questions in a short sentence. However, analyzing the structure of text written for this specific purpose poses a particular challenge for NLP: the *whens* will mostly be dates, the *whats* will include a lot of numbers and named entities and the *whos* will be mostly named entities: teams, company names, organizatons.  

NLP tecniques like Tf-Idf would be particularly good at assigning importance to the named entities in the corpus of business objectives, since their frequency on the overall document is low. However, a transformation such as this would inform the classification algorithm about the entities that are most likely to be associated with completed objectives. This is not CrossLead's challenge, as they would like to predict which types of objectives are more thoroghly completed. 

In order to address this, the first task is to perform entity recognition on the corpus to replace these entities with a tag so that the structure of the business objective is preserved, but not the particulars of the team or date in the text. For this I use Spacy's built in entity recognition tool and replace (in the tagged text) all numbers and dates with their entity code. As for particular team names, the ideal thing would be to train a model for entity recognition, but since it would require manual coding, a workaround is to tag the text and replace all proper nouns with a code. So, a fictitious objective would be preprocessed as follows before applying Tf-Idf:

Original:

>Identify 8 key business partners for Team Alpha to propose microblogging deal by end of 2016.


Entities and proper nouns:

>Identify **NUM** key business partners for **PROPN** to propose microblogging deal by **DATE**


Stopwords and lemmatizing:

>Identify **NUM** key business partner **PROPN** propose microblogging deal **DATE**

Once the text has been processed, I use Tf-Idf to vectorize the feature space taking the collection of objective descriptions as a corpus and each individual description as a document. In this case, most of the work is being done by the inverse document frequency component of Tf-Idf, given that short sentences are unlikely to have words repeated more than once (Tf will be 1 for most of the words). While this might seem odd, the feature space produced by this transformation is the best way to find *uniqueness* of the words in the objective description (like in other short text contexts -e.g. finding the relevance of search terms or tweets).

Once the features have been vectorized, the next task is to apply a classification algorithm to predict the class of each objective. The choice of a classifier has to take into account that interpretability is one of the main objectives since we want to be able to construct a style editor from the feature importance analysis. This is the main reason why I chose an SVM classifier (with C=1) with a linear kernel, since it allows me to extract the features (in this case bi-grams and words) with the highest *coefficients* to analyze the words that were most predictive of each particular class. The model achieves an accuracy of 65%. Experimenting with different kernels and doing grid search can boost the accuracy up to 1% (with RBF kernel, C= 1.3 and Gamma=0.9), but performing feature importance analysis with RBF is not straightforward given that we do not know the mapping function explicitly. 

<img src="https://www.dropbox.com/s/v61u51pnwnk3da4/Screenshot%202017-02-09%2014.54.15.png?raw=1" />

## Can this be improved?

