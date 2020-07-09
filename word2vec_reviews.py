import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')

from gensim.models import Word2Vec
import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

reviews_json = pd.read_json('AMAZON_FASHION_5.json', lines= True)
reviews = []
tokenized_reviews = []
num_reviews = 0
for review in reviews_json['reviewText']:
    if not (pd.isna(review)):
        num_reviews += 1
        reviews.append(review)
        review = nltk.word_tokenize(review)
        tokenized_reviews.append(review)


model = Word2Vec(tokenized_reviews, min_count=3,size= 100,workers=3, window =3, sg = 0)
model.build_vocab(sentences= reviews, update= True)
model.train(reviews, total_examples= num_reviews, epochs= 30)

model.wv.get_vector('good')