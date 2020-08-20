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


class Word2VecEmbeddingGenerator():
    def __init__(self, review_json_file):
        reviews_json = pd.read_json('AMAZON_FASHION_5.json', lines=True)
        reviews = []
        self.tokenized_reviews = []
        num_reviews = 0
        for review in reviews_json['reviewText']:
            if not (pd.isna(review)):
                num_reviews += 1
                reviews.append(review)
                review = nltk.word_tokenize(review)
                self.tokenized_reviews.append(review)

        self.model = Word2Vec(self.tokenized_reviews, min_count=3, size=100, workers=3, window=3, sg=0)
        self.model.build_vocab(sentences=reviews, update=True)
        self.model.train(reviews, total_examples=num_reviews, epochs=30)

    def embeddings(self):
        embedding_array = []
        for review in self.tokenized_reviews:
            # get zero vector
            review_emb = self.model.wv.get_vector('good')
            review_emb -= self.model.wv.get_vector('good')
            for word in review:
                review_emb += self.model.wv.get_vector(word)
            embedding_array.append(review_emb)

        return embedding_array
