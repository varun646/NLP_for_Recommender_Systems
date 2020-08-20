import pandas as pd
import numpy as np
from data import UserItemRatingDataset, SampleGenerator
from engine import Engine
from transformers import BertConfig
from seq2seq_reviews import Seq2Seq, Seq2SeqEmbeddingGenerator
from BERT_embedding import BertEmbeddingGenerator


def generate_tensors(embedding_generator, review_json):
    json_file = pd.read_json(review_json, lines=True)

    user_ids = []
    [user_ids.append(id) for id in json_file['reviewerID'] if not pd.isna(id)]

    items = []
    [items.append(item) for item in json_file['asin'] if not pd.isna(item)]

    ratings = []
    [ratings.append(rating) for rating in json_file['overall'] if not pd.isna(rating)]

    user_ids = np.array(user_ids)
    items = np.array(items)
    ratings = np.array(ratings)

    user_ids = pd.DataFrame(user_ids)
    items = pd.DataFrame(items)
    ratings = pd.DataFrame(ratings)
    embeddings = pd.DataFrame(np.array(embedding_generator(embedding_generator.embeddings())))

    # ['userId', 'itemId', 'rating', 'review_embedding']
    tensors = pd.DataFrame.append(user_ids)
    tensors = pd.DataFrame.append(ratings)
    tensors = pd.DataFrame.append(user_ids)
    tensors = pd.DataFrame.append(embeddings)

    return tensors


def test_nlp_algs(review_json):
    bert_config = BertConfig()
    bert_engine = Engine(bert_config)
    bert_embedding_generator = BertEmbeddingGenerator(review_json)
    bert_tensors = generate_tensors(bert_embedding_generator, review_json)

    seq2seq_config = Seq2Seq.get_config()
    seq2seq_engine = Engine(seq2seq_config)
    seq2seq_embedding_generator = Seq2SeqEmbeddingGenerator(review_json)
    seq2seq_tensors = generate_tensors(seq2seq_embedding_generator, review_json)

    bert_rating_dataset = UserItemRatingDataset(bert_tensors)
    seq2seq_rating_dataset = UserItemRatingDataset(seq2seq_tensors)

    bert_evaluation_tool = SampleGenerator(bert_rating_dataset)
    seq2seq_evaluation_tool = SampleGenerator(seq2seq_rating_dataset)

    return (bert_evaluation_tool.evaluate_data(), seq2seq_evaluation_tool.evaluate_data())


test_nlp_algs('AMAZON_FASHION_5.json')
