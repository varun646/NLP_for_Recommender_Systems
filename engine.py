import torch
from torch.autograd import Variable
from mlp import MLP
# from mlp import ShareLayer
from utils import use_optimizer, use_cuda
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import numpy as np
import math


class Engine(object):
    """Meta Engine for training & evaluating NCF model
    """

    def __init__(self, config):
        self.config = config  # model configuration
        self.model = MLP(config)
        if config['use_cuda'] is True:
            self.model.cuda()
        self.opt = use_optimizer(self.model, config)
        self.crit = torch.nn.MSELoss()

    def train_single_batch(self, review_embeddings, ratings):
        self.opt.zero_grad()
        ratings_pred = self.model(review_embeddings)
        loss = self.crit(ratings_pred.squeeze(1), book_rating)
        loss.backward()
        self.opt.step()
        if self.config['use_cuda'] is True:
            loss = loss.data.cpu().numpy()
        else:
            loss = loss.data.numpy()
        return loss

    def train_an_epoch(self, train_book_loader, train_movie_loader, epoch_id, user_overlap=False, item_overlap=False):
        self.model.train()
        total_loss = 0
        for batch in zip(train_loader):
            ratings, review_embeddings = Variable(book_batch[2]), Variable(book_batch[3])
            ratings = ratings.float()
            if self.config['use_cuda'] is True:
                ratings = ratings.cuda()
                review_embeddings = review_embeddings.cuda()
            loss = self.train_single_batch(reveiw_embeddings, ratings)
            total_loss += loss
        return total_loss

    def evaluate(self, evaluate_data):
        self.model.eval()
        users, items, review_embeddings, ratings = evaluate_data[0], evaluate_data[1], Variable(evaluate_data[2]), \
                                                   evaluate_data[3]
        if self.config['use_cuda'] is True:
            review_embeddings = review_embeddings.cuda()
        pred_ratings = self.model(review_embeddings)
        pred_ratings = pred_ratings.detach().numpy()

        rmse = math.sqrt(mean_squared_error(pred_ratings, ratings))
        mae = mean_absolute_error(pred_ratings, ratings)

        unique_users = list(set(users))
        recommend, precision, recall = [], [], []
        for index in range(len(user)):
            recommend.append((user[index], item[index], ratings[index], pred_ratings[index]))
        for user in unique_users:
            user_ratings = [x for x in recommend if x[0] == user]
            user_ratings.sort(key=lambda x: x[3], reverse=True)
            user_ratings = user_ratings[:5]
            n_rel = sum((true_r >= 3.5) for (_, _, true_r, _) in user_ratings)
            n_rec_k = sum((est >= 3.5) for (_, _, _, est) in user_ratings)
            n_rel_and_rec_k = sum(((true_r >= 3.5) and (est >= 3.5))
                                  for (_, _, true_r, est) in user_ratings)
            precision.append(n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1)
            recall.append(n_rel_and_rec_k / n_rel if n_rel != 0 else 1)
        precision = np.mean(precision)
        recall = np.mean(recall)
        return rmse, mae, precision, recall

    def save(self, dirname, filename):
        with open(os.path.join(dirname, filename), 'wb') as f:
            torch.save(self.model.state_dict(), f)
