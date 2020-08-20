import torch
import random
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset

random.seed(0)


class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating, embedding> Tensor into Pytorch Dataset"""

    def __init__(self, user_tensor, item_tensor, target_tensor, review_embedding_tensor):
        """
        args:

            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor
        self.review_embedding_tensor = review_embedding_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index], \
               self.review_embedding_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)


class SampleGenerator(object):
    """Construct dataset for NCF"""

    def __init__(self, ratings):
        """
        args:
            ratings: pd.DataFrame, which contains 4 columns
            ['userId', 'itemId', 'rating', 'review_embedding']
        """
        assert 'userId' in ratings.columns
        assert 'itemId' in ratings.columns
        assert 'rating' in ratings.columns
        assert 'review_embedding' in ratings.columns

        self.ratings = ratings
        self.normalize_ratings = self._normalize(ratings)
        self.user_pool = set(self.ratings['userId'].unique())
        self.item_pool = set(self.ratings['itemId'].unique())
        self.train_ratings, self.test_ratings = self._split_loo(self.normalize_ratings)
        # self.train_ratings, self.test_ratings = self._split_loo(self.ratings)

    def _normalize(self, ratings):
        """normalize into [0, 1] from [0, max_rating]"""
        ratings = deepcopy(ratings)
        max_rating = ratings.rating.max()
        ratings['rating'] = ratings.rating * 1.0 / max_rating
        return ratings

    def _split_loo(self, ratings):
        """leave one out train/test split """
        # ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        cut = 4 * len(ratings) // 5
        train = ratings[:cut]
        test = ratings[cut:]
        return train[['userId', 'itemId', 'rating', 'review_embedding']], test[
            ['userId', 'itemId', 'rating', 'review_embedding']]

    def instance_a_train_loader(self, batch_size):
        """instance train loader for one training epoch"""
        users, items, ratings, review_embeddings = [], [], [], []
        train_ratings = self.train_ratings
        train_ratings = train_ratings.sample(frac=1)
        for row in train_ratings.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rating))
            review_embeddings.append(row.review_embedding)
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.LongTensor(items),
                                        target_tensor=torch.FloatTensor(ratings),
                                        review_embedding_tensor=torch.FloatTensor(review_embeddings))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    @property
    def evaluate_data(self):
        """create evaluate data"""
        test_ratings = self.test_ratings
        test_users, test_items, test_review_embeddings, test_ratings = [], [], [], []
        for row in test_ratings.itertuples():
            test_users.append(int(row.userId))
            test_items.append(int(row.itemId))
            test_ratings.append(float(row.rating))
            test_review_embeddings.append(row.review_embedding)
        return [test_users, test_items, torch.FloatTensor(test_review_embeddings), test_ratings]
