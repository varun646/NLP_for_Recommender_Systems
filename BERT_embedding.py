# USED http://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#1-loading-pre-trained-bert FOR CODE
import torch
import pandas as pd
from transformers import BertTokenizer, BertForPreTraining, BertConfig


class BertEmbeddingGenerator():
    def __init__(self, review_json_file):
        # store reviews from json
        self.reviews_json = pd.read_json(review_json_file, lines=True)
        # Load pre-trained model tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('./uncased_L-4_H-256_A-4/')

    def get_marked_reviews(self):
        # token for beginning of sentence
        CLS_TOKEN = "[CLS] "

        # token for separating/ending sentences
        SEP_TOKEN = " [SEP]"

        marked_reviews = []
        for review in self.reviews_json['reviewText']:
            if not (pd.isna(review)):
                marked_reviews.append(CLS_TOKEN + review + SEP_TOKEN)

        return marked_reviews

    def generate_review_embedding(model, tokenizer, marked_review):
        # Split the sentence into tokens.
        tokenized_text = tokenizer.tokenize(marked_review)

        # Map the token strings to their vocabulary indeces.
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        # Mark each of the tokens as belonging to sentence "1".
        segments_ids = [1] * len(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # Run the text through BERT, and collect all of the hidden states produced
        # from all 12 layers.
        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)

            # Evaluating the model will return a different number of objects based on
            # how it's  configured in the `from_pretrained` call earlier. In this case,
            # becase we set `output_hidden_states = True`, the third item will be the
            # hidden states from all layers. See the documentation for more details:
            # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
            hidden_states = outputs[2]

        # Concatenate the tensors for all layers. We use `stack` here to
        # create a new dimension in the tensor.
        token_embeddings = torch.stack(hidden_states, dim=0)

        token_embeddings.size()

        torch.Size([13, 1, 22, 768])

        # Remove dimension 1, the "batches".
        token_embeddings = torch.squeeze(token_embeddings, dim=1)

        token_embeddings.size()
        torch.Size([13, 22, 768])

        # Swap dimensions 0 and 1.
        token_embeddings = token_embeddings.permute(1, 0, 2)

        token_embeddings.size()

        torch.Size([22, 13, 768])

        # `token_vecs` is a tensor with shape [22 x 768]
        token_vecs = hidden_states[-2][0]

        # Calculate the average of all 22 token vectors.
        sentence_embedding = torch.mean(token_vecs, dim=0)

        return sentence_embedding

    def embeddings(self):
        # Load pre-trained model (weights)
        config = BertConfig.from_json_file('./uncased_L-4_H-256_A-4/bert_config.json')
        config.output_hidden_states = True
        model = BertForPreTraining.from_pretrained('./uncased_L-4_H-256_A-4/bert_model.ckpt.index', from_tf=True,
                                                   config=config)

        # Put the model in "evaluation" mode, meaning feed-forward operation.
        model.eval()
        review_embeddings = []
        for review in self.get_marked_reviews():
            embedding = self.generate_review_embedding(model, self.tokenizer, review)
            review_embeddings.append(embedding)
        return review_embeddings
