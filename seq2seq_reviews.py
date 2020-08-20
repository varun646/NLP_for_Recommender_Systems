#used https://www.kaggle.com/columbine/seq2seq-pytorch for Seq2Seq model code
import pandas as pd
import nltk
import random
import time
import csv

import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.data import Field, BucketIterator

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src : [sen_len, batch_size]
        embedded = self.dropout(self.embedding(src))

        # embedded : [sen_len, batch_size, emb_dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [sen_len, batch_size, hid_dim * n_directions]
        # hidden = [n_layers * n_direction, batch_size, hid_dim]
        # cell = [n_layers * n_direction, batch_size, hid_dim]

        return hidden, cell



class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=self.n_layers, dropout=dropout)

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch_size]
        # hidden = [n_layers * n_dir, batch_size, hid_dim]
        # cell = [n_layers * n_dir, batch_size, hid_dim]

        input = input.unsqueeze(0)
        # input : [1, ,batch_size]
        
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch_size, emb_dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [seq_len, batch_size, hid_dim * n_dir]
        # hidden = [n_layers * n_dir, batch_size, hid_dim]
        # cell = [n_layers * n_dir, batch_size, hid_dim]

        # seq_len and n_dir will always be 1 in the decoder
        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch_size, output_dim]
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            'hidden dimensions of encoder and decoder must be equal.'
        assert encoder.n_layers == decoder.n_layers, \
            'n_layers of encoder and decoder must be equal.'

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [sen_len, batch_size]
        # trg = [sen_len, batch_size]
        # teacher_forcing_ratio : the probability to use the teacher forcing.
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        # embeddings.append(hidden)

        # first input to the decoder is the <sos> token.
        input = trg[0, :]
        for t in range(1, trg_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states.
            output, hidden, cell = self.decoder(input, hidden, cell)

            # replace predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not.
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions.
            top1 = output.argmax(1)
            # update input : use ground_truth when teacher_force
            input = trg[t] if teacher_force else top1

        return outputs

    def embedding(self, src):
        emb, cell = self.encoder(src)
        return emb

    def get_config(self):
        config = {'batch_size': 128,
                  'optimizer': 'adam',
                  'lr': 1e-2,
                  'latent_dim': self.encoder.hid_dim,
                  'nlayers': self.encoder.n_layers,
                  'layers': [self.encoder.input_dim, self.encoder.hid_dim, self.encoder.emb_dim, self.decoder.hid_dim,
                             self.decoder.output_dim],
                  # layers[0] is the concat of latent user vector & latent item vector
                  'use_cuda': False}  # CHANGE THIS}
        return config


class Seq2SeqEmbeddingGenerator():
    def __init__(self, review_json_file):

        SOS_TOKEN = '<sos>'
        EOS_TOKEN = '<eos>'

        reviews_json = pd.read_json(review_json_file, lines=True)
        df1 = []
        csv_list = [["Source", "Target"]]

        self.embeddings = []

        self.SRC_REVIEWS = Field(sequential=True, tokenize=lambda text: nltk.word_tokenize(text), init_token=SOS_TOKEN,
                                 eos_token=EOS_TOKEN, lower=True)
        self.TRG_REVIEWS = Field(sequential=True, tokenize=lambda text: nltk.word_tokenize(text), init_token=SOS_TOKEN,
                                 eos_token=EOS_TOKEN, lower=True)

        for review in reviews_json['reviewText']:
            if not (pd.isna(review)):
                csv_list.append([review, review])
                review = nltk.word_tokenize(review)
                df1.append(review)

        with open('updated_csv.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(csv_list)

        tt_data = torchtext.data.TabularDataset('updated_csv.csv', format="csv", fields=[('Source', self.SRC_REVIEWS),
                                                                                         ('Target', self.TRG_REVIEWS)])

        self.SRC_REVIEWS.build_vocab(df1, min_freq=2)
        self.TRG_REVIEWS.build_vocab(df1, min_freq=2)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_iter, self.valid_iter, self.test_iter = BucketIterator.splits((tt_data, tt_data, tt_data),
                                                                                 batch_size=128,
                                                                                 device=self.device,
                                                                                 sort_key=lambda ex: len(ex.Source))

        # initialize model
        INPUT_DIM = len(self.SRC_REVIEWS.vocab)
        OUTPUT_DIM = len(self.TRG_REVIEWS.vocab)
        ENC_EMB_DIM = 256
        DEC_EMB_DIM = 256
        HID_DIM = 512
        N_LAYERS = 2
        ENC_DROPOUT = 0.5
        DEC_DROPOUT = 0.5

        encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
        decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

        self.model = Seq2Seq(encoder, decoder, self.device).to(self.device)

    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def train(model, iterator, optimizer, criterion, clip):
        model.train()

        epoch_loss = 0

        for batch in iterator:
            # print(batch)
            src = batch.Source
            trg = batch.Target
            # print(src)

            optimizer.zero_grad()
            # trg = [sen_len, batch_size]
            # output = [trg_len, batch_size, output_dim]
            output = model(src, trg)
            output_dim = output.shape[-1]

            # transfrom our output : slice off the first column, and flatten the output into 2 dim.
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            # trg = [(trg_len-1) * batch_size]
            # output = [(trg_len-1) * batch_size, output_dim]

            loss = criterion(output, trg)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(iterator)

    def evaluate(model, iterator, criterion):
        model.eval()

        epoch_loss = 0

        with torch.no_grad():
            for i, batch in enumerate(iterator):
                src = batch.Source
                trg = batch.Target

                output = model(src, trg, 0)  # turn off teacher forcing.

                # trg = [sen_len, batch_size]
                # output = [sen_len, batch_size, output_dim]
                output_dim = output.shape[-1]

                output = output[1:].view(-1, output_dim)
                trg = trg[1:].view(-1)

                loss = criterion(output, trg)

                epoch_loss += loss.item()

        return epoch_loss / len(iterator)

    # a function that used to tell us how long an epoch takes.
    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def get_embeddings(model, iter):
        embs = []
        for batch in model.train_iter:
            embs.append(model.embedding(batch.Source))

        return embs

    def embeddings(self):

        self.model.apply(self.init_weights)

        print(f'The model has {count_parameters(model):,} trainable parameters')

        optimizer = optim.Adam(self.model.parameters())
        TRG_PAD_IDX = self.TRG_REVIEWS.vocab.stoi[self.TRG_REVIEWS.pad_token]
        criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

        N_EPOCHS = 10

        CLIP = 1

        best_valid_loss = float('inf')

        for epoch in range(N_EPOCHS):

            start_time = time.time()

            train_loss = self.train(self.model, self.train_iter, optimizer, criterion=criterion, clip=CLIP)
            valid_loss = self.evaluate(self.model, self.valid_iter, criterion)

            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), 'Seq2SeqModel.pt')
            print(f"Epoch: {epoch+1:02} | Time {epoch_mins}m {epoch_secs}s")
            print(f"\tTrain Loss: {train_loss:.3f}")
            print(f"\tValid Loss: {valid_loss:.3f}")

        return self.get_embeddings(self.model, self.train_iter)
