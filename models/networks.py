import torch.nn as nn
import numpy as np
import torch

from torch.autograd import Variable
from sklearn.metrics import roc_auc_score

from embeddings import Embeddings

class SocialLSTM(nn.Module):
    """
    LSTM model for predicting conflict between Reddit communities.
    Can incorporate social embeddings of users and communities/subreddits.
    """

    def _load_glove_embeddings(self):
        print("Loading word embeddings...")
        with open(self.config['paths']['WORD_EMBEDS']) as fp:
            embeddings = np.empty((self.config['constants']['VOCAB_SIZE'], self.config['constants']['WORD_EMBED_DIM']), dtype=np.float32)
            for i, line in enumerate(fp):
                embeddings[i, :] = list(map(float, line.split()[1:]))
        return embeddings

    def _load_user_embeddings(self):
        print("Loading user embeddings...")
        embeds = Embeddings(self.config['paths']['USER_EMBEDS'])
        return embeds._vecs

    def _load_subreddit_embeddings(self):
        print("Loading subreddit embeddings...")
        embeds = Embeddings(self.config['paths']['SUBREDDIT_EMBEDS'])
        return embeds._vecs

    def __init__(self, config, prepend_social=True):
        """
        Initializes the SocialLSTM model based on the configuration.
        """
        super(SocialLSTM, self).__init__()
        self.config = config
        batch_size = config['constants']['BATCH_SIZE']
        hidden_dim = config['training']['hidden_dim']
        dropout = None if config['training']['single_layer'] else config['training']['dropout']

        glove_embeds = self._load_glove_embeddings()
        self.glove_embeds = torch.FloatTensor(glove_embeds)
        self.pad_embed = torch.zeros(1, config['constants']['WORD_EMBED_DIM'])
        self.unk_embed = torch.FloatTensor(1, config['constants']['WORD_EMBED_DIM'])
        self.unk_embed.normal_(std=1. / np.sqrt(config['constants']['WORD_EMBED_DIM']))
        self.word_embeds = nn.Parameter(torch.cat([self.glove_embeds, self.pad_embed, self.unk_embed], dim=0),
                                        requires_grad=False)
        self.embed_module = nn.Embedding(config['constants']['VOCAB_SIZE'] + 2, config['constants']['WORD_EMBED_DIM'])
        self.embed_module.weight = self.word_embeds

        user_embeds = self._load_user_embeddings()
        self.user_embeds = nn.Embedding(config['constants']['NUM_USERS'] + 1, config['constants']['WORD_EMBED_DIM'])
        self.user_embeds.weight = nn.Parameter(torch.cat([torch.FloatTensor(user_embeds), self.pad_embed]), requires_grad=False)

        subreddit_embeds = self._load_subreddit_embeddings()
        self.subreddit_embeds = nn.Embedding(config['constants']['NUM_SUBREDDITS'] + 1, config['constants']['WORD_EMBED_DIM'])
        self.subreddit_embeds.weight = nn.Parameter(torch.cat([torch.FloatTensor(subreddit_embeds), self.pad_embed]), requires_grad=False)

        self.hidden_dim = hidden_dim
        self.prepend_social = prepend_social

        init_hidden_data = torch.zeros(1 if dropout is None else 2, batch_size, self.hidden_dim)
        if config['settings']['CUDA']:
            init_hidden_data = init_hidden_data.cuda()
        self.init_hidden = (Variable(init_hidden_data, requires_grad=False),
                            Variable(init_hidden_data, requires_grad=False))

        self.rnn = nn.LSTM(input_size=config['constants']['WORD_EMBED_DIM'], hidden_size=hidden_dim,
                           num_layers=1 if dropout is None else 2, dropout=0. if dropout is None else dropout)

        self.final_dense = config['training']['final_dense']
        self.include_meta = config['training']['include_meta']
        self.include_embeds = config['training']['final_layer_social']
        out_layer1_outdim = self.hidden_dim if self.final_dense else config['constants']['NUM_CLASSES']
        # input_dim = self.hidden_dim
        if self.include_meta and self.include_embeds: 
            self.out_layer1 = nn.Linear(self.hidden_dim+config['constants']['SF_LEN']+3*config['constants']['WORD_EMBED_DIM'], out_layer1_outdim)
        elif self.include_embeds:
            self.out_layer1 = nn.Linear(self.hidden_dim+3*config['constants']['WORD_EMBED_DIM'], out_layer1_outdim)
        elif self.include_meta:
            self.out_layer1 = nn.Linear(self.hidden_dim+config['constants']['SF_LEN'], out_layer1_outdim)
        else:
            self.out_layer1 = nn.Linear(self.hidden_dim, out_layer1_outdim)
        if self.final_dense:
            self.relu = nn.Tanh()
            self.out_layer2 = nn.Linear(self.hidden_dim, config['constants']['NUM_CLASSES'])

    def forward(self, text_inputs, user_inputs, subreddit_inputs, metafeats, lengths):
        text_inputs = self.embed_module(text_inputs)
        user_inputs = self.user_embeds(user_inputs)
        subreddit_inputs = self.subreddit_embeds(subreddit_inputs)
        if self.prepend_social is True:
            inputs = torch.cat([user_inputs, subreddit_inputs, text_inputs], dim=0)
        elif self.prepend_social is False:
            inputs = torch.cat([text_inputs, user_inputs, subreddit_inputs], dim=0)
        else:
            inputs = text_inputs
            lengths = [l - 3 for l in lengths]

        inputs = nn.utils.rnn.pack_padded_sequence(inputs, lengths)
        outputs, h = self.rnn(inputs, self.init_hidden)

        h, lengths = nn.utils.rnn.pad_packed_sequence(outputs)
        h = h.sum(dim=0).squeeze()
        lengths = lengths.clone().detach()
        if self.config['settings']['CUDA']:
            lengths = lengths.cuda()
        h = h.t().div(Variable(lengths))
        self.h = h

        final_input = h.t()
        if self.include_meta:
            final_input = torch.cat([final_input, metafeats.t()], dim=1)
        if self.include_embeds:
            final_input = torch.cat([final_input, user_inputs.squeeze(), subreddit_inputs[0], subreddit_inputs[1]], dim=1)
        if not self.final_dense:
            weights = self.out_layer1(final_input)
        else:
            weights = self.out_layer2(self.relu(self.out_layer1(final_input)))
        return weights