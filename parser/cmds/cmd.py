# -*- coding: utf-8 -*-

import os
from parser.utils import Embedding
from parser.utils.alg import neg_log_likelihood, directed_acyclic_graph
from parser.utils.common import pad, unk, bos, eos
from parser.utils.corpus import CoNLL, Corpus
from parser.utils.field import BertField, Field, NGramField, SegmentField
from parser.utils.fn import get_spans
from parser.utils.metric import SegF1Metric

import torch
import torch.nn as nn
from transformers import BertTokenizer


class CMD(object):

    def __call__(self, args):
        self.args = args
        if not os.path.exists(args.file):
            os.mkdir(args.file)
        if not os.path.exists(args.fields) or args.preprocess:
            print("Preprocess the data")

            self.CHAR = Field('chars', pad=pad, unk=unk,
                              bos=bos, eos=eos, lower=True)
                              
            # TODO span as label, modify chartfield to spanfield
            self.SEG = SegmentField('segs')

            if args.feat == 'bert':
                tokenizer = BertTokenizer.from_pretrained(args.bert_model)
                self.FEAT = BertField('bert',
                                      pad='[PAD]',
                                      bos='[CLS]',
                                      eos='[SEP]',
                                      tokenize=tokenizer.encode)
                self.fields = CoNLL(CHAR=(self.CHAR, self.FEAT),
                                    SEG=self.SEG)
            elif args.feat == 'bigram':
                self.BIGRAM = NGramField(
                    'bichar', n=2, pad=pad, unk=unk, bos=bos, eos=eos, lower=True)
                self.fields = CoNLL(CHAR=(self.CHAR, self.BIGRAM),
                                    SEG=self.SEG)
            elif args.feat == 'trigram':
                self.BIGRAM = NGramField(
                    'bichar', n=2, pad=pad, unk=unk, bos=bos, eos=eos, lower=True)
                self.TRIGRAM = NGramField(
                    'trichar', n=3, pad=pad, unk=unk, bos=bos, eos=eos, lower=True)
                self.fields = CoNLL(CHAR=(self.CHAR,
                                          self.BIGRAM,
                                          self.TRIGRAM),
                                    SEG=self.SEG)
            else:
                self.fields = CoNLL(CHAR=self.CHAR,
                                    SEG=self.SEG)

            train = Corpus.load(args.ftrain, self.fields)
            embed = Embedding.load(
                'data/tencent.char.200.txt',
                args.unk) if args.embed else None
            self.CHAR.build(train, args.min_freq, embed)
            if hasattr(self, 'FEAT'):
                self.FEAT.build(train)
            if hasattr(self, 'BIGRAM'):
                embed = Embedding.load(
                    'data/tencent.bi.200.txt',
                    args.unk) if args.embed else None
                self.BIGRAM.build(train, args.min_freq,
                                  embed=embed,
                                  dict_file=args.dict_file)
            if hasattr(self, 'TRIGRAM'):
                embed = Embedding.load(
                    'data/tencent.tri.200.txt',
                    args.unk) if args.embed else None
                self.TRIGRAM.build(train, args.min_freq,
                                   embed=embed,
                                   dict_file=args.dict_file)
            # TODO
            self.SEG.build(train)
            torch.save(self.fields, args.fields)
        else:
            self.fields = torch.load(args.fields)
            if args.feat == 'bert':
                self.CHAR, self.FEAT = self.fields.CHAR
            elif args.feat == 'bigram':
                self.CHAR, self.BIGRAM = self.fields.CHAR
            elif args.feat == 'trigram':
                self.CHAR, self.BIGRAM, self.TRIGRAM = self.fields.CHAR
            else:
                self.CHAR = self.fields.CHAR
            # TODO
            self.SEG = self.fields.SEG
        # TODO loss funciton 
        # self.criterion = nn.CrossEntropyLoss()
        # # [B, E, M, S]
        # self.trans = (torch.tensor([1., 0., 0., 1.]).log().to(args.device),
        #               torch.tensor([0., 1., 0., 1.]).log().to(args.device),
        #               torch.tensor([[0., 1., 1., 0.],
        #                             [1., 0., 0., 1.],
        #                             [0., 1., 1., 0.],
        #                             [1., 0., 0., 1.]]).log().to(args.device))

        args.update({
            'n_chars': self.CHAR.vocab.n_init,
            'pad_index': self.CHAR.pad_index,
            'unk_index': self.CHAR.unk_index
        })

        # TODO
        vocab = f"{self.CHAR}\n"
        if hasattr(self, 'FEAT'):
            args.update({
                'n_feats': self.FEAT.vocab.n_init,
            })
            vocab += f"{self.FEAT}\n"
        if hasattr(self, 'BIGRAM'):
            args.update({
                'n_bigrams': self.BIGRAM.vocab.n_init,
            })
            vocab += f"{self.BIGRAM}\n"
        if hasattr(self, 'TRIGRAM'):
            args.update({
                'n_trigrams': self.TRIGRAM.vocab.n_init,
            })
            vocab += f"{self.TRIGRAM}\n"

        print(f"Override the default configs\n{args}")
        print(vocab[:-1])

    def train(self, loader):
        self.model.train()

        for data in loader:
            # TODO label
            if self.args.feat == 'bert':
                chars, feats, segs = data
                feed_dict = {"chars": chars, "feats": feats}
            elif self.args.feat == 'bigram':
                chars, bigram, segs = data
                feed_dict = {"chars": chars, "bigram": bigram}
            elif self.args.feat == 'trigram':
                chars, bigram, trigram, segs = data
                feed_dict = {"chars": chars,
                             "bigram": bigram, "trigram": trigram}
            else:
                chars, segs = data
                feed_dict = {"chars": chars}

            self.optimizer.zero_grad()

            batch_size, seq_len = chars.shape
            # fenceposts length: (B)
            lens = chars.ne(self.args.pad_index).sum(1) - 1
            # TODO purpose
            # (B, 1, L-1)
            mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)
            # TODO purpose
            # for example, seq_len=10, fenceposts=7, pad=2
            # for each sentence, get a L-1*L-1 matrix
            # span (i, i) and pad are masked 
            # [[False,  True,  True,  True,  True,  True,  True, False, False],
            #  [False, False,  True,  True,  True,  True,  True, False, False],
            #  [False, False, False,  True,  True,  True,  True, False, False],
            #  [False, False, False, False,  True,  True,  True, False, False],
            #  [False, False, False, False, False,  True,  True, False, False],
            #  [False, False, False, False, False, False,  True, False, False],
            #  [False, False, False, False, False, False, False, False, False],
            #  [False, False, False, False, False, False, False, False, False],
            #  [False, False, False, False, False, False, False, False, False]]
            # (B, L-1, L-1)
            mask = mask & mask.new_ones(seq_len-1, seq_len-1).triu_(1)
            # (B, L-1, L-1)
            s_span = self.model(feed_dict)

            with torch.autograd.set_detect_anomaly(True):
                loss = self.get_loss(s_span, segs, mask)
                
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.args.clip)
            self.optimizer.step()
            self.scheduler.step()

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()

        total_loss, metric = 0, SegF1Metric()

        for data in loader:
            if self.args.feat == 'bert':
                chars, feats, segs = data
                feed_dict = {"chars": chars, "feats": feats}
            elif self.args.feat == 'bigram':
                chars, bigram, segs = data
                feed_dict = {"chars": chars, "bigram": bigram}
            elif self.args.feat == 'trigram':
                chars, bigram, trigram, segs = data
                feed_dict = {"chars": chars,
                             "bigram": bigram, "trigram": trigram}
            else:
                chars, segs = data
                feed_dict = {"chars": chars}

            # TODO mask
            mask = chars.ne(self.args.pad_index)
            # TODO
            lens = mask.sum(1).tolist()
            s_span = self.model(feed_dict)
            # TODO
            loss = self.get_loss(scores, segs, mask)

            pred_segs = self.decode(s_span, mask)
            gold_segs = segs

            total_loss += loss.item()
            metric(pred_segs, gold_segs)

        total_loss /= len(loader)

        # TODO metric
        return total_loss, "metric"

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()

        all_segs = []
        for data in loader:
            if self.args.feat == 'bert':
                chars, feats = data
                feed_dict = {"chars": chars, "feats": feats}
            elif self.args.feat == 'bigram':
                chars, bigram = data
                feed_dict = {"chars": chars, "bigram": bigram}
            elif self.args.feat == 'trigram':
                chars, bigram, trigram = data
                feed_dict = {"chars": chars,
                             "bigram": bigram, "trigram": trigram}
            else:
                chars = data
                feed_dict = {"chars": chars}
            # TODO
            mask = chars.ne(self.args.pad_index)
            s_span = self.model(feed_dict)
            # TODO
            pred_segs = directed_acyclic_graph(s_span, mask)
            # TODO
            all_segs.extend(pred_segs)
        # TODO

        return all_segs

    def get_loss(self, s_span, segs, mask):
        """crf loss

        Args:
            s_span (Tensor(B, N, N)): scores for candidate words (i, j)
            segs (Tensor(B, N, N)): groud truth words
            mask (Tensor(B, N, N)): actual 

        Returns:
            loss [type]: 
            span_probs (Tensor(B, N, N)): marginal probability for candidate words
        """

        # span_mask = spans & mask
        # span_loss, span_probs = crf(s_span, mask, spans, self.args.marg)

        loss = neg_log_likelihood(s_span, segs, mask)

        return loss

    def decode(self, s_span, mask):

        pred_spans = directed_acyclic_graph(s_span, mask)
        
        preds = pred_spans

        return preds
