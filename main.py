import math

import torch
from torch import nn
import datasets
from torch.utils.data import DataLoader
import numpy as np
from transformers import AutoTokenizer


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, d_key, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.WK = torch.nn.Linear(d_model, d_key * n_heads)
        self.WQ = torch.nn.Linear(d_model, d_key * n_heads)
        self.WV = torch.nn.Linear(d_model, d_key * n_heads)
        self.d_key = d_key
        self.n_heads = n_heads
        self.d_model = d_model
        self.WO = torch.nn.Linear(d_key * n_heads, d_model)

    def forward(self, key_input, query_input, value_input, attention_mask):
        '''

        :param
        key_input: [batch, time1, d_model]
        query_input: [batch, time2, d_model]
        value_input: [batch, time1, d_model]
        attention_mask: [batch, time1]

        :return: the same dimensions tensor as value_input
        '''
        k = self.WK(key_input)  # batch,time1,d_key*n_heads
        q = self.WQ(query_input)  # batch,time2,d_key*n_heads
        v = self.WV(value_input)  # batch, time1, d_model*heads

        batch_size = v.shape[0]

        # [batch, n_heads, time, d_key]
        q = q.view(batch_size, -1, self.n_heads, self.d_key).transpose(2, 1)  # batch, heads, time2, d_key
        k = k.view(batch_size, -1, self.n_heads, self.d_key).transpose(2, 1)  # batch, heads, time1, d_key
        v = v.view(batch_size, -1, self.n_heads, self.d_key).transpose(2, 1)  # batch, heads, time1, d_key

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_key)  # batch, heads, time2, time1
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask[:, None, None, :] == 0, float('-inf'))
        attention = torch.softmax(scores, dim=-1)  # batch, heads, time2, time1

        embeddings = attention @ v  # batch, heads, time2, d_key

        embeddings = embeddings.transpose(2, 1)  # batch, time2, d_key, heads
        embeddings = embeddings.contiguous().flatten(2)
        embeddings = self.WO(embeddings)
        return embeddings


class PositionEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super(PositionEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        exp_term = torch.arange(0, d_model, 2)
        div_term = torch.exp(exp_term * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        self.dp = nn.Dropout(p=0.1)

    def forward(self, x):
        # x[B, T, d]
        x = x + self.pe[:, :x.size(1), :]
        x = self.dp(x)
        return x


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, d_key, n_heads):
        super(TransformerBlock, self).__init__()
        self.mha = MultiHeadAttention(d_model, d_key, n_heads)
        self.norm1 = torch.nn.LayerNorm(normalized_shape=d_model)
        self.norm2 = torch.nn.LayerNorm(normalized_shape=d_model)

        self.ff = torch.nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model),
                                      nn.Dropout(p=0.1))

        self.drop = nn.Dropout(p=0.1)

    def forward(self, embeddings, attention_mask):
        # input [batch_size, time, emb]
        outputs = self.norm1(
            self.ff(self.norm1(self.mha(embeddings, embeddings, embeddings, attention_mask) + embeddings)) + embeddings)
        outputs = self.drop(outputs)
        return outputs


class Transformer(nn.Module):
    def __init__(self, d_model, d_key, n_heads, n_layers, vocab_size, max_len):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(d_model
                                                      , d_key, n_heads) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, 2)

        self.embedder = torch.nn.Embedding(vocab_size + 2, d_model)

        self.pe = PositionEncoding(max_len, d_model)
        self.dp = nn.Dropout(p=0.1)
        self.nl = nn.LayerNorm(d_model)

    def forward(self, input_dict):
        input_ids = input_dict['input_ids']
        attention_mask = input_dict['attention_mask']
        embeddings = self.embedder(input_ids)
        embeddings = self.pe(embeddings)
        for l in self.layers:
            embeddings = l(embeddings, attention_mask)
        embeddings = self.nl(embeddings)
        embeddings = self.dp(embeddings)
        logits = self.fc(embeddings[:, 0])
        return logits


if __name__ == "__main__":
    ds = datasets.load_dataset('glue', 'sst2')
    ds = ds.with_format('torch')
    train_ds = ds['train']
    val_ds = ds['validation']

    max_len = 40
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')


    def collate(samples):
        output_dict = {'labels': [], 'input_ids': [], 'attention_mask': []}
        for s in samples:
            s_encoded = tokenizer.encode_plus(s['sentence'], padding='max_length', max_length=max_len, truncation=True,
                                              return_tensors='pt')
            output_dict['labels'].append(torch.unsqueeze(s['label'], dim=0))
            for k in ['input_ids', 'attention_mask']:
                output_dict[k].append(s_encoded[k])
        output_dict = {k: torch.cat(v, dim=0).cuda()
                       for k, v in output_dict.items()}
        return output_dict


    batch_size = 32
    num_workers = 0
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate,
                                  num_workers=num_workers)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate,
                                num_workers=num_workers)

    d_model = 64
    d_key = 16
    n_heads = 4
    n_layers = 2
    epochs = 5
    print_freq = 100
    vocab_size = tokenizer.vocab_size
    model = Transformer(d_model, d_key, n_heads, n_layers, vocab_size, max_len)
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):

        model.train()
        for i, batch in enumerate(train_dataloader):
            targets = batch['labels']
            batch.pop('labels')
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(input=logits, target=targets)
            loss.backward()
            optimizer.step()
            if i % print_freq == 0:
                print(f'loss={loss.item()}')

        y_true = []
        y_pred = []
        model.eval()
        for batch in val_dataloader:
            targets = batch['labels']
            batch.pop('labels')
            with torch.no_grad():
                logits = model(batch)
            outputs = torch.softmax(logits, dim=-1)
            outputs = torch.argmax(outputs, dim=-1)
            y_true.extend(targets.detach().cpu().numpy())
            y_pred.extend(list(outputs.detach().cpu().numpy().astype(np.int)))
        val_acc = np.mean(np.equal(y_pred, y_true))
        print(f' epoch={epoch} acc={val_acc}')
