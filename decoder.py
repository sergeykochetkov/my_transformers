import math

import torch
from torch import nn
import datasets
from torch.utils.data import DataLoader
import numpy as np
from transformers import AutoTokenizer
from sklearn.metrics import average_precision_score
from main import TransformerBlock, PositionEncoding


class CasualSelfAttention(nn.Module):

    def __init__(self, d_model, d_key, n_heads, max_len):
        super().__init__()
        self.WK = torch.nn.Linear(d_model, d_key * n_heads)
        self.WQ = torch.nn.Linear(d_model, d_key * n_heads)
        self.WV = torch.nn.Linear(d_model, d_key * n_heads)
        self.d_key = d_key
        self.n_heads = n_heads
        self.d_model = d_model
        self.WO = torch.nn.Linear(d_key * n_heads, d_model)

        self.casual_mask = torch.tril(torch.eye(max_len, max_len)).view(1, 1, max_len, max_len)

    def forward(self, keys, queries, values, pad_mask=None):
        '''
        x: batch, time, emb
        pad_mask: [batch, time]
        '''
        keys = self.WK(keys)  # batch time d_key*n_head
        queries = self.WQ(queries)
        values = self.WV(values)
        batch_size, time = keys.shape[:2]
        keys = keys.view(batch_size, -1, self.n_heads, self.d_key).transpose(1, 2)  # batch n_head time d_key
        queries = queries.view(batch_size, -1, self.n_heads, self.d_key).transpose(1, 2)  # batch n_head time d_key
        values = values.view(batch_size, -1, self.n_heads, self.d_key).transpose(1, 2)  # batch n_head time d_key

        attn_scores = keys @ queries.transpose(2, 3) / math.sqrt(self.d_key)  # batch n_head time time

        casual_mask = self.casual_mask[:, :, :time, :time].to(attn_scores.device)
        attn_scores.masked_fill(casual_mask == 0, float('-inf'))
        if pad_mask is not None:
            attn_scores.masked_fill(pad_mask[:, None, None, :] == 0, float('-inf'))

        attn = torch.softmax(attn_scores, dim=-1)

        context_emb = attn @ values  # batch n_head time time @ batch n_head time d_key = batch n_head time d_key

        context_emb = context_emb.transpose(1, 2).contiguous().view(batch_size, time, -1)  # batch n_head time d_key

        context_emb = self.WO(context_emb)
        return context_emb


class Decoder(nn.Module):
    def __init__(self, d_model, d_key, n_heads, n_layers, max_len, vocab_size):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, CasualSelfAttention(d_model, d_key, n_heads, max_len)) for _ in range(n_layers)])
        self.embedder = nn.Embedding(vocab_size, d_model)
        self.pe = PositionEncoding(max_len, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.dp = nn.Dropout(p=0.1)
        self.fc = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, vocab_size))

    def forward(self, input_dict):
        input_ids = input_dict['input_ids']
        attention_mask = input_dict['attention_mask']
        embeddings = self.embedder(input_ids)
        embeddings = self.pe(embeddings)
        for l in self.layers:
            embeddings = l(embeddings, attention_mask)
        embeddings = self.ln(embeddings)
        embeddings = self.dp(embeddings)
        logits = self.fc(embeddings)
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
    model = Decoder(d_model, d_key, n_heads, n_layers, max_len, vocab_size)
    model.cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):

        model.train()
        for i, batch in enumerate(train_dataloader):
            batch.pop('labels')
            targets = batch['input_ids']
            targets = torch.roll(targets, shifts=-1, dims=1)
            targets[:, -1] = tokenizer.pad_token_id
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(input=logits.view(-1, vocab_size), target=targets.view(-1))
            loss.backward()
            optimizer.step()
            if i % print_freq == 0:
                print(f'loss={loss.item()}')

        y_true = []
        y_pred = []
        model.eval()
        for batch in val_dataloader:
            batch.pop('labels')
            targets = batch['input_ids']
            targets = torch.roll(targets, shifts=-1, dims=1)
            targets[:, -1] = tokenizer.pad_token_id
            with torch.no_grad():
                logits = model(batch)
            outputs = torch.softmax(logits, dim=-1)
            outputs = torch.argmax(outputs, dim=-1)
            y_true.extend(targets.detach().cpu().numpy())
            y_pred.extend(list(outputs.detach().cpu().numpy().astype(np.int)))
        val_acc = np.mean(np.equal(y_pred, y_true))
        print(f' epoch={epoch} acc={val_acc}')
