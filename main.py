import torch
from torch import nn
import datasets
from torch.utils.data import DataLoader
import numpy as np
from transformers import AutoTokenizer


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, model_d, key_dim, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.WK = torch.nn.Parameter(torch.zeros((1, model_d, key_dim * n_heads), dtype=torch.float))
        self.WQ = torch.nn.Parameter(torch.zeros((1, model_d, key_dim * n_heads), dtype=torch.float))
        self.WV = torch.nn.Parameter(torch.zeros((1, model_d, model_d * n_heads), dtype=torch.float))
        self.key_dim = key_dim
        self.n_heads = n_heads
        self.model_d = model_d
        self.WO = torch.nn.Linear(n_heads, 1)

    def forward(self, key_input, query_input, value_input, attention_mask):
        '''

        :param input: [batch, time, model_d]
        :return: the same dimensions tensor
        '''
        keys = torch.matmul(key_input, self.WK)  # batch,time,key_dim*n_heads
        querys = torch.matmul(query_input, self.WQ)  # batch,time,key_dim*n_heads
        values = torch.matmul(value_input, self.WV)  # batch, time, model_d*heads

        batch_size = values.shape[0]
        querys = torch.transpose(querys, 2, 1)  # batch, key_dim*heads, time
        querys = torch.reshape(querys, (batch_size * self.n_heads, self.key_dim, -1))  # batch*heads, key_dim, time

        keys = torch.transpose(keys, 2, 1)  # batch, key_dim*heads, time
        keys = torch.reshape(keys, (batch_size * self.n_heads, self.key_dim, -1))  # batch*heads, key_dim, time
        keys = torch.transpose(keys, 2, 1)  # batch*nheads, time, key_dim

        key_query = torch.matmul(keys, querys)  # batch*nheads, time, time
        attention = torch.softmax(key_query, dim=1)

        values = torch.transpose(values, 2, 1)  # batch, model_d*heads, time
        values = torch.reshape(values, (batch_size * self.n_heads, self.model_d, -1))  # batch*heads, model_d, time

        embeddings = torch.matmul(values, attention)  # batch*heads, model_d, time

        embeddings = torch.reshape(embeddings, (batch_size, self.n_heads, self.model_d, -1))
        embeddings = torch.transpose(embeddings, 3, 1)  # batch, time, model_d, heads
        embeddings = self.WO(embeddings)
        embeddings = torch.squeeze(embeddings, dim=-1)
        return embeddings


class TransformerBlock(torch.nn.Module):
    def __init__(self, model_d, key_dim, n_heads):
        super(TransformerBlock, self).__init__()
        self.mha = MultiHeadAttention(model_d, key_dim, n_heads)
        self.norm1 = torch.nn.LayerNorm(normalized_shape=model_d)
        self.norm2 = torch.nn.LayerNorm(normalized_shape=model_d)

        self.ff = torch.nn.Sequential(nn.Linear(model_d, model_d), nn.ReLU(), nn.Linear(model_d, model_d))

    def forward(self, embeddings, attention_mask):
        # input [batch_size, time, emb]
        outputs = self.norm1(
            self.ff(self.norm1(self.mha(embeddings, embeddings, embeddings, attention_mask) + embeddings)) + embeddings)
        return outputs


class Transformer(nn.Module):
    def __init__(self, model_d, key_dim, n_heads, n_layers, vocab_size):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(model_d
                                                      , key_dim, n_heads) for _ in range(n_layers)])
        self.fc = nn.Linear(model_d, 1)

        self.embedder = torch.nn.Embedding(vocab_size + 2, model_d)

    def forward(self, input_dict):
        input_ids = input_dict['input_ids']
        attention_mask = input_dict['attention_mask']
        embeddings = self.embedder(input_ids)
        for l in self.layers:
            embeddings = l(embeddings, attention_mask)
        logits = self.fc(embeddings)
        return logits


if __name__ == "__main__":
    ds = datasets.load_dataset('glue', 'sst2')
    ds = ds.with_format('torch')
    train_ds = ds['train']
    val_ds = ds['validation']

    max_len = 510
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')


    def collate(samples):
        output_dict = {'labels': [], 'input_ids': [], 'attention_mask': []}
        for s in samples:
            s_encoded = tokenizer.encode_plus(s['sentence'], padding='max_length', max_length=max_len, truncation=True,
                                              return_tensors='pt')
            output_dict['labels'].append(torch.unsqueeze(s['label'], dim=0))
            for k in ['input_ids', 'attention_mask']:
                output_dict[k].append(s_encoded[k])
        output_dict = {k: torch.cat(v, dim=0)
                       for k, v in output_dict.items()}
        return output_dict


    train_dataloader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate, num_workers=0)
    val_dataloader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate, num_workers=0)

    model_d = 64
    key_dim = 32
    n_heads = 4
    n_layers = 5
    epochs = 5

    vocab_size = tokenizer.vocab_size
    model = Transformer(model_d, key_dim, n_heads, n_layers, vocab_size)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        for batch in train_dataloader:
            targets = batch['labels']
            batch.pop('labels')
            logits = model(batch)
            loss = criterion(input=logits, target=targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        y_true = []
        y_pred = []
        for batch in val_dataloader:
            inputs = batch['sentence']
            targets = batch['label']
            logits = model(inputs)
            outputs = torch.sigmoid(logits)
            outputs = outputs > 0.5
            y_true.extend(targets)
            y_pred.extend(outputs)
        val_acc = np.mean(y_pred == y_pred)
        print(f' epoch={epoch} acc={val_acc}')
