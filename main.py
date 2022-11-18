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
        self.WV = torch.nn.Linear(d_model, d_model * n_heads)
        self.d_key = d_key
        self.n_heads = n_heads
        self.d_model = d_model
        self.WO = torch.nn.Linear(d_model * n_heads, d_model)

    def forward(self, key_input, query_input, value_input, attention_mask):
        '''

        :param
        key_input: [batch, time1, d_model]
        query_input: [batch, time2, d_model]
        value_input: [batch, time1, d_model]

        :return: the same dimensions tensor as value_input
        '''
        keys = self.WK(key_input)  # batch,time1,d_key*n_heads
        querys = self.WQ(query_input)  # batch,time2,d_key*n_heads
        values = self.WV(value_input)  # batch, time1, d_model*heads

        batch_size = values.shape[0]
        querys = torch.transpose(querys, 2, 1)  # batch, d_key*heads, time2
        querys = torch.reshape(querys, (batch_size * self.n_heads, self.d_key, -1))  # batch*heads, d_key, time2

        keys = torch.transpose(keys, 2, 1)  # batch, d_key*heads, time1
        keys = torch.reshape(keys, (batch_size * self.n_heads, self.d_key, -1))  # batch*heads, d_key, time1
        keys = torch.transpose(keys, 2, 1)  # batch*nheads, time1, d_key

        key_query = torch.matmul(keys, querys)  # batch*nheads, time1, time2
        attention = torch.softmax(key_query, dim=1)  # batch*nheads, time1, time2

        values = torch.transpose(values, 2, 1)  # batch, d_model*heads, time1
        values = torch.reshape(values, (batch_size * self.n_heads, self.d_model, -1))  # batch*heads, d_model, time1

        embeddings = torch.matmul(values, attention)  # batch*heads, d_model, time2

        embeddings = torch.reshape(embeddings, (batch_size, self.n_heads, self.d_model, -1))
        embeddings = torch.transpose(embeddings, 3, 1)  # batch, time2, d_model, heads
        embeddings = torch.flatten(embeddings, 2)
        embeddings = self.WO(embeddings)
        return embeddings


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, d_key, n_heads):
        super(TransformerBlock, self).__init__()
        self.mha = MultiHeadAttention(d_model, d_key, n_heads)
        self.norm1 = torch.nn.LayerNorm(normalized_shape=d_model)
        self.norm2 = torch.nn.LayerNorm(normalized_shape=d_model)

        self.ff = torch.nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

    def forward(self, embeddings, attention_mask):
        # input [batch_size, time, emb]
        outputs = self.norm1(
            self.ff(self.norm1(self.mha(embeddings, embeddings, embeddings, attention_mask) + embeddings)) + embeddings)
        return outputs


class Transformer(nn.Module):
    def __init__(self, d_model, d_key, n_heads, n_layers, vocab_size):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(d_model
                                                      , d_key, n_heads) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, 1)

        self.embedder = torch.nn.Embedding(vocab_size + 2, d_model)

    def forward(self, input_dict):
        input_ids = input_dict['input_ids']
        attention_mask = input_dict['attention_mask']
        embeddings = self.embedder(input_ids)
        for l in self.layers:
            embeddings = l(embeddings, attention_mask)
        logits = self.fc(embeddings[:, 0])
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
        output_dict = {k: torch.cat(v, dim=0).cuda()
                       for k, v in output_dict.items()}
        return output_dict


    batch_size = 128
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)

    d_model = 64
    d_key = 32
    n_heads = 4
    n_layers = 5
    epochs = 5
    print_freq = 100
    vocab_size = tokenizer.vocab_size
    model = Transformer(d_model, d_key, n_heads, n_layers, vocab_size)
    model.cuda()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(epochs):

        model.train()
        for i, batch in enumerate(train_dataloader):
            targets = batch['labels']
            batch.pop('labels')
            logits = model(batch)
            loss = criterion(input=logits.squeeze(-1), target=targets.float())
            optimizer.zero_grad()
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
            outputs = torch.sigmoid(logits)
            outputs = outputs > 0.5
            y_true.extend(targets.detach().cpu().numpy())
            y_pred.extend(list(outputs.detach().cpu().numpy().astype(np.int).squeeze(-1)))
        val_acc = np.mean(np.equal(y_pred, y_true))
        print(f' epoch={epoch} acc={val_acc}')
        print(f' epoch={epoch} acc={val_acc}')
