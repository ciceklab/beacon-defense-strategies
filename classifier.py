import torch.nn as nn
import torch

class OnlineDefender:
    def __init__(self, classifier, device="cuda", threshold=0.05):
        self.classifier = classifier.to(device)
        # self.threshold = threshold
        self.device = device

        # keep a buffer of past queries
        self.history = []

    def reset(self):
        """Reset history (e.g., new attacker session)."""
        self.history = []
        self.classifier.eval()

    def process_query(self, new_query):
        """
        new_query: tensor of shape (input_dim,)
        Returns: decision (1 = honest, 0 = lie)
        """

        # self.classifier.eval()
        # print(f"New query: {new_query}\n Shape: {new_query.shape}")
        new_query = new_query.unsqueeze(0).unsqueeze(0)
        # shape: (1, 1, input_dim)

        if len(self.history) == 0:
            # no history yet, prob_before = 0
            prob_before = torch.tensor([0.0], device=self.device)
            seq_before = torch.zeros((1, 0, new_query.size(-1)), device=self.device)
        else:
            seq_before = torch.stack(self.history, dim=1)  # (1, seq_len, input_dim)
            prob_before, _ = self.classifier(seq_before)

        # add new query
        seq_after = torch.cat([seq_before, new_query], dim=1)
        prob_after, _ = self.classifier(seq_after)

        # store the new query in history (regardless of decision)
        self.history.append(new_query.squeeze(0))  

        # print(f"Prob before: {prob_before}\n after: {prob_after}\n history: {self.history}")

        # TODO: this may need to be changed
        return (prob_after - prob_before)

        # decision rule
        # if (prob_after - prob_before) > self.threshold:
        #     return 0  # lie
        # else:
        #     return 1  # honest

    def eval_mode(self):
        self.classifier.eval()

    def train_mode(self):
        self.classifier.train()

class IdentificationRNN(nn.Module):
    def __init__(self, query_dim, hidden_dim=64, num_layers=1):
        super(IdentificationRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # RNN backbone (GRU is simpler & stable, LSTM also possible)
        self.rnn = nn.GRU(
            input_size=query_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Classifier head
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden=None):
        # x: [batch=1, seq=1, query_dim]
        out, hidden = self.rnn(x, hidden)  # out: [1,1,hidden_dim]
        out = out[:, -1, :]                # take last timestep [1, hidden_dim]

        logits = self.fc(out)              # [1,1]
        prob = self.sigmoid(logits)        # probability in [0,1]

        return prob.squeeze(1), hidden     # prob: [1], hidden: for next step
    
# class IdentificationRNN(nn.Module):
#     def __init__(self, input_dim, hidden_dim=64, num_layers=1):
#         super().__init__()
#         self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, 1)
#         self.sigmoid = nn.Sigmoid()
        
#     def forward(self, x):
#         # x: (batch, seq_len, input_dim)
#         h, _ = self.rnn(x)
#         out = self.fc(h[:, -1, :])   # only use last hidden state
#         return self.sigmoid(out).squeeze(-1)  # (batch,)

# class RealTimeQueryClassifier(nn.Module):
#     def __init__(self, vocab_size, emb_dim, hidden_dim, num_stats):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, emb_dim)
#         self.lstm = nn.LSTM(emb_dim + 1 + num_stats, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, 1)

#     def forward(self, x_step, h=None):
#         """
#         x_step: (batch, 1, features) → just the new query step
#         h: previous hidden state (h, c) or None
#         """
#         pos = x_step[:, :, 0].long()
#         ans = x_step[:, :, 1].unsqueeze(-1)
#         stats = x_step[:, :, 2:]
#         emb = self.embedding(pos)
#         inp = torch.cat([emb, ans, stats], dim=2)
#         out, h_new = self.lstm(inp, h)   # process 1 timestep
#         logits = self.fc(out).squeeze(-1)  # (batch, 1)
#         probs = torch.sigmoid(logits)      # attacker probability
#         return probs, h_new
