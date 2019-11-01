import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as functional
from hyperopt import fmin, tpe
from hyperopt import hp
import hyperopt

torch.manual_seed(1)

def preprocess():
    test_sentence = """Captain Hook must remember
    Not to scratch his toes.
    Captain Hook must watch out
    And never pick his nose.
    Captain Hook must be gentle
    When he shakes your hand.
    Captain Hook must be careful
    Openin' sardine cans
    And playing tag and pouring tea
    and turnin' pages of this book.
    Lots of folks I'm glad I ain'tâ€”
    But mostly Captain Hook!""".split()
    trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2]) for i in range(len(test_sentence) - 2)]
    vocab = set(test_sentence)
    word_to_ix = {word: i for i, word in enumerate(vocab)}

    return trigrams, vocab, word_to_ix


class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)
    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


def objective(args):
    trigrams, vocab, word_to_ix = preprocess()
    losses = []
    loss_function = nn.NLLLoss()

    # adjusting params & using params to retrain
    lrate = args['lr']

    model = NGramLanguageModeler(len(vocab), 50, 2)
    optimizer = optim.SGD(model.parameters(), lr=lrate)

    for epoch in range(100):
        total_loss = 0
        for context, target in trigrams:
            context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
            model.zero_grad()
            log_probs = model(context_idxs)
            loss = loss_function(log_probs, torch.tensor(
                [word_to_ix[target]], dtype=torch.long))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss)
    # print(losses)  # The loss decreased every iteration over the training data!

    sims = []
    probs = []
    candidates = ["Hook", "sardine"]

    related_embedding = model.embeddings(autograd.Variable(torch.LongTensor([word_to_ix["Captain"]])))

    for word in candidates:
        probs.append(log_probs[0][word_to_ix[word]])
    print("Min loss: {}".format(str(losses[-1])))
    print("Predicted word (probability): %s\n" % (candidates[0] if probs[0] > probs[1] else candidates[1]))
    return losses[-1]


#######################################################################

if __name__ == '__main__':
    # search space
    space = {
        'lr': hp.choice('lr', [0.0, 0.0001, 0.005, 0.01, 0.04, 0.05, 0.08, 0.1]),
    }
    # fmin function
    best = fmin(objective, space, algo=hyperopt.rand.suggest, max_evals=10)
    print('\n\nBest values for params: \n')
    print(hyperopt.space_eval(space, best))
    print('')
