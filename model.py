#!/usr/bin/python
#
# Adapted from code written by Robert Guthrie
# https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html#sphx-glr-beginner-nlp-word-embeddings-tutorial-py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as functional

torch.manual_seed(1)

CONTEXT_SIZE = 2
EMBEDDING_DIM = 50
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
# we should tokenize the input, but we will ignore that for now
# build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]
# print the first 3, just so you can see what they look like
# print(trigrams[:3])

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}


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


losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.05)

for epoch in range(100):
    total_loss = 0
    for context, target in trigrams:

        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in tensors)
        context_idxs = torch.tensor([word_to_ix[w]
                                     for w in context], dtype=torch.long)

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = model(context_idxs)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a tensor)
        loss = loss_function(log_probs, torch.tensor(
            [word_to_ix[target]], dtype=torch.long))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += loss.item()
    losses.append(total_loss)
# print(losses)  # The loss decreased every iteration over the training data!

sims = []
probs = []
candidates = ["Hook", "sardine"]

# calculate the probability of the word that is closer to Captain
related_embedding = model.embeddings(
    autograd.Variable(torch.LongTensor([word_to_ix["Captain"]])))

for word in candidates:
    # Probability
    probs.append(log_probs[0][word_to_ix[word]])

print("Predicted word (probability): %s" %
      (candidates[0] if probs[0] > probs[1] else candidates[1]))
