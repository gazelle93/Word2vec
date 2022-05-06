import torch
import torch.nn as nn
from text_processing import get_nlp_pipeline, word_tokenization

# Custom CBOW
def get_word2idx(vocab):
    word_to_ix = {}
    word_to_ix['PAD'] = 0
    for idx, word in enumerate(vocab):
        word_to_ix[word] = idx+1
    return word_to_ix

def get_idx2word(vocab):
    ix_to_word = {}
    ix_to_word[0] = 'PAD'
    for idx, word in enumerate(vocab):
        ix_to_word[idx+1] = word
    return ix_to_word

def build_input_output(_input_text, window):
    io_pair = []

    for _input_tokens in _input_text:
        for idx, word in enumerate(_input_tokens):
            context = []
            start = idx - window
            end = idx + window + 1

            for cur in range(start,end):
                if cur < 0:
                    context.append('PAD')
                if cur != idx and cur >= 0 and cur < len(_input_tokens):
                    context.append(_input_tokens[cur])
                if cur >= len(_input_tokens):
                    context.append('PAD')

            io_pair.append([context, word])
    return io_pair

def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)


class custom_cbow(torch.nn.Module):
    def __init__(self, vocab_size, hidden_dim, embedding_dim, window):
        super(custom_cbow, self).__init__()

        # initialize lookup table
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.window = window

        # projection layer
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.activation_function1 = nn.ReLU()

        # output layer
        self.linear2 = nn.Linear(hidden_dim, vocab_size)
        self.activation_function2 = nn.LogSoftmax(dim=-1)


    def forward(self, _inputs):
        embeds = sum(self.embeddings(_inputs)).view(1,-1)

        out = self.linear1(embeds)
        projected_out = self.activation_function1(out)
        output = self.linear2(projected_out)
        log_probs = self.activation_function2(output)
        return projected_out, log_probs


def build_vocab(text_list, selected_nlp_pipeline, nlp_pipeline):
    input_tokens = []
    vocab = []
    for _text in text_list:
        tokenized_text = word_tokenization(_text, selected_nlp_pipeline, nlp_pipeline)
        vocab += tokenized_text
        input_tokens.append(tokenized_text)

    return vocab, input_tokens


def train_custom_skipgram_model(model, input_tokens, window, word_to_ix):
    input_output_context = build_input_output(input_tokens, window)
    loss_function = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(10):
        total_loss = 0

        for context, target in input_output_context:
            context_vector = make_context_vector(context, word_to_ix)
            _, log_probs = model(context_vector)
            total_loss += loss_function(log_probs, torch.tensor([word_to_ix[target]]))

        #optimize at the end of each epoch
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return model

def get_custom_word_embeddings(model, cur_text, selected_nlp_pipeline, nlp_pipeline, word_to_ix):
    tks = word_tokenization(cur_text, selected_nlp_pipeline, nlp_pipeline)
    embeddings = list(model.parameters())[0]
    embeddings = embeddings.cpu().detach()
    result = []
    for tk in tks:
        result.append(embeddings[word_to_ix[tk]])
    return result
