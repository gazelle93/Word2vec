import argparse
from text_processing import get_nlp_pipeline, word_tokenization
import my_skipgram, my_cbow

def main(args):
    text_list = ["We are about to study the idea of a computational process.", 
             "Computational processes are abstract beings that inhabit computers.",
            "As they evolve, processes manipulate other abstract things called data.",
            "The evolution of a process is directed by a pattern of rules called a program.",
            "People create programs to direct processes.",
            "In effect, we conjure the spirits of the computer with our spells."]
    
    cur_text = "People create a computational process."
    
    selected_nlp_pipeline = get_nlp_pipeline(args.nlp_pipeline)
    
            

    # Skip-gram Embedding
    if args.encoding == "skipgram":
        vocab, input_tokens = my_skipgram.build_vocab(text_list, selected_nlp_pipeline, args.nlp_pipeline)
        word_to_ix = my_skipgram.get_word2idx(set(vocab))
        skipgram_model = my_skipgram.custom_skipgram(len(set(vocab))+1, args.hidden_dim, args.emb_dim, args.window)
        trained_model = my_skipgram.train_custom_skipgram_model(skipgram_model, input_tokens, args.window, word_to_ix)
        embeddings = my_skipgram.get_custom_word_embeddings(trained_model, cur_text, selected_nlp_pipeline, args.nlp_pipeline, word_to_ix)
        print("Customized Skip-gram Embedding Result")
        print(embeddings)

    # CBOW Embedding
    if args.encoding == "cbow":
        vocab, input_tokens = my_cbow.build_vocab(text_list, selected_nlp_pipeline, args.nlp_pipeline)
        word_to_ix = my_cbow.get_word2idx(set(vocab))
        cbow_model = my_cbow.custom_cbow(len(set(vocab))+1, args.hidden_dim, args.emb_dim, args.window)
        trained_model = my_cbow.train_custom_skipgram_model(cbow_model, input_tokens, args.window, word_to_ix)
        embeddings = my_cbow.get_custom_word_embeddings(trained_model, cur_text, selected_nlp_pipeline, args.nlp_pipeline, word_to_ix)
        print("Customized CBOW Embedding Result")
        print(embeddings)

          
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--nlp_pipeline", default="spacy", type=str, help="NLP preprocessing pipeline.")
    parser.add_argument("--encoding", default="skipgram", type=str, help="The selection of encoding method.")
    parser.add_argument("--emb_dim", default=10, type=int, help="The size of word embedding.")
    parser.add_argument("--hidden_dim", default=128, type=int, help="The size of hidden dimension.")
    parser.add_argument("--window", default=2, type=int, help="The selected window size.")
    args = parser.parse_args()

    main(args)
