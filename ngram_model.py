import random
from tqdm import tqdm
from collections import defaultdict, Counter
from nltk.tokenize import WordPunctTokenizer, word_tokenize
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import sentencepiece as spm
import os
import pickle
from transformers import AutoTokenizer
from transformers import BertTokenizer

from tokenizers import Tokenizer
from tokenizers.models import BPE

UNK, EOS = "_UNK_", "_EOS_"

# Function to count n-grams
def count_ngrams(tokenized_lines, n):
    counts = defaultdict(Counter)
    for tokens in tqdm(tokenized_lines):
        tokens.append(EOS)
        for i in range(len(tokens)):
            prefix = tuple(tokens[max(0, i-n+1):i])
            if len(prefix) < n-1:
                prefix = (UNK,) * (n-1-len(prefix)) + prefix
            next_token = tokens[i]
            counts[prefix][next_token] += 1
    return counts

# N-gram language model class
class NGramLanguageModel:
    def __init__(self, tokenized_lines, n):
        assert n >= 1
        self.n = n
        counts = count_ngrams(tokenized_lines, self.n)
        self.probs = defaultdict(Counter)
        for prefix, token_counts in counts.items():
            count_sum = sum(token_counts.values())
            for token, count in token_counts.items():
                self.probs[prefix][token] = count / count_sum

    def get_possible_next_tokens(self, prefix):
        prefix = prefix.split()
        prefix = prefix[max(0, len(prefix) - self.n + 1):]
        prefix = [UNK] * (self.n - 1 - len(prefix)) + prefix
        return self.probs[tuple(prefix)]

    def get_next_token_prob(self, prefix, next_token):
        return self.get_possible_next_tokens(prefix).get(next_token, 0)
    
# Function to remove special tokens (e.g., Ġ)
def clean_bpe_tokens(tokens):
    return [token.replace('Ġ', ' ').strip() for token in tokens]

# Function to get tokens using different tokenizers
def get_tokens(name, lines):
    if name == "WordPunctTokenizer":
        tokenizer = WordPunctTokenizer()
        return [tokenizer.tokenize(line.lower()) for line in lines]
    elif name == "word_tokenize":
        return [word_tokenize(line.lower()) for line in lines]
#    elif name == "BPE":
#        tokenizer = Tokenizer(models.BPE())
#        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
#        trainer = trainers.BpeTrainer(vocab_size=1000, min_frequency=2)
#        tokenizer.train_from_iterator(lines, trainer)
#        return [tokenizer.encode(line.lower()).tokens for line in lines]
#    elif name == "BPE":
#        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
   
#        return [tokenizer.tokenize(line.lower()) for line in lines]
    elif name == "BPE":
        tokenizer = AutoTokenizer.from_pretrained("gpt2")  # GPT-2 uses BPE
        return [clean_bpe_tokens(tokenizer.tokenize(line.lower())) for line in lines]
    elif name == "Wordpiece":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        return [tokenizer.tokenize(line.lower()) for line in lines]
    else:
        raise ValueError("Invalid tokenizer name")

# Train and save models
def train_and_save_models(lines, model_dir):
    models = {}
    for tokenizer_name in ["WordPunctTokenizer", "word_tokenize", "BPE","Wordpiece"]:
        tokenized_lines = get_tokens(tokenizer_name, lines)
        model = NGramLanguageModel(tokenized_lines, n=2)
        models[tokenizer_name] = model
        with open(os.path.join(model_dir, f"{tokenizer_name}_model.pkl"), "wb") as f:
            pickle.dump(model, f)

# Load models
def load_models(model_dir):
    models = {}
    for tokenizer_name in ["WordPunctTokenizer", "word_tokenize", "BPE","Wordpiece"]:
        with open(os.path.join(model_dir, f"{tokenizer_name}_model.pkl"), "rb") as f:
            models[tokenizer_name] = pickle.load(f)
    return models

# Function to get the next token
def get_next_token(lm, prefix, use_highest_prob=True):
    possible_next_tokens = lm.get_possible_next_tokens(prefix)
    if not possible_next_tokens:
        return EOS, []
    top_five = possible_next_tokens.most_common(5)
    words = [item[0] for item in top_five]
    if use_highest_prob:
        highest_prob_token = max(possible_next_tokens, key=possible_next_tokens.get)
        return highest_prob_token, words
    else:
        # Randomly select from top 5 words
        selected_token = random.choice(words)
        return selected_token, words

# Function to generate next lines
def next_lines(prefix, lm, use_highest_prob=True):
    if not prefix:
        return "Input cannot be empty.", []
    prefix = prefix
    for i in range(1):
        next_token, top_five = get_next_token(lm, prefix, use_highest_prob)
        prefix += ' ' + next_token
        if prefix.endswith(EOS) or len(lm.get_possible_next_tokens(prefix)) == 0:
            break
    return prefix, top_five

# Function to preview tokenizer with an image
def preview_tokenizer(name):
    if name == "WordPunctTokenizer":
        return "preview_images/wordpuctokenizer.png"
    elif name == "word_tokenize":
        return "preview_images/word_tokenizer.png"
    elif name == "BPE":
        return "preview_images/BPE.png"
    elif name == 'Wordpiece':
        return "preview_images/worpiece.png"
    else:
        raise ValueError("Invalid tokenizer name")
