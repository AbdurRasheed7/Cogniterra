import sys
sys.path.insert(0, 'D:/ai_repro_engine')
from agents.parser_agent import parse_paper

text = parse_paper('1708.05031').lower()

nlp_triggers = [
    "attention mechanism", "multi-head attention", "self-attention",
    "machine translation", "language model", "seq2seq", "sequence to sequence",
    "natural language processing", "named entity recognition",
    "sentiment analysis", "question answering",
    "attention is all you need", "positional encoding",
    "neural machine translation", "bleu score",
    "tokenization", "subword", "masked language model",
    "text generation task", "pre-training language",
    "language understanding", "bert pre-training",
    "encoder decoder attention", "transformer model for"
]

rec_triggers = [
    "recommendation system", "recommender system",
    "collaborative filtering", "matrix factorization",
    "user-item interaction", "rating prediction",
    "personalized recommendation", "top-k recommendation",
    "movielens", "implicit feedback", "explicit feedback",
    "click through rate", "cold start problem"
]

print("NLP triggers found:")
print([t for t in nlp_triggers if t in text])
print("\nREC triggers found:")
print([t for t in rec_triggers if t in text])