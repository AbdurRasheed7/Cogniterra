import sys
sys.path.insert(0, 'D:/ai_repro_engine')
from agents.parser_agent import parse_paper

result = parse_paper('1609.02907')
text = result.lower() if isinstance(result, str) else result['filtered_text'].lower()

nlp_triggers = [
    'attention mechanism', 'multi-head attention', 'self-attention',
    'machine translation', 'language model', 'seq2seq', 'sequence to sequence',
    'natural language processing', 'named entity recognition',
    'text classification', 'sentiment analysis', 'question answering',
    'bert', 'gpt', 'transformer', 'attention is all you need',
    'positional encoding', 'bleu', 'tokenization', 'subword',
    'encoder decoder', 'masked language', 'text generation',
    'translation', 'summarization', 'pre-training language'
]

matches = [t for t in nlp_triggers if t in text]
print("NLP triggers found in GCN paper:")
print(matches)