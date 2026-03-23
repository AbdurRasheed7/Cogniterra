import sys, re
sys.path.insert(0, 'D:/ai_repro_engine')
from agents.parser_agent import parse_paper

text = parse_paper('1202.2745').lower()

checks = {
    "graph": [
        "graph convolutional network", "graph neural network",
        "node classification", "message passing", "adjacency matrix",
        "graph convolution", "spectral graph", "graph laplacian",
        "semi-supervised classification with graph",
        "gcn", "gnn", "graph attention", "graphsage",
        "graph isomorphism", "molecular graph",
        "cora", "citeseer", "pubmed dataset", "citation network"
    ],
    "CV_precheck": [
        "image classification on imagenet",
        "top-1 accuracy", "top-5 accuracy",
        "vgg very deep", "very deep convolutional networks for large-scale",
        "mobilenet", "efficientnet",
        "densely connected convolutional", "densenet",
        "feature pyramid network", "object detection with deep"
    ],
    "recommendation": [
        "recommendation system", "recommender system",
        "collaborative filtering", "matrix factorization",
        "user-item interaction", "rating prediction",
        "movielens", "implicit feedback", "explicit feedback",
        "personalized recommendation", "top-k recommendation",
        "click through rate", "cold start problem"
    ],
    "nlp": [
        "attention mechanism", "multi-head attention", "self-attention",
        "machine translation", "language model", "seq2seq",
        "natural language processing", "named entity recognition",
        "sentiment analysis", "question answering",
        "attention is all you need", "positional encoding",
        "neural machine translation", "bleu score",
        "tokenization", "subword", "masked language model",
        "text generation task", "pre-training language",
        "language understanding", "bert pre-training",
        "encoder decoder attention", "transformer model for"
    ],
    "rl_long": [
        "reinforcement learning", "reward function", "policy gradient",
        "q-learning", "markov decision", "actor-critic",
        "openai gym", "action space", "state space", "episode reward",
        "value function", "temporal difference", "replay buffer",
        "epsilon greedy", "discount factor",
        "proximal policy", "trust region", "gymnasium", "rollout"
    ],
    "rl_short_wordboundary": ["dqn", "ppo", "ddpg", "a3c", "a2c"],
    "generative": [
        "generative adversarial", "variational autoencoder",
        "image generation", "gan", "vae", "diffusion model",
        "denoising diffusion", "score matching", "image synthesis",
        "style transfer", "super resolution", "normalizing flow",
        "ddpm", "stable diffusion", "noise prediction", "latent diffusion",
        "generator network", "discriminator network"
    ],
    "algorithm": [
        "sorting algorithm", "dynamic programming", "binary search",
        "tree traversal", "shortest path", "time complexity",
        "space complexity", "big o notation", "data structure",
        "hash table", "divide and conquer", "backtracking"
    ],
    "image_classification": [
        "image classification", "object detection", "image recognition",
        "convolutional neural network", "resnet", "vgg", "alexnet",
        "mnist", "cifar", "imagenet", "feature maps", "pooling layer",
        "conv2d", "residual block", "skip connection", "dense block",
        "semantic segmentation", "instance segmentation", "bounding box",
        "feature pyramid", "object recognition", "visual recognition",
        "convolutional", "cnn"
    ]
}

print(f"Paper: 1202.2745 (Multi-Column CNN)\n")
for domain, triggers in checks.items():
    if domain == "rl_short_wordboundary":
        matches = [t for t in triggers if re.search(rf'\b{t}\b', text)]
    else:
        matches = [t for t in triggers if t in text]
    if matches:
        print(f"  [{domain}] TRIGGERED: {matches}")
    else:
        print(f"  [{domain}] clean")