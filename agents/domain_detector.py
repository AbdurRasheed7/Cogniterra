import re

DOMAIN_KEYWORDS = {
    "nlp": [
        "natural language", "text classification", "sentiment analysis", "BERT",
        "transformer", "attention mechanism", "multi-head attention", "self-attention",
        "word embedding", "tokenization", "language model", "sequence to sequence",
        "named entity", "text dataset", "corpus", "vocabulary", "NLP", "sentence",
        "word2vec", "GloVe", "machine translation", "BLEU", "encoder decoder",
        "attention is all you need", "positional encoding", "feed-forward network",
        "query key value", "masked attention", "cross attention", "text generation",
        "question answering", "summarization", "language understanding",
        "pre-training", "fine-tuning", "GPT", "T5", "RoBERTa", "XLNet",
        "translation", "seq2seq", "autoregressive", "token", "subword"
    ],
    "reinforcement_learning": [
        "reinforcement learning", "reward function", "policy gradient", "Q-learning",
        "agent", "environment", "action space", "state space", "OpenAI Gym",
        "Markov decision", "DQN", "PPO", "actor-critic", "episode reward",
        "value function", "temporal difference", "exploration exploitation",
        "replay buffer", "epsilon greedy", "discount factor", "DDPG", "SAC",
        "A3C", "A2C", "proximal policy", "trust region", "model-based RL",
        "monte carlo", "trajectory", "rollout", "gymnasium"
    ],
    "graph": [
        "graph neural network", "point cloud", "graph matching",
        "message passing", "node classification", "graph convolution",
        "edge features", "adjacency matrix", "graph embedding",
        "3d object", "GNN", "GCN", "graph attention network",
        "knowledge graph", "graph network", "graph convolutional",
        "spectral graph", "spatial graph", "graph pooling",
        "node embedding", "link prediction", "graph classification",
        "GraphSAGE", "GAT", "graph isomorphism", "molecular graph",
        "social network", "heterogeneous graph", "dynamic graph"
    ],
    "generative": [
        "generative adversarial", "GAN", "variational autoencoder", "VAE",
        "image generation", "latent space", "generator", "discriminator",
        "diffusion model", "denoising diffusion", "score matching",
        "image synthesis", "style transfer", "super resolution",
        "inpainting", "text to image", "flow model", "normalizing flow",
        "DDPM", "stable diffusion", "noise prediction", "U-Net generation"
    ],
    "recommendation": [
        "recommendation system", "collaborative filtering", "user-item",
        "matrix factorization", "MovieLens", "rating prediction", "user preference",
        "item embedding", "implicit feedback", "explicit feedback", "recommender",
        "user embedding", "item rating", "user behavior", "purchase history",
        "click through", "CTR", "recall@k", "precision@k", "NDCG",
        "cold start", "sparse matrix", "interaction matrix",
        "personalized recommendation", "top-k recommendation", "e-commerce"
    ],
    "algorithm": [
        "sorting algorithm", "graph algorithm", "dynamic programming",
        "binary search", "tree traversal", "shortest path", "complexity analysis",
        "data structure", "hash table", "linked list", "recursion",
        "time complexity", "space complexity", "big O", "greedy algorithm",
        "divide and conquer", "backtracking", "memoization"
    ],
    "image_classification": [
        "image classification", "convolutional", "CNN", "conv2d", "image recognition",
        "MNIST", "CIFAR", "ImageNet", "pixel", "visual", "object detection",
        "feature maps", "pooling", "batch normalization", "image dataset",
        "convolutional neural network", "residual block", "resnet", "vgg", "alexnet",
        "depthwise", "separable convolution", "skip connection", "dense connection",
        "squeeze excitation", "object recognition", "image segmentation",
        "semantic segmentation", "instance segmentation", "bounding box",
        "anchor", "feature pyramid", "visual recognition"
    ]
}

DOMAIN_DATASETS = {
    "image_classification": "MNIST (28x28 grayscale, 10 classes)",
    "nlp":                  "20NewsGroups (text classification)",
    "recommendation":       "MovieLens 100K (user-item ratings)",
    "reinforcement_learning": "CartPole-v1 (OpenAI Gym)",
    "algorithm":            "Custom test cases (pure Python)",
    "generative":           "MNIST (28x28 grayscale, for generation)",
    "graph":                "Synthetic graph data (node classification)",
    "unknown":              "Unable to determine — manual selection needed"
}


def detect_domain(paper_text):
    text_lower = paper_text.lower()

    # ── Priority order: most specific first, CNN dead last ────────────────────

    # -2. Graph PRE-CHECK — must be BEFORE CV pre-check because GCN paper
    #     mentions "semi-supervised classification" which could trigger CV
    if any(term in text_lower for term in [
        "graph convolutional network", "graph neural network",
        "node classification", "message passing", "adjacency matrix",
        "graph convolution", "spectral graph", "graph laplacian",
        "semi-supervised classification with graph",
        "gcn", "gnn", "graph attention", "graphsage",
        "graph isomorphism", "molecular graph",
        "cora", "citeseer", "pubmed dataset", "citation network"
    ]):
        return _result("graph", ["graph/GNN domain detected"])

    # -1. CV PRE-CHECK — only fire on terms that are 100% unambiguous CV
    #     Must NOT appear in RL papers (DDPG mentions "deep" and "network" everywhere)
    if any(term in text_lower for term in [
        "image classification on imagenet",
        "top-1 accuracy", "top-5 accuracy",
        "vgg very deep", "very deep convolutional networks for large-scale",
        "mobilenet", "efficientnet",
        "densely connected convolutional", "densenet",
        "feature pyramid network", "object detection with deep"
    ]):
        return _result("image_classification", ["CV pre-check detected"])

    # 0. Graph PRE-CHECK (fallback) — before NLP because GCN papers mention "text classification"
    #    in experiments section when benchmarking on text datasets
    if any(term in text_lower for term in [
        "graph convolutional network", "graph neural network",
        "node classification", "message passing", "adjacency matrix",
        "graph convolution", "spectral graph", "graph laplacian",
        "semi-supervised classification with graph",
        "gcn", "gnn", "graph attention", "graphsage",
        "graph isomorphism", "molecular graph"
    ]):
        return _result("graph", ["graph/GNN domain detected"])

    # 0.5. Recommendation PRE-CHECK — before NLP because NCF/rec papers
    #      mention "natural language processing" in related work
    if any(term in text_lower for term in [
        "recommendation system", "recommender system",
        "collaborative filtering", "matrix factorization",
        "user-item interaction", "rating prediction",
        "movielens", "implicit feedback", "explicit feedback",
        "personalized recommendation", "top-k recommendation",
        "click through rate", "cold start problem"
    ]):
        return _result("recommendation", ["recommendation system detected"])

    # 1. NLP — after recommendation pre-check
    #    Removed "text classification", "bert", "transformer" alone —
    #    they appear in GCN/graph papers as benchmark comparisons
    if any(term in text_lower for term in [
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
    ]):
        return _result("nlp", ["NLP task detected"])

    # 2. Reinforcement Learning — before CV because RL papers use CNNs (e.g. Atari DQN)
    # Note: short terms like "ppo", "sac", "dqn" must be checked as whole words
    #       to avoid matching substrings like "approach", "saccade", "adjacent"
    import re as _re
    rl_long_terms = [
        "reinforcement learning", "reward function", "policy gradient",
        "q-learning", "markov decision", "actor-critic",
        "openai gym", "action space", "state space", "episode reward",
        "value function", "temporal difference", "replay buffer",
        "epsilon greedy", "discount factor",
        "proximal policy", "trust region", "gymnasium", "rollout"
    ]
    rl_short_terms = ["dqn", "ppo", "ddpg", "a3c", "a2c"]  # removed "sac" — too ambiguous as standalone word
    rl_match = any(term in text_lower for term in rl_long_terms) or                any(_re.search(rf'\b{term}\b', text_lower) for term in rl_short_terms)
    if rl_match:
        return _result("reinforcement_learning", ["reinforcement learning detected"])

    # 3. Graph — before CV because "graph convolutional" contains "convolutional"
    if any(term in text_lower for term in [
        "graph neural network", "graph convolutional", "graph convolution",
        "node classification", "message passing", "adjacency matrix",
        "graph embedding", "gnn", "gcn", "graph attention",
        "knowledge graph", "graph network", "link prediction",
        "graph classification", "graphsage", "graph isomorphism",
        "molecular graph", "point cloud"
    ]):
        return _result("graph", ["graph/GNN domain detected"])

    # 4. Generative — before CV because GANs/VAEs heavily use CNNs
    #    Short terms like "gan", "vae" checked as whole words to avoid substrings
    import re as _re2
    gen_long_terms = [
        "generative adversarial", "variational autoencoder",
        "image generation", "diffusion model",
        "denoising diffusion", "score matching", "image synthesis",
        "style transfer", "super resolution", "normalizing flow",
        "ddpm", "stable diffusion", "noise prediction", "latent diffusion",
        "generator network", "discriminator network"
    ]
    gen_short_terms = ["vae"]  # removed "gan" — appears as standalone word in non-generative papers
    gen_match = any(term in text_lower for term in gen_long_terms) or                 any(_re2.search(rf'\b{term}\b', text_lower) for term in gen_short_terms)
    if gen_match:
        return _result("generative", ["generative model detected"])

    # 5. Recommendation
    if any(term in text_lower for term in [
        "recommendation system", "recommender system",
        "collaborative filtering", "matrix factorization",
        "user-item interaction", "rating prediction",
        "personalized recommendation", "top-k recommendation",
        "movielens", "implicit feedback", "explicit feedback",
        "click through rate", "cold start problem"
    ]):
        return _result("recommendation", ["recommendation system detected"])

    # 6. Algorithm (pure CS — no ML frameworks)
    if any(term in text_lower for term in [
        "sorting algorithm", "dynamic programming", "binary search",
        "tree traversal", "shortest path", "time complexity",
        "space complexity", "big o notation", "data structure",
        "hash table", "divide and conquer", "backtracking"
    ]):
        return _result("algorithm", ["algorithm detected"])

    # 7. Image Classification — LAST resort
    #    Only fires if none of the above matched
    if any(term in text_lower for term in [
        "image classification", "object detection", "image recognition",
        "convolutional neural network", "resnet", "vgg", "alexnet",
        "mnist", "cifar", "imagenet", "feature maps", "pooling layer",
        "conv2d", "residual block", "skip connection", "dense block",
        "semantic segmentation", "instance segmentation", "bounding box",
        "feature pyramid", "object recognition", "visual recognition",
        "convolutional", "cnn"
    ]):
        return _result("image_classification", ["CV/image classification detected"])

    # ── Fallback: keyword frequency scoring ───────────────────────────────────
    scores = {}
    matched_keywords = {}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        score   = 0
        matches = []
        for keyword in keywords:
            count = text_lower.count(keyword.lower())
            if count > 0:
                score += count
                matches.append(keyword)
        scores[domain]           = score
        matched_keywords[domain] = matches

    best_domain = max(scores, key=scores.get)
    best_score  = scores[best_domain]

    if best_score < 2:
        best_domain = "unknown"

    sorted_scores = sorted(scores.values(), reverse=True)
    second_score  = sorted_scores[1] if len(sorted_scores) > 1 else 0

    if best_score == 0:
        confidence = 0
    elif second_score == 0:
        confidence = 100
    else:
        confidence = min(100, int((best_score / (best_score + second_score)) * 100))

    if confidence < 60:
        best_domain = "unknown"

    return {
        "domain":           best_domain,
        "confidence":       confidence,
        "score":            best_score,
        "matched_keywords": matched_keywords.get(best_domain, [])[:5],
        "dataset":          DOMAIN_DATASETS.get(best_domain, "Unknown"),
        "all_scores":       scores
    }


def _result(domain, keywords):
    """Helper — returns a clean priority-override result."""
    return {
        "domain":           domain,
        "confidence":       95,
        "score":            99,
        "matched_keywords": keywords,
        "dataset":          DOMAIN_DATASETS[domain],
        "all_scores":       {}
    }


def format_domain_report(detection):
    domain     = detection['domain']
    confidence = detection['confidence']
    keywords   = detection['matched_keywords'][:5]

    report = "\n--- DOMAIN DETECTION ---\n"

    if domain == "unknown":
        report += "⚠️  Domain: UNKNOWN — Could not determine paper domain\n"
        report += "💡 Defaulting to image classification (project focus)\n"
    else:
        emoji_map = {
            "image_classification":  "🖼️",
            "nlp":                   "📝",
            "recommendation":        "🎯",
            "reinforcement_learning":"🤖",
            "algorithm":             "⚙️",
            "graph":                 "🕸️",
            "generative":            "🎨"
        }
        emoji  = emoji_map.get(domain, "📄")
        report += f"{emoji}  Domain: {domain.replace('_', ' ').upper()}\n"
        report += f"📊 Confidence: {confidence}%\n"
        report += f"🔑 Key terms found: {', '.join(keywords)}\n"
        report += f"💾 Will use dataset: {detection['dataset']}\n"

    return report


def get_code_domain(detection):
    domain = detection['domain']

    mapping = {
        "image_classification":  "ml",
        "generative":            "ml",          # no generative template yet — fallback to ml
        "nlp":                   "nlp",
        "recommendation":        "recommendation",
        "reinforcement_learning":"rl",
        "algorithm":             "algorithm",
        "graph":                 "graph",
        "unknown":               "ml"
    }

    return mapping.get(domain, "ml")