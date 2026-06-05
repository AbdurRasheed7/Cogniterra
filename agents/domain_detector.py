import re
import json
import os

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
        "replay buffer", "epsilon greedy", "discount factor", "DDPG",
        "A3C", "A2C", "proximal policy", "trust region", "model-based RL",
        "monte carlo", "trajectory", "rollout", "gymnasium"
    ],
    "graph": [
        "graph neural network", "point cloud", "graph matching",
        "message passing", "node classification", "graph convolution",
        "edge features", "adjacency matrix", "graph embedding",
        "GNN", "GCN", "graph attention network",
        "graph network", "graph convolutional",
        "spectral graph", "spatial graph", "graph pooling",
        "node embedding", "link prediction", "graph classification",
        "GraphSAGE", "GAT", "graph isomorphism", "molecular graph",
        "social network", "heterogeneous graph", "dynamic graph",
        "cora", "citeseer", "pubmed"
    ],
    "generative": [
        "generative adversarial", "variational autoencoder",
        "image generation", "latent space", "generator", "discriminator",
        "diffusion model", "denoising diffusion", "score matching",
        "image synthesis", "style transfer", "super resolution",
        "inpainting", "text to image", "flow model", "normalizing flow",
        "DDPM", "stable diffusion", "noise prediction"
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

# ---------------------------------------------------------------------------
# Task string -> domain mapping used by structure cache override
# ---------------------------------------------------------------------------
# Maps substrings of the "task" field in cached structure JSON to domains.
# This is the source of truth when keyword detection gets confused by
# papers that mention terms from other domains in their related work.
# e.g. BatchNorm paper mentions "language model" but task = "image classification"

_TASK_TO_DOMAIN = {
    # CV / image tasks
    "image classification":   "image_classification",
    "object detection":       "image_classification",
    "image segmentation":     "image_classification",
    "semantic segmentation":  "image_classification",
    "instance segmentation":  "image_classification",
    "visual recognition":     "image_classification",
    "image recognition":      "image_classification",
    # Training technique papers — applied to images
    "batch normalization":    "image_classification",
    "knowledge distillation": "image_classification",
    "model compression":      "image_classification",
    "network pruning":        "image_classification",
    "quantization":           "image_classification",
    "mixed precision":        "image_classification",
    # NLP tasks
    "machine translation":    "nlp",
    "text classification":    "nlp",
    "language modeling":      "nlp",
    "question answering":     "nlp",
    "named entity":           "nlp",
    "sentiment":              "nlp",
    # RL tasks
    "reinforcement learning": "reinforcement_learning",
    "control":                "reinforcement_learning",
    "policy":                 "reinforcement_learning",
    "reward":                 "reinforcement_learning",
    # Graph tasks
    "node classification":    "graph",
    "graph classification":   "graph",
    "link prediction":        "graph",
    # Recommendation tasks
    "rating prediction":      "recommendation",
    "collaborative filtering":"recommendation",
    "recommendation":         "recommendation",
}


def _check_structure_cache(paper_id: str):
    """
    Look up the cached paper structure JSON written by coder_agent.py.

    If found and the task field maps to a known domain, return a detection
    result dict — this overrides keyword-based detection entirely.

    Returns None if no cache exists or task is unrecognised.

    This is the robust fix for misclassification: instead of patching
    keyword priority lists (fragile), we use the LLM's own structured
    extraction of what the paper is actually about.
    """
    if not paper_id:
        return None

    cache_path = os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "tests", "structure",
            f"{paper_id}_structure.json",
        )
    )

    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            structure = json.load(f)

        task = structure.get("task", "").lower().strip()
        if not task:
            return None

        for task_key, domain in _TASK_TO_DOMAIN.items():
            if task_key in task:
                print(f"   🗂️  Structure cache override: task={task!r} → domain={domain}")
                return _result(domain, [f"structure cache: task={task!r}"])

        # Task found but not in our map — log and fall through to keyword detection
        print(f"   ⚠️  Structure cache: unrecognised task={task!r} — using keyword detection")
        return None

    except (json.JSONDecodeError, OSError):
        return None


def detect_domain(paper_text, paper_id=None):
    """
    Detect the domain of a paper from its text.

    Args:
        paper_text: filtered paper text from parse_paper()
        paper_id:   arXiv ID (optional) — used to check structure cache first.
                    Pass this whenever available to get more accurate detection.

    Returns:
        Detection result dict with keys: domain, confidence, score,
        matched_keywords, dataset, all_scores.
    """
    # ── Structure cache override (most reliable signal) ───────────────────────
    # Check this BEFORE any keyword logic. The structure cache contains the
    # LLM's own extraction of what task the paper solves, which is more
    # accurate than keyword matching for papers that mention other domains
    # in their related work (e.g. BatchNorm mentioning "language model").
    if paper_id:
        cached = _check_structure_cache(paper_id)
        if cached is not None:
            return cached

    text_lower = paper_text.lower()

    # ── Priority order: most specific first ──────────────────────────────────

    # 1. NLP — Transformer/BERT papers first (mention "attention" everywhere)
    if any(term in text_lower for term in [
        "attention is all you need", "multi-head attention", "self-attention",
        "machine translation", "language model", "seq2seq", "sequence to sequence",
        "natural language processing", "named entity recognition",
        "sentiment analysis", "question answering",
        "positional encoding", "neural machine translation", "bleu score",
        "tokenization", "subword", "masked language model",
        "text generation task", "pre-training language",
        "language understanding", "bert pre-training",
        "encoder decoder attention", "transformer model for"
    ]):
        return _result("nlp", ["NLP task detected"])

    # 2. Graph — before CV because "graph convolutional" contains "convolutional"
    #    and GCN/GAT papers mention "classification" in benchmarks
    if any(term in text_lower for term in [
        "graph neural network", "graph convolutional", "graph convolution",
        "node classification", "message passing", "adjacency matrix",
        "graph embedding", "gcn", "gnn", "graph attention",
        "graphsage", "graph isomorphism", "molecular graph",
        "cora dataset", "citeseer", "pubmed dataset", "citation network"
    ]) or re.search(r'\bgat\b', text_lower):
        return _result("graph", ["graph/GNN domain detected"])

    # 3. Recommendation — before NLP because NCF mentions "natural language"
    if any(term in text_lower for term in [
        "recommendation system", "recommender system",
        "collaborative filtering", "matrix factorization",
        "user-item interaction", "rating prediction",
        "movielens", "implicit feedback", "explicit feedback",
        "personalized recommendation", "top-k recommendation",
        "click through rate", "cold start problem"
    ]):
        return _result("recommendation", ["recommendation system detected"])

    # 4. Reinforcement Learning — before CV because RL papers use CNNs
    #    Short terms (dqn, ppo, a3c) checked with word boundaries
    rl_long = [
        "reinforcement learning", "reward function", "policy gradient",
        "q-learning", "markov decision", "actor-critic",
        "openai gym", "action space", "state space", "episode reward",
        "value function", "temporal difference", "replay buffer",
        "epsilon greedy", "discount factor", "ddpg",
        "proximal policy", "trust region", "gymnasium", "rollout"
    ]
    rl_short = ["dqn", "ppo", "a3c", "a2c"]  # sac removed — appears in non-RL papers
    if any(t in text_lower for t in rl_long) or \
       any(re.search(rf'\b{t}\b', text_lower) for t in rl_short):
        return _result("reinforcement_learning", ["reinforcement learning detected"])

    # 5. Generative — before CV because GANs use CNNs
    #    gan/vae checked with word boundaries — appear as substrings elsewhere
    gen_long = [
        "generative adversarial", "variational autoencoder",
        "image generation", "diffusion model", "denoising diffusion",
        "score matching", "image synthesis", "style transfer",
        "super resolution", "normalizing flow", "ddpm",
        "stable diffusion", "noise prediction", "latent diffusion",
        "generator network", "discriminator network"
    ]
    gen_short = ["vae"]  # gan removed — appears as standalone word in non-GAN papers
    if any(t in text_lower for t in gen_long) or \
       any(re.search(rf'\b{t}\b', text_lower) for t in gen_short):
        return _result("generative", ["generative model detected"])

    # 6. CV pre-checks — before algorithm because CV papers mention "learning algorithm"
    #    Use specific phrases that only appear in CV papers
    if any(term in text_lower for term in [
        "deep residual learning", "image classification on imagenet",
        "top-1 accuracy", "top-5 accuracy",
        "vgg very deep", "very deep convolutional networks for large-scale",
        "mobilenet", "efficientnet", "densely connected convolutional",
        "feature pyramid network", "object detection with deep"
    ]):
        return _result("image_classification", ["CV pre-check detected"])

    # 7. Algorithm — pure CS papers (no ML, no CV)
    if any(term in text_lower for term in [
        "sorting algorithm", "dynamic programming", "binary search",
        "tree traversal", "shortest path", "time complexity",
        "space complexity", "big o notation", "data structure",
        "hash table", "divide and conquer", "backtracking"
    ]):
        return _result("algorithm", ["algorithm detected"])

    # 8. Image Classification — last resort
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

    # ── Fallback: keyword frequency scoring ──────────────────────────────────
    scores = {}
    matched_keywords = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = 0
        matches = []
        for keyword in keywords:
            count = text_lower.count(keyword.lower())
            if count > 0:
                score += count
                matches.append(keyword)
        scores[domain] = score
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
        "generative":            "ml",
        "nlp":                   "nlp",
        "recommendation":        "recommendation",
        "reinforcement_learning":"rl",
        "algorithm":             "algorithm",
        "graph":                 "graph",
        "unknown":               "ml"
    }
    return mapping.get(domain, "ml")