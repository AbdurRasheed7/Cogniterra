import sys
sys.path.insert(0, 'D:/ai_repro_engine')
from agents.parser_agent import parse_paper

text = parse_paper('1706.03762').lower()

cv_triggers = [
    "deep residual learning",
    "image classification on imagenet",
    "top-1 accuracy", "top-5 accuracy",
    "vgg very deep", "very deep convolutional networks for large-scale",
    "depthwise separable convolution", "mobilenet", "efficientnet",
    "densely connected convolutional", "densenet",
    "feature pyramid network", "object detection with deep"
]

matches = [t for t in cv_triggers if t in text]
print("CV triggers found in Transformer paper:")
print(matches)