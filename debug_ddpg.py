import sys
sys.path.insert(0, 'D:/ai_repro_engine')
from agents.parser_agent import parse_paper

result = parse_paper('1509.02971')
text = result.lower() if isinstance(result, str) else result['filtered_text'].lower()

cv_triggers = [
    "deep residual learning",
    "image classification on imagenet",
    "top-1 accuracy", "top-5 accuracy",
    "imagenet classification",
    "vgg very deep", "very deep convolutional networks for large-scale",
    "depthwise separable convolution", "mobilenet", "efficientnet",
    "densely connected convolutional", "densenet",
    "feature pyramid network", "object detection with deep"
]

matches = [t for t in cv_triggers if t in text]
print("CV triggers found in DDPG paper:")
print(matches)