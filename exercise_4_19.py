"""Exercise 4.19: Compare sentiment predictions for positive and negative text."""

from transformers import pipeline


if __name__ == "__main__":
    classifier = pipeline("sentiment-analysis")
    sentences = [
        "The story was moving, and the actors gave wonderful performances.",
        "The plot was confusing and the film was far too long.",
        "The visuals were impressive, but the ending disappointed me.",
    ]
    for sentence, result in zip(sentences, classifier(sentences)):
        print(f"Text: {sentence}")
        print(f"Prediction: {result['label']} ({result['score']:.3f})\n")