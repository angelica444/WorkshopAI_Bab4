"""Example 4.23: Run sentiment analysis with a Hugging Face pipeline."""

from transformers import pipeline


if __name__ == "__main__":
    classifier = pipeline("sentiment-analysis")
    print(classifier("This is a good movie."))