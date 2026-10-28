"""Example 4.24: Answer a question from a short context paragraph."""

from transformers import pipeline


if __name__ == "__main__":
    question_answerer = pipeline("question-answering")
    result = question_answerer(
        question="What is the name of the company?",
        context="We created Biox Systems Ltd company back in the year of 2000.",
    )
    print(result)