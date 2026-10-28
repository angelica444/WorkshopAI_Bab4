"""Exercise 4.20: Ask different questions about a new text passage."""

from transformers import pipeline


if __name__ == "__main__":
    question_answerer = pipeline("question-answering")
    context = (
        "Ada Lovelace worked with Charles Babbage on ideas for the Analytical Engine. "
        "Her notes, published in 1843, included a method for calculating Bernoulli numbers."
    )
    questions = [
        "Who did Ada Lovelace work with?",
        "When were her notes published?",
        "What calculation method did the notes describe?",
    ]
    for question in questions:
        result = question_answerer(question=question, context=context)
        print(f"Question: {question}")
        print(f"Answer: {result['answer']} (score={result['score']:.3f})\n")