"""Exercise 4.18: Run NLTK analysis on a different text passage."""

import nltk


def ensure_nltk_data():
    resources = (
        "punkt",
        "punkt_tab",
        "averaged_perceptron_tagger_eng",
        "maxent_ne_chunker_tab",
        "words",
    )
    for resource in resources:
        nltk.download(resource, quiet=True)


if __name__ == "__main__":
    ensure_nltk_data()
    sentence = (
        "Ada Lovelace wrote about Charles Babbage's analytical engine. "
        "Modern computers use programs to solve many kinds of problems."
    )
    tokens = nltk.word_tokenize(sentence)
    print("Tokens:", tokens)
    tagged = nltk.pos_tag(tokens)
    print("Part-of-speech tags:", tagged)
    print("Named entities:", nltk.ne_chunk(tagged))