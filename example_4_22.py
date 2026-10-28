"""Example 4.22: Tokenize, tag, and extract entities with NLTK."""

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


def main():
    ensure_nltk_data()
    sentence = (
        "Artificial intelligence systems learn patterns from data. "
        "Recurrent neural networks can process sequences of words."
    )
    tokens = nltk.word_tokenize(sentence)
    print("Tokens:", tokens)
    tagged = nltk.pos_tag(tokens)
    print("Part-of-speech tags:", tagged)
    print("Named entities:", nltk.ne_chunk(tagged))


if __name__ == "__main__":
    main()