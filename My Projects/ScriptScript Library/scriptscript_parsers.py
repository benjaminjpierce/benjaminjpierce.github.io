from collections import Counter
from textblob import TextBlob
import string
import numpy as np

def bible_parser(text, stop_words=None):
    """
        Parser that processes bible book files
    """

    # remove punctuation & convert to lowercase
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator).lower()

    # split text into individual words
    words = text.split()

    # exclude words w/ numerical characters (to remove book/chapter/verse from words)
    words = [word for word in words if not any(char.isdigit() for char in word)]

    # filter out specified stop words
    if stop_words:
        words = [word for word in words if word not in stop_words]

    # conduct sentiment analysis, divide text into percentiles for analysis over course of text
    percentiles = np.percentile(np.arange(len(words)), np.linspace(0, 100, 10))
    percentiles = percentiles.astype(int)
    percentile_texts = [words[percentiles[i]:percentiles[i + 1]] for i in range(len(percentiles) - 1)]

    # calculate sentiment for each percentile
    sentiment_over_percentiles = [
        {'polarity': TextBlob(" ".join(percentile)).sentiment.polarity,
         'subjectivity': TextBlob(" ".join(percentile)).sentiment.subjectivity}
        for percentile in percentile_texts
    ]


    # result data dictionary
    results = {
        'wordcount': Counter(words),
        'sentiment_over_percentiles': sentiment_over_percentiles
    }

    return results