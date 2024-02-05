from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import string
import pandas as pd
from sankeyLib import generate_and_show_sankey
from wordcloud import WordCloud
from matplotlib.gridspec import GridSpec
import numpy as np
from textblob import TextBlob


class ScriptScript:

    def __init__(self):
        # string  --> {filename/label --> statistics}
        # "wordcounts" --> {"A": wc_A, "B": wc_B, ....}
        self.data = defaultdict(dict)


    def _save_results(self, label, results):
        """
            Save results of processing a text file to internal state variable

            Args:
            - label (str): label/filename associated with  text
            - results (dict): dictionary containing results of analysis

            Returns:
            - None
        """

        for k, v in results.items():
            self.data[k][label] = v


    @staticmethod
    def _default_parser(text, stop_words=None):
        """
            Default parser that processes generic text files
            Extracts word counts, word length, text length, and text sentiment
        """

        # remove punctuation & convert to lowercase
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator).lower()

        # split text into individual words
        words = text.split()

        # filter out any specified stop words
        if stop_words:
            words = [word for word in words if word not in stop_words]

        # calculate default metrics
        num_words = len(words)
        avg_word_length = sum(len(word) for word in words) / num_words if num_words > 0 else 0

        # conduct sentiment analysis, divide text into percentiles for analysis over course of text
        percentiles = np.percentile(np.arange(len(words)), np.linspace(0, 100, 10))
        percentiles = percentiles.astype(int)
        percentile_texts = [words[percentiles[i]:percentiles[i + 1]] for i in range(len(percentiles) - 1)]

        # calculate sentiment for each percentile
        try:
            sentiment_over_percentiles = [
                {'polarity': TextBlob(" ".join(percentile)).sentiment.polarity,
                'subjectivity': TextBlob(" ".join(percentile)).sentiment.subjectivity}
                for percentile in percentile_texts
            ]

        except Exception as e:
            raise TextAnalysisError(f"Error in TextBlob sentiment analysis for {filename}: {str(e)}") from e

        # result data dictionary
        results = {
            'wordcount': Counter(words),
            'numwords': num_words,
            'avg_word_length': avg_word_length,
            'sentiment_over_percentiles': sentiment_over_percentiles
        }

        return results


    def load_text(self, filename, label=None, parser=None, stop_words=None):
        """ Registers a text document with the framework
        Extracts and stores data to be used in later
        visualizations. """

        try:
            if not filename.lower().endswith('.txt'):
                raise TextAnalysisError(f"Invalid file format for {filename}. Expected '.txt'")

            with open(filename, 'r', encoding='utf-8') as file:
                text = file.read()
                if parser is None:
                    results = ScriptScript._default_parser(text, stop_words)
                else:
                    results = parser(text, stop_words)

            if label is None:
                label = filename

            # store the results of processing one file
            # in the internal state (data)
            self._save_results(label, results)

        except Exception as e:
            raise TextAnalysisError(f"Error loading text from {filename}: {str(e)}")


    def compare_num_words(self):
        """ A DEMONSTRATION OF A CUSTOM VISUALIZATION
        A trivially simple barchart comparing number
        of words in each registered text file. """

        num_words = self.data['numwords']
        plt.figure()
        for label, nw in num_words.items():
            plt.bar(label, nw)
        plt.xlabel('Text Label')
        plt.ylabel('Number of Words')
        plt.title('Number of Words in Each Text File')
        plt.show()


    def wordcount_sankey(self, word_list=None, k=5):
        """
            Generate a Sankey diagram for word counts

            Args:
                word_list (List[str]): user-defined list of words
                k (int): number of most common words to consider

            Returns:
                None
        """

        # get k most common words from each book
        common_words = self._get_common_words(k)

        if word_list:
            # use user-defined words if provided
            selected_words = set(word_list)
        else:
            # union of common words from each book
            selected_words = set(word for common_set in common_words.values() for word in common_set)

        # prepare data for Sankey diagram
        sankey_data = {'source': [], 'target': [], 'value': []}

        for text, word_counts in self.data['wordcount'].items():
            for word in selected_words:
                sankey_data['source'].append(text)
                sankey_data['target'].append(word)
                sankey_data['value'].append(word_counts[word])

        # create df from Sankey data
        sankey_df = pd.DataFrame(sankey_data)

        # generate and show Sankey diagram
        generate_and_show_sankey(sankey_df, ['source', 'target'], vals='value', title='Wordcount Sankey')


    def _get_common_words(self, k):
        """
            Get k most common words from each text file

            Args:
                k (int): number of most common words to retrieve

            Returns:
                dict: dictionary mapping text file names to sets of common words
        """

        common_words = {}

        for metric, data in self.data.items():
            if metric == 'wordcount':
                for text, word_counts in data.items():
                    # get k most common words from each text's wordcount
                    common_words[text] = [word for word, _ in word_counts.most_common(k) if word.isalpha()]

        return common_words


    def generate_word_cloud(self):
        """
            Generate word clouds for each book based on word count data
        """

        # get books and word counts
        books = list(self.data['wordcount'].keys())
        wordcount_data = {book: self.data['wordcount'][book] for book in books}

        # create grid of subplots
        num_books = len(books)
        rows = int(num_books / 2) + (num_books % 2)
        cols = 2
        gs = GridSpec(rows, cols, width_ratios=[1, 1])

        # generate word clouds and plot
        for i, book in enumerate(books):
            wordcloud = WordCloud(width=400, height=200, background_color='white').generate_from_frequencies(
                wordcount_data[book])

            # plot the WordCloud
            plt.subplot(gs[i])
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.title(f"Word Cloud for {book}")

        plt.tight_layout()
        plt.show()


    def plot_sentiment_over_time(self):
        """
            Plot sentiment scores over time for each text file
        """

        # prepare data
        sentiment_data = {'Text': [], 'Percentile': [], 'Polarity': [], 'Subjectivity': []}

        for text, sentiment_percentiles in self.data['sentiment_over_percentiles'].items():
            for percentile, scores in enumerate(sentiment_percentiles, start=1):
                sentiment_data['Text'].append(text)
                sentiment_data['Percentile'].append(percentile)
                sentiment_data['Polarity'].append(scores['polarity'])
                sentiment_data['Subjectivity'].append(scores['subjectivity'])

        # create df for plotting
        sentiment_df = pd.DataFrame(sentiment_data)

        # plot polarity and subjectivity
        plt.figure(figsize=(12, 6))

        for text in sentiment_df['Text'].unique():
            data = sentiment_df[sentiment_df['Text'] == text]
            plt.plot(data['Percentile'], data['Polarity'], label=f'{text} - Polarity')

        for text in sentiment_df['Text'].unique():
            data = sentiment_df[sentiment_df['Text'] == text]
            plt.plot(data['Percentile'], data['Subjectivity'], label=f'{text} - Subjectivity', linestyle='dashed')

        plt.xlabel('Percentile of Text')
        plt.ylabel('Score')
        plt.title('Sentiment Across Each Book')
        plt.legend()
        plt.show()


class TextAnalysisError(Exception):
    pass
