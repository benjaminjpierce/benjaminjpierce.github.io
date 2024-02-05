from scriptscript import ScriptScript
import pprint as pp


def main():

    # create ScriptScript instance
    ss = ScriptScript()


    stopwords = ['to', 'you', 'the', 'a', 'i']

    # load text files w/ custom parser
    ss.load_text('bible books/test.txt', 'Althea', stop_words=stopwords)


    # print collected data
    pp.pprint(ss.data)

    # visualizations
    ss.compare_num_words()
    ss.wordcount_sankey(k=5)
    ss.plot_sentiment_over_time()
    ss.generate_word_cloud()


if __name__ == '__main__':
    main()
