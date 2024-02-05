from scriptscript import ScriptScript
import pprint as pp
from scriptscript_parsers import bible_parser

def main():

    # create ScriptScript instance
    ss = ScriptScript()

    # initialize stop words to exclude
    stop_words = ['the', 'me', 'is', 'them', 'then', 'my', 'which', 'when', 'from', 'where', 'am', 'as', 'because',
                  'on', 'into', 'there', 'are', 'these','and', 'that', 'of', 'he', 'i', 'him', 'unto', 'not', 'in',
                  'to', 'they', 'but', 'was', 'his', 'for', 'a', 'you', 'have', 'it', 'this', 'be', 'with', 'if',
                  'had', 'will', 'therefore', 'we', 'should', 'were', 'went', 'also', 'do', 'at', 'what', 'did']

    # load text files w/ custom parser
    ss.load_text('bible books/Matthew.txt', 'Matthew', parser=bible_parser, stop_words=stop_words)
    ss.load_text('bible books/Mark.txt', 'Mark', parser=bible_parser, stop_words=stop_words)
    ss.load_text('bible books/Luke.txt', 'Luke', parser=bible_parser, stop_words=stop_words)
    ss.load_text('bible books/John.txt', 'John', parser=bible_parser, stop_words=stop_words)

    # print collected data
    pp.pprint(ss.data)

    # visualizations
    ss.wordcount_sankey()
    ss.plot_sentiment_over_time()
    ss.generate_word_cloud()


if __name__ == '__main__':
    main()
