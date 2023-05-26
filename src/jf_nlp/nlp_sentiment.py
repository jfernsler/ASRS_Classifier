from transformers import pipeline

from .nlp_dataloaders import NLPDataLoader
from .nlp_globals import *

def main():
    lyrics = NLPDataLoader(LYRICS_DATA)
    lyric_split = lyrics.data.iloc[0]['text'].split('\n')
    print(lyric_split)

if __name__ == '__main__':
    main()

# sentiment_analysis = pipeline("sentiment-analysis")
# foo = sentiment_analysis("I love you")
# print(foo)