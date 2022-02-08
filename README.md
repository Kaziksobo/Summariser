# summariser

## Explanation

If a url is inputted, the program will use the [newspaper3k module](https://pypi.org/project/newspaper3k/) to scrape the text of the article.\
Otherwise, the program will then use the [wikipedia api](https://github.com/martin-majlis/Wikipedia-API/) to find the corresponding article for the text inputted, and then get the text of the article.\
Then a [BART transformer](https://arxiv.org/pdf/1910.13461.pdf) (pre-trained on the [bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) model) is used to generate a summary of the given text
This summary is then checked for accuracy using the Rouge-2 F1 score from the [Rouge package](https://aclanthology.org/W04-1013.pdf).\
The reference summary used is the first three sentences of the article.
