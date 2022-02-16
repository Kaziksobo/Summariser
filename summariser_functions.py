from datetime import datetime
from newspaper import Article
from transformers import BartForConditionalGeneration, BartTokenizer
import wikipedia
import wikipediaapi
from rouge import Rouge
from nltk.tokenize import sent_tokenize
from csv import writer, reader
wiki_api = wikipediaapi.Wikipedia('en')

def article_scraper(text):
    # scrapes text of the article using newspaper3k
    try:
        article = Article(text)
        article.download()
        article.parse()
    except:
        return 'Error - that article cannot be summarised', 'Error'
    return article.text.replace('\n', ''), article.title

def wiki_scraper(text):
    # This code is so crap
    # I use both a python wrapper for the wikipedia api and an unmaintened wikipedia module
    # the module has a useful search function which I don't think the api wrapper has
    # but the module's function for getting the page from its title is buggy
    try:
        text_title = wikipedia.search(text)[0]
        wikisearch = wiki_api.page(text_title)
    except:
        return 'Error - that topic cannot be summarised', 'Error'
    return wikisearch.text, text_title

def csv_checker(text_title):
    # checks if the title of the text to be summarised is in the log file, if so gets the summary from there
    with open('log.csv', 'rt') as f:
        log = reader(f)
        return next(
            ((row[1], float(row[2])) for row in log if text_title == row[0]),
            (None, None),
        )


def summary_generator(text):
    # creates summary using BART transformer from huggingfaces
    checkpoint = 'facebook/bart-large-cnn'
    # create tokenizer using checkpoint model
    tokenizer = BartTokenizer.from_pretrained(checkpoint)
    #tokenizes inputs using created tokenizer, truncating to the model's max length, using pytorch tensors
    input_ids = tokenizer.batch_encode_plus(
        [text], 
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors='pt'
    )
    # creates model using BART for conditional generation, pretrained on the checkpoint model
    model = BartForConditionalGeneration.from_pretrained(checkpoint)
    # selects summary tokens, using beam search with 5 beams, stopping when all beam hypotheses reached the EOS token
    summary_ids = model.generate(
        input_ids['input_ids'],
        num_beams=5,
        early_stopping=True
    )
    # decodes the summary tokens, not counting any special tokens the model may have generated
    return tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True,
    )

def summary_score(summary, text):
    # calculate the accuracy of the summary using the rouge metric
    # uses the first three sentences of the article as the refernce summary (may be a good idea? not sure?)
    reference = ' '.join(sent_tokenize(text)[:3])
    scores = Rouge().get_scores(summary, reference)[0]
    # uses the Rouge-2 (2-gram) F1-score (combination of recall and precision)
    return scores['rouge-2']['f']

def log(text_title, summary, score, time):
    # logs the summary and its details to a log file
    row_contents = [text_title, summary, score, time, str(datetime.now())]
    with open('log.csv', 'a', encoding='utf-8', newline='') as f_object:
        write_object = writer(f_object)
        write_object.writerow(row_contents)
        f_object.close()

def report(text_title, summary, score, flag):
    # logs the summary and some details (including the report reason) to a seperate log file
    row_contents = [text_title, summary, score, flag]
    with open('flagged.csv', 'a', encoding='utf-8', newline='') as f_object:
        write_object = writer(f_object)
        write_object.writerow(row_contents)
        f_object.close()