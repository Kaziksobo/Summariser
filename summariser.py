from csv import writer
import gradio
from time import time
from validators import url
from nltk.tokenize import sent_tokenize
import os
from summariser_functions import article_scraper, csv_checker, wiki_scraper, summary_generator, summary_score, log, report

def summariser(text_inputted):
    # returns - error, text_title, summary
    try:
        file = open('log.csv', 'r')
    except FileNotFoundError:
        file = open('log.csv', 'w')
        csv_writer = writer(file)
        csv_writer.writerow(['text title', 'summary', 'score', 'generation time', 'datetime'])
    file.close()
    start = time()
    text, text_title = article_scraper(text_inputted) if url(text_inputted) else wiki_scraper(text_inputted)
    if text_title == 'Error':
        return 'That input could not be summarised', None, None
    print(f'summarising {text_title}')
    summary = None
    summary, score = csv_checker(text_title)
    if not summary:
        summary = summary_generator(text)
        # fixing summary format
        summary = sent_tokenize(summary)
        if summary[-1][-1] != '.':
            summary.pop()
        summary = ' '.join(summary)
        summary = summary.replace('; ', '')

        score = round(summary_score(summary, text), 2)

    print(score)

    end = time()
    timer = round(end - start, 2)
    print(f'summary generated in {timer}s')
    log(text_title, summary, score, timer)
    
    # calculates reduction in size from the original text to the summary
    reduction = round(((len(text) - len(summary)) / len(text)) * 100, 2)

    error = ''

    score = round(summary_score(summary, text), 2)

    # if the rouge score is under 0.4, it will return the summary, along with an error message stating the low rouge score
    # (this number is arbitrarily picked, might be a good choice, might not be, who knows???)
    if score < 0.4:
        return (f'The summary generated may not be accurate (Rouge-2 F1-Score score of {round(score, 4)})'), text_title, summary,
    else:
        return None, text_title, summary

description = 'Uses a BART transformer to generate an accurate summary of an article or topic'

explanation = """<p style="text-align: center;">
    Explanation: <br>
    If a url is inputted, the program will use the <a href="https://pypi.org/project/newspaper3k/">newspaper3k library</a> to scrape the text of the article.<br>
    Otherwise, the program will then use the <a href="https://github.com/martin-majlis/Wikipedia-API/">wikipedia api</a> to find the corresponding article for the text inputted. It will then get the text of the article.<br>
    Then a <a href="https://arxiv.org/pdf/1910.13461.pdf">BART transformer</a> (using the <a href="https://huggingface.co/facebook/bart-large-cnn">bart-large-cnn</a> model) is used to generate a summary of the given text.<br>
    This summary is then checked for accuracy using the F1 score from the <a href="https://aclanthology.org/W04-1013.pdf">Rouge package</a>.
    The reference summary used is the first three sentences of the text.<br><br>
    <a href="https://github.com/Kaziksobo/Summarizer">Github</a>
</p>"""

ui = gradio.Interface(
    summariser, 
    [
        gradio.inputs.Textbox(
            placeholder='Enter topic/link to article', 
            label = 'Text'
        )
    ],
    [
        gradio.outputs.Textbox(label = 'Error'),
        gradio.outputs.Textbox(label = 'Text summarised'),
        gradio.outputs.Textbox(label = 'Summary')
    ], 
    layout='horizontal',
    title='Summariser',
    description=description,
    article=explanation,
    theme='dark-grass',
    flagging_options=['Incorrect accuracy score', 'Incorrect text summarised', 'other']
)

ui.launch(share=True)
