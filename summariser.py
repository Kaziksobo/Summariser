from transformers import BartForConditionalGeneration, BartTokenizer
from newspaper import Article
from validators import url
from nltk.tokenize import sent_tokenize
from rouge import Rouge
import gradio
import wikipedia as wiki_lib
import wikipediaapi
wiki_api = wikipediaapi.Wikipedia('en')

def wiki_scraper(text):
    # gets text content of corresponding wikipedia page
    wikisearch = wiki_api.page(text)
    if wikisearch.exists():
        text_title = wiki_lib.page(text).title
    else:
        # if no wiki page for input text, use wiki lib's search function and take first result
        try:
            text = wiki_lib.search(text)[0]
            text_title = text
        except:
            return '', ''
        wikisearch = wiki_api.page(text)
    # returns - text of wiki page, title of wiki page
    return wikisearch.text, text_title

def article_scraper(text):
    #get text content of article
    try:
        article = Article(text)
        article.download()
        article.parse()
    except:
        return '', ''
    # returns - article text in one line, title of article
    return article.text.replace('\n', ''), article.title

def summary_generator(text):
    # creates summary using BART transformer
    # the model used:
    pre_trained_model = 'facebook/bart-large-cnn'
    # creates tokenizer, pretrained on the model
    tokenizer = BartTokenizer.from_pretrained(pre_trained_model)
    # tokenizes the input using created tokenizer, sets the maximum token length to the max length of the model used, uses pytorch tensors
    inputs = tokenizer.batch_encode_plus([text], truncation=True, max_length=tokenizer.model_max_length, return_tensors='pt')
    # creates model using BART for conditional generation, pretrained on model
    model = BartForConditionalGeneration.from_pretrained(pre_trained_model)
    # selects the sumary tokens
    summary_ids = model.generate(inputs['input_ids'])
    # decodes the summary tokens, not counting any special tokens the model may have generated
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summary_score(summary, text):
    # Uses Rouge to check the accuracy of the summary
    # creates the reference summary using the first 3 sentences of the text 
    # (may not be the best way to do it but I can't think of a better way)
    reference_summary = '\n'.join(sent_tokenize(text)[:3])
    # calculates the Rouge scores of the summary compared with the reference summary
    scores = (Rouge().get_scores(summary, reference_summary))[0]
    # scores_list = [scores['rouge-1']['f'], scores['rouge-2']['f'], scores['rouge-l']['f']]
    # just uses the F1 score from rouge-2
    return scores['rouge-2']['f']

def summariser(text):
    # returns - error, text_title, summary
    if url(text):
        # if url inputted, summarise the article in that url
        text, text_title = article_scraper(text)
        if not text:
            return 'That article cannot be summarised', None, None
    else:
        # if input not url, summarise wiki article of input
        text, text_title = wiki_scraper(text)
        if not text:
            return 'That topic cannot be summarised', None, None
    
    summary = summary_generator(text)

    new_summary = sent_tokenize(summary)
    if new_summary[-1][-1] != '.':
        new_summary.pop()
    summary = ' '.join(new_summary)
    summary = summary.replace('; ', '')

    score = summary_score(summary, text)

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
    The reference summary used is the first three sentences of the article.<br><br>
    <a href="https://github.com/Kaziksobo/Summarizer">Github</a>
</p>"""

ui = gradio.Interface(
    summariser, 
    [
        gradio.inputs.Textbox(placeholder='Enter topic/link to article', label = 'Text')
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
