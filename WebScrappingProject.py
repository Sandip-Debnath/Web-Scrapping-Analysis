import requests
from bs4 import BeautifulSoup
import pandas as pd
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from textstat import flesch_reading_ease, syllable_count

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Read URLs from Input.xlsx
input_df = pd.read_excel('Input.xlsx')

# Function to extract article text and save to file
def extract_and_save(url, url_id):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract article title
        article_title = soup.find('title').get_text()

        # Identify the tag and class that contains the article content
        article_content = find_article_content(soup)  # Helper function to find article content
        if article_content:
            # Extract article text
            article_text = ' '.join([p.get_text() for p in article_content.find_all('p')])  # Assuming paragraphs are used

            # Save to file
            with open(f"{url_id}.txt", 'w', encoding='utf-8') as file:
                file.write(f"{article_title}\n\n{article_text}")
        else:
            raise ValueError(f"Could not find article content on {url}")
    except requests.exceptions.RequestException as e:
        print(f"Error accessing {url}: {e}")
        raise  # Re-raise the exception to indicate failure

# Function to find article content based on variations in HTML structure
def find_article_content(soup):
    # Replace 'article-content' with the actual class or tag that contains the article content
    article_content = soup.find('div', class_='article-content')
    if not article_content:
        article_content = soup.find('article')  # Or try finding by 'article' tag

    return article_content

# Loop through URLs and extract data
rows_to_drop = []  # List to store indices of rows to be dropped
for index, row in input_df.iterrows():
    try:
        extract_and_save(row['URL'], row['URL_ID'])
    except ValueError as ve:
        print(ve)
        rows_to_drop.append(index)

# Drop rows where articles are not found
input_df.drop(rows_to_drop, inplace=True)

# Add columns for output attributes if not present in the input file
output_columns = [
    'POSITIVE SCORE',
    'NEGATIVE SCORE',
    'POLARITY SCORE',
    'SUBJECTIVITY SCORE',
    'AVG SENTENCE LENGTH',
    'PERCENTAGE OF COMPLEX WORDS',
    'FOG INDEX',
    'AVG NUMBER OF WORDS PER SENTENCE',
    'COMPLEX WORD COUNT',
    'WORD COUNT',
    'SYLLABLE PER WORD',
    'PERSONAL PRONOUNS',
    'AVG WORD LENGTH'
]

for col in output_columns:
    if col not in input_df.columns:
        input_df[col] = None

# Function to perform text analysis
def perform_text_analysis(text):
    blob = TextBlob(text)
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)

    stop_words = set(stopwords.words('english'))

    # Compute variables
    positive_score = blob.sentiment.polarity
    negative_score = blob.sentiment.subjectivity
    polarity_score = blob.sentiment.polarity
    subjectivity_score = blob.sentiment.subjectivity
    avg_sentence_length = len(words) / len(sentences)
    percentage_of_complex_words = len([word for word in words if len(word) > 6 and word.lower() not in stop_words]) / len(words)
    fog_index = 0.4 * (avg_sentence_length + percentage_of_complex_words)
    avg_number_of_words_per_sentence = len(words) / len(sentences)
    complex_word_count = len([word for word in words if len(word) > 6 and word.lower() not in stop_words])
    word_count = len(words)
    syllable_per_word = syllable_count(text) / len(words)
    personal_pronouns = sum(1 for word in words if word.lower() in ['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves'])
    avg_word_length = sum(len(word) for word in words) / len(words)

    return positive_score, negative_score, polarity_score, subjectivity_score, \
           avg_sentence_length, percentage_of_complex_words, fog_index, \
           avg_number_of_words_per_sentence, complex_word_count, word_count, \
           syllable_per_word, personal_pronouns, avg_word_length

# Loop through text files and perform analysis
for index, row in input_df.iterrows():
    with open(f"{row['URL_ID']}.txt", 'r', encoding='utf-8') as file:
        article_text = file.read()

    # Perform text analysis
    analysis_results = perform_text_analysis(article_text)

    # Update the output DataFrame
    for col, result in zip(output_columns, analysis_results):
        input_df.at[index, col] = result

# Save the updated DataFrame to Output Data Structure.xlsx
input_df.to_excel("Output Data Structure.xlsx", index=False)
