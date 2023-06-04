import requests
import PyPDF2
import pytesseract
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from summa import keywords
from transformers import pipeline, AutoTokenizer

# Extract text from the PDF using PyPDF2
pdf_file = open('school_handbook.pdf', 'rb')
pdf_reader = PyPDF2.PdfReader(pdf_file)

# Initialize the NLP pipeline for question-answering using transformers
question_answering_pipeline = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad', tokenizer='distilbert-base-uncased-distilled-squad')

# Iterate over all pages of the PDF
for page_number in range(len(pdf_reader.pages)):
    page = pdf_reader.pages[page_number]
    text = page.extract_text()
    sentences = sent_tokenize(text)
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    filtered_sentences = [sentence for sentence in sentences if sentence.lower() not in stop_words]
    max_segment_length = 1024

    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    text_chunks = []
    current_chunk = ""
    for sentence in filtered_sentences:
        encoded = tokenizer.encode(sentence, add_special_tokens=True)
        if len(current_chunk) + len(encoded) < max_segment_length:
            current_chunk += " " + sentence
        else:
            text_chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        text_chunks.append(current_chunk.strip())

    for chunk in text_chunks:
        question = input("Enter your question: ")
        result = question_answering_pipeline(question=question, context=chunk)

        print("Answer:", result['answer'])
        print("Score:", result['score'])
        print()
