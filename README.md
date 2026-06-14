# Web RAG Chatbot 🤖

A chatbot that reads trusted and authorized websites
like Wikipedia and answers your questions accurately
using AI — no guessing, no assumptions!

## What is RAG?
RAG stands for Retrieval Augmented Generation.
Instead of guessing answers, it:
1. Reads a real website 🌐
2. Finds the most relevant information 🔍
3. Gives you an accurate AI answer ✅

## Why is it trustable?
- Uses authorized websites like Wikipedia
- Answers are based on real website content
  that is verified and proven by real people
- Not based on AI assumptions or human memory
- Fetches the answer directly from the source

## How it works
1. You give it a Wikipedia URL
2. BeautifulSoup reads all the text from that page
3. TF-IDF searches for the most relevant paragraph
   matching your question
4. LLaMA AI reads that paragraph and gives you
   a clean accurate answer

## Technologies Used
- Python
- BeautifulSoup — extracts text from websites
- TF-IDF — smart search to find relevant paragraphs
- LLaMA AI by Meta (hosted on NVIDIA) — generates
  the final answer from the retrieved text

## How to Run

Install libraries:
pip install requests beautifulsoup4 scikit-learn numpy

Run the chatbot:
python chatbot.py

## Example

URL used: https://en.wikipedia.org/wiki/Artificial_intelligence

You: What is deep learning?

Chatbot: Deep learning is a type of machine learning
that uses multiple layers of artificial neural networks
to extract features from raw input data.

## Note
Best results with Wikipedia and other
trusted authorized websites!
