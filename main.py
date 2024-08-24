import os
from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from together import Together

app = Flask(__name__)

def fetch_competitors(query, location):
    search_url = f"https://www.google.com/search?q=Top+10+national+competitors+of+{query}+in+{location}&num=8"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    
    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    competitors = []

    for result in soup.find_all('div', class_='BNeawe s3v9rd AP7Wnd'):
        text = result.get_text()
        competitors.append(text)

    return competitors

def compute_similarity_scores(competitor_names, context_text):
    vectorizer = TfidfVectorizer()
    documents = [context_text] + competitor_names
    tfidf_matrix = vectorizer.fit_transform(documents)
    context_vector = tfidf_matrix[0:1]
    competitor_vectors = tfidf_matrix[1:]
    similarity_scores = cosine_similarity(context_vector, competitor_vectors).flatten()
    scores = dict(zip(competitor_names, similarity_scores))
    return scores

@app.route('/get_competitors', methods=['POST'])
def get_competitors():
    data = request.json
    business_name = data.get('business')
    location = data.get('location')
    # Define the business and location
    business_name = "TATA MOTORS"
    #location = "India"
    
    # Fetch the competitors data
    competitors_list = fetch_competitors(business_name, location)
    print(competitors_list)
    
    # Format the competitors list into a string
    competitors_text = "\n".join(competitors_list)
    
    api_key = os.getenv("TOGETHER_API_KEY")
    
    # Use the Together API to get the response
    client = Together(api_key=api_key)
    
    # Craft the message content with dynamic business name
    message_content = f"Give just names of top 10 competitors of {business_name} from text:\n{competitors_text}"
    
    stream = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        messages=[{"role": "user", "content": message_content}],
        max_tokens=512,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1,
        stop=["<|eot_id|>", "<|eom_id|>"],
        stream=True,
    )
    
    # for chunk in stream:
    #     print(chunk.choices[0].delta.content or "", end="", flush=True)
    
    complete_response_text = ""
    
    for chunk in stream:
        content = chunk.choices[0].delta.content
        # print(content)
        if content:
            complete_response_text += content
    
    print(complete_response_text)
    
    # print(complete_response_text)
    # Given output text
    output_text = complete_response_text
    
    # Regular expression pattern to extract competitors
    pattern = r'\d+\.\s*([^0-9]+?)(?=\n\d+\.\s|$)'
    
    # Extract matches
    matches = re.findall(pattern, output_text, re.DOTALL)
    
    # Clean and format the competitor names
    competitor_names = [name.strip() for name in matches if name.strip()]
    
    # Print the list of competitor names
    # print(competitor_names)
    
    # Given list
    output = competitor_names
    # Remove the last item’s extra text
    cleaned_last_item = output[-1].split('\n\n')[0].strip()
    
    # Update the list with the cleaned last item
    competitor_names = output[:-1] + [cleaned_last_item]
    
    # Print the list of competitor names
    print(competitor_names)
    
    def compute_similarity_scores(competitor_names, context_text):
        vectorizer = TfidfVectorizer()
        documents = [context_text] + competitor_names
        tfidf_matrix = vectorizer.fit_transform(documents)
        # Compute similarity scores
        context_vector = tfidf_matrix[0:1]
        competitor_vectors = tfidf_matrix[1:]
        similarity_scores = cosine_similarity(context_vector, competitor_vectors).flatten()
        scores = dict(zip(competitor_names, similarity_scores))
        return scores
    
    scores = compute_similarity_scores(competitor_names, competitors_text)
    print(scores)
    return scores

def fetch_competitors(query, location):
    # Example URL for scraping (you may need to adapt this for actual use)
    search_url = f"https://www.google.com/search?q=Top+10+national+competitors+of+{query}+in+{location}&num=8"

    # Send request to search URL
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)

    # Check if request was successful
    if response.status_code != 200:
        print("Failed to retrieve search results")
        return []

    # Parse HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    competitors = []

    # Extract competitors (adapt the extraction logic based on actual HTML structure)
    for result in soup.find_all('div', class_='BNeawe s3v9rd AP7Wnd'):
        text = result.get_text()
        competitors.append(text)
    # print(result)

    return competitors




if __name__ == '__main__':
    app.run(debug=True)
