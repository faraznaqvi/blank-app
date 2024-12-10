import json
import time
import random  # Add this import statement to use random
import streamlit as st
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re
import boto3
from langchain.llms import Bedrock

nltk.download('punkt')

article_path = "https://scholar.google.com/citations?view_op=view_citation&hl=en&citation_for_view="
JINA_API = "jina_b70bd77a2f48420f9ddcba6340c1365ewYBbh-aCAMiFUJ4qkF-bIXbzUAxW"
# Step 1: Set up DynamoDB Client
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
dynamodb_client = boto3.client('dynamodb', region_name='us-east-1')
gscholar_table = dynamodb.Table('links')
abstract_table = dynamodb.Table('article_metadata')
bedrock_client = boto3.client(service_name="bedrock-runtime")

# Initialize Bedrock client
bedrock_client = boto3.client("bedrock-runtime")
llm = Bedrock(
    model_id="anthropic.claude-v2:1",
    client=bedrock_client,
    model_kwargs={'max_tokens_to_sample': 512}
)


def save_to_dynamodb(user_id, citation_id):
    try:
        # Check if user_id exists in the table
        response = gscholar_table.get_item(Key={'gscholar_id': user_id})
        if 'Item' in response:
            print(f"User ID {user_id} already exists in DynamoDB. Skipping insertion.")
            return

        # Add new user_id if not found
        gscholar_table.put_item(
            Item={
                'gscholar_id': user_id,
                'link_id': citation_id,
            }
        )
        print(f"Data successfully saved to DynamoDB for User ID {user_id}.")
    except Exception as e:
        print(f"Error saving data to DynamoDB: {e}")


def save_abstract(abstract, citation_id):
    try:
        # Check if user_id exists in the table
        response = abstract_table.get_item(Key={'citation_id': citation_id})
        if 'Item' in response:
            print(f"Citation ID {citation_id} already exists in DynamoDB. Skipping insertion.")
            return

        print(abstract['Description'])

        # Add new user_id if not found
        abstract_table.put_item(
            Item={
                'citation_id': citation_id,
                'abstract': abstract['Description'],
                'authors': abstract['Authors'],
                'link': abstract['link'],
                'publishedDate': abstract['Published Date'],
            }
        )
        print(f"Data successfully saved to DynamoDB for User ID {citation_id}.")
    except Exception as e:
        print(f"Error saving data to DynamoDB: {e}")


def filter_non_existing_ids(table_name, ids):
    """Filters out IDs that already exist in DynamoDB."""
    try:
        # Batch Get Request
        client = boto3.client('dynamodb', region_name='us-east-1')  # Replace 'your-region' with your AWS region
        keys = [{'citation_id': {'S': id}} for id in ids]

        response = client.batch_get_item(
            RequestItems={
                table_name: {
                    'Keys': keys
                }
            }
        )

        # Extract existing IDs from response
        existing_items = response['Responses'].get(table_name, [])
        existing_ids = {item['citation_id']['S'] for item in existing_items}

        # Filter out IDs that already exist in the table
        non_existing_ids = [id for id in ids if id not in existing_ids]

        return non_existing_ids

    except Exception as e:
        print(f"Error checking IDs in DynamoDB: {e}")
        return []


# Helper functions
def parse_jina_response(response_text, user_id):
    """Parse the Jina API response to extract research paper links."""
    try:
        links = re.findall(r'\((https?://[^\s)]+)\)', response_text)
        filtered_links = [link for link in links if
                          'title' not in link and 'pubdate' not in link and 'cites' not in link][0:10]
        citation_ids = [extract_citation_id(link) for link in filtered_links]
        citation_ids = filter_non_existing_ids('article_metadata', citation_ids)
        save_to_dynamodb(user_id, citation_ids)
        # Print the filtered links
        # for link in filtered_links:
        # print(link)
        # soup = BeautifulSoup(response_text, 'html.parser')
        # table = soup.find('table')
        # links = []
        #
        # if table:
        #     rows = table.find_all('tr')[1:]  # Skip header row
        #     for row in rows:
        #         columns = row.find_all('td')
        #         if len(columns) > 0:
        #             link = columns[0].find('a', href=True)
        #             if link:
        #                 links.append(link['href'])

        return citation_ids
    except Exception as e:
        st.error(f"Error parsing Jina response: {e}")
        return []


def extract_citation_id(url):
    """Extracts the 'citation_for_view' ID from the URL."""
    match = re.search(r'citation_for_view=([\w:-]+)', url)
    return match.group(1) if match else None


def extract_user_id(scholar_url):
    match = re.search(r'user=([\w-]+)', scholar_url)
    return match.group(1) if match else None


def scrape_google_scholar(url, keyword):
    user_id = extract_user_id(url)
    try:
        headers = {
            "Authorization": f"Bearer {JINA_API}"
        }
        response = requests.get(f"https://r.jina.ai/{url}", headers=headers)
        if response.status_code == 200:
            return parse_jina_response(response.text, user_id)
        else:
            st.error(f"Failed to fetch data: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error scraping Google Scholar: {e}")
        return []


def extract_abstract(citation_id):
    """Extract abstract text from a research paper URL."""
    try:
        headers = {
            "Authorization": f"Bearer {JINA_API}"
        }
        # st.write(f"{citation_id} ID not found")
        response = requests.get(f"https://r.jina.ai/{article_path}{citation_id}", headers=headers)
        return response.text
    except Exception as e:
        st.error(f"Error extracting abstract from {url}: {e}")
        return ""


def extract_metadata_from_abstract(abstracts, citation_id):
    """Extract metadata from an abstract using Bedrock Claude-v2."""
    prompt = (
            """Extract the following information from the provided text and output it in strict JSON format. Use the exact key names as specified below, and ensure the structure is not altered:

            Required JSON keys and their descriptions:  (All keys should be in sentence format: Only first letter capital)
            1. "Published Date": The publication date of the article.  
            2. "Authors": A list of authors contributing to the article.  
            3. "JournalType": The type of publication (e.g., Journal, Conference, Patent).  
            4. "Description": A summary or abstract of the article.  
            5. "link": The URL link to the article.  (the link should not be from scholar.google.com)

            Guidelines for valid JSON output:  
            1. Keys must be enclosed in double quotes (").  
            2. String values must also be enclosed in double quotes (").  
            3. The structure and key names must match the exact format provided above.

            Input Abstract: """ + abstracts + """
            Output: Strict JSON format with the specified keys.
        """
    )
    retries = 5
    delay = 1  # initial delay in seconds

    for attempt in range(retries):
        try:
            response = llm(prompt)
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            json_str = response[start_idx:end_idx]

            # Convert to a Python dictionary
            response = json.loads(json_str)
            save_abstract(response, citation_id)
            return response
        except Exception as e:
            if "ThrottlingException" in str(e) and attempt < retries - 1:
                # Exponential backoff: wait before retrying
                wait_time = delay * (2 ** attempt) + random.uniform(0, 1)
                # st.warning(f"Throttling error encountered. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                # st.error(f"Error extracting metadata using Claude: {e}")
                return ""
    return ""


# def rank_abstracts(abstracts, research_interest):
#     """Rank abstracts based on similarity to the research interest."""
#     corpus = [research_interest] + [entry['abstract']['S'] for entry in abstracts]
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
#     similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
#
#     ranked_abstracts = sorted(zip(abstracts, similarity_scores), key=lambda x: x[1], reverse=True)
#     return ranked_abstracts


def rank_abstracts(abstracts_and_links, research_interest):
    """Rank abstracts based on similarity to the research interest."""

    # Create the corpus including the research interest and the abstracts
    corpus = [research_interest] + [entry['abstract'] for entry in abstracts_and_links]

    # Vectorize the corpus using TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    # Compute cosine similarity between research interest and abstracts
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten() * 100

    # Sort abstracts by similarity scores
    ranked_abstracts = sorted(zip(abstracts_and_links, similarity_scores), key=lambda x: x[1], reverse=True)

    return ranked_abstracts

# Chunk keys into batches of 100 items (DynamoDB limit per request)
def chunk_keys(keys, chunk_size=100):
    for i in range(0, len(keys), chunk_size):
        yield keys[i:i + chunk_size]

# Streamlit UI
st.set_page_config(
    page_title="Research Abstract Finder",
    page_icon="📚",
    layout="wide",  # Can also be 'centered'
    initial_sidebar_state="expanded",
)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Find & Rank Abstracts", "About"]
)

# About page
if page == "About":
    st.title("About This App")
    st.markdown("""
        This app allows researchers to:
        - Extract abstracts and metadata from Google Scholar.
        - Save metadata to DynamoDB.
        - Rank abstracts based on similarity to a given research interest.

        **Powered by Streamlit and AWS Services.**
    """)
    st.sidebar.markdown("Navigate to other pages using the radio buttons above.")

# Home page
elif page == "Home":
    st.title("Welcome to the Research Abstract Finder")
    st.markdown("""
        Use this app to extract and rank research abstracts.
        Navigate to the **Find & Rank Abstracts** page to get started!
    """)

# Find & Rank Abstracts page
elif page == "Find & Rank Abstracts":
    st.title("Research Abstract Finder and Ranker")

    # Input fields
    st.sidebar.header("Inputs")
    url = st.sidebar.text_input("Google Scholar URL")
    research_interest = st.sidebar.text_area("Research Interest")
    keyword = st.sidebar.text_input("Keyword")

    if st.sidebar.button("Rank Abstracts"):
        if not url or not research_interest or not keyword:
            st.error("Please fill in all fields.")
        else:
            citation_ids = scrape_google_scholar(url, keyword)

            if not citation_ids:
                print("All links are saved.")
            else:
                abstracts = []
                st.info("Extracting abstracts...")
                for citation_id in citation_ids:
                    abstract = extract_abstract(citation_id)
                    if abstract:
                        abstracts.append([abstract, citation_id])

                if not abstracts:
                    st.warning("No abstracts could be extracted.")
                else:
                    st.success("Abstracts extracted. Extracting metadata...")
                    for abstract in abstracts:
                        extract_metadata_from_abstract(abstract[0], abstract[1])

            st.success("Ranking abstracts...")
            user_id = extract_user_id(url)
            response = gscholar_table.get_item(Key={'gscholar_id': user_id})

            abstracts_and_links = []

            if 'Item' in response and 'link_id' in response['Item']:
                link_ids = response['Item']['link_id']

                if link_ids:
                    keys = [{'citation_id': {'S': link_id}} for link_id in link_ids]

                    for key_batch in chunk_keys(keys):
                        request_items = {
                            'article_metadata': {
                                'Keys': key_batch
                            }
                        }
                        batch_response = dynamodb_client.batch_get_item(RequestItems=request_items)
                        data = batch_response['Responses']['article_metadata']

                        abstracts_and_links.extend(
                            [{'link': entry['link']['S'], 'abstract': entry['abstract']['S']} for entry in data])

            if not abstracts_and_links:
                st.error("No abstracts found to rank.")
            else:
                abstracts_and_links = [entry for entry in abstracts_and_links if entry['abstract']]

                if abstracts_and_links:
                    ranked_abstracts = rank_abstracts(abstracts_and_links, research_interest)

                    st.write("### Ranked Abstracts with Metadata")
                    for i, (metadata, score) in enumerate(ranked_abstracts):
                        st.markdown(f"""
                             <div style="
                                border: 1px solid #ddd; 
                                border-radius: 8px; 
                                padding: 15px; 
                                margin-bottom: 10px; 
                                background-color: #f0f8ff;  /* Soft light blue background */
                                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                            ">
                                <h4 style="font-size: 18px; font-weight: bold; color: #2a4d77;">Rank {i + 1}: (Score: {score:.2f}%)</h4>
                                <p style="color: #4d4d4d; font-size: 16px;"><strong>Abstract:</strong> {metadata['abstract']}</p>
                                <p style="color: #0b0e10; font-size: 16px;"><strong>Link:</strong> <a href="{metadata['link']}" target="_blank" style="color: #007acc;">{metadata['link']}</a></p>
                            </div>
                            """, unsafe_allow_html=True)
                    # for i, (metadata, score) in enumerate(ranked_abstracts):
                    #     st.write(f"**Rank {i + 1}:** (Score: {score:.2f}%)")
                    #     st.write(f"**Abstract:** {metadata['abstract']}")
                    #     st.write(f"**Link:** {metadata['link']}")
                    #     st.write("---")
                else:
                    st.error("No valid abstracts found for ranking.")
