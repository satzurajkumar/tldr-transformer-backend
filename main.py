# main.py
import os
import logging
import re
from typing import List, Optional

import nltk # Ensure NLTK is imported
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer


from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Security, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool # For running sync code in async context
from collections import Counter

# --- Configuration & Initialization ---

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Key Configuration
EXPECTED_API_KEY = os.getenv("TOPIC_DISCOVERER_API_KEY", "your_super_secret_api_key_here_12345")
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Initialize FastAPI app
app = FastAPI(
    title="Topic Discoverer API (NLTK)",
    version="1.1.1", # Incremented version for the fix
    description="API for analyzing text to discover topics using NLP (NLTK)."
)

# CORS Configuration
origins = [
    "chrome-extension://your_extension_id_here", # Replace with your actual extension ID
    "http://localhost",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*", API_KEY_NAME, "Content-Type"],
)

# --- NLTK Setup ---
lemmatizer = None
nltk_stop_words = None

REQUIRED_NLTK_RESOURCES = {
    "tokenizers/punkt": "punkt",
    "corpora/stopwords": "stopwords",
    "taggers/averaged_perceptron_tagger": "averaged_perceptron_tagger",
    "corpora/wordnet": "wordnet",
    "corpora/omw-1.4": "omw-1.4"  # Open Multilingual Wordnet, needed by WordNetLemmatizer
}

@app.on_event("startup")
async def startup_event():
    global lemmatizer, nltk_stop_words
    logger.info("Initializing NLTK resources...")
    for path, resource_id in REQUIRED_NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
            logger.info(f"NLTK resource '{resource_id}' found.")
        except LookupError: # Corrected: Only catch LookupError here
            logger.warning(f"NLTK resource '{resource_id}' not found. Downloading...")
            try:
                nltk.download(resource_id, quiet=True)
                logger.info(f"NLTK resource '{resource_id}' downloaded successfully.")
            except Exception as download_e: # Generic exception for download failure
                logger.error(f"Failed to download NLTK resource '{resource_id}': {download_e}")
                # Depending on the resource, you might want to raise an error or exit
    
    lemmatizer = WordNetLemmatizer()
    nltk_stop_words = set(stopwords.words('english'))
    logger.info("NLTK Lemmatizer and stopwords initialized.")


# --- Pydantic Models (remain the same) ---
class TextPayload(BaseModel):
    text: str = Field(..., min_length=50, description="Text content to be analyzed. Minimum 50 characters.")

class KeywordsResponse(BaseModel):
    keywords: List[str]
    message: Optional[str] = None

class ErrorResponse(BaseModel):
    detail: str

# --- API Key Dependency (remains the same) ---
async def get_api_key(api_key_received: str = Security(api_key_header)):
    if not api_key_received:
        logger.warning("API key missing in request.")
        raise HTTPException(status_code=403, detail="API key is missing.")
    if api_key_received == EXPECTED_API_KEY:
        return api_key_received
    else:
        logger.warning("Invalid API key received.")
        raise HTTPException(status_code=403, detail="Invalid API key.")

# --- Helper function for NLTK POS tagging to WordNet POS tags ---
def get_wordnet_pos(treebank_tag):
    """Converts treebank POS tags to WordNet POS tags for lemmatization."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN # Default to noun

# --- Helper function for NLP processing with NLTK ---
CUSTOM_STOP_WORDS = {
    "page", "site", "website", "article", "content", "information", "click", "view", "menu",
    "home", "search", "contact", "news", "post", "blog", "comment", "read", "more", "share",
    "like", "follow", "also", "however", "therefore", "example", "e.g.", "i.e.", "etc", "fig",
    "image", "photo", "video", "copyright", "rights", "reserved", "terms", "privacy", "policy",
    "subscribe", "login", "logout", "register", "account", "learn", "help", "faq", "support",
    "services", "products", "company", "about", "us", "welcome", "thank", "thanks",
    "nbsp", "amp", "quot", "apos", "lt", "gt", "advertisement", "cookie", "cookies", "settings",
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "today", "yesterday", "tomorrow",
    "get", "make", "see", "say", "tell", "ask", "go", "come", "know", "time", "year", "day", "way", "man", "thing", "woman", "life", "child", "world", "school", "state", "family", "student", "group", "country", "problem", "hand", "part", "place", "case", "week", "system", "program", "question", "work", "government", "number", "night", "point", "home", "water", "room", "mother", "area", "money", "story", "fact", "month", "lot", "right", "study", "book", "eye", "job", "word", "business", "issue", "side", "kind", "head", "house", "service", "friend", "father", "power", "hour", "game", "line", "end", "member", "law", "car", "city", "community", "name", "president", "team", "minute", "idea", "kid", "body", "back", "parent", "face", "others", "level", "office", "door", "health", "person", "art", "war", "history", "party", "result", "change", "morning", "reason", "research", "girl", "guy", "moment", "air", "teacher", "force", "education"
}

def process_text_with_nltk(text_to_analyze: str):
    global lemmatizer, nltk_stop_words 

    if not lemmatizer or not nltk_stop_words:
        logger.error("NLTK lemmatizer or stopwords not initialized. Cannot process text.")
        raise RuntimeError("NLP resources not available. Initialization might have failed.")

    words = word_tokenize(text_to_analyze.lower())
    all_stop_words = nltk_stop_words.union(CUSTOM_STOP_WORDS)
    
    filtered_words = [
        word for word in words 
        if word.isalpha() and word not in all_stop_words and len(word) > 2
    ]
    
    if not filtered_words:
        return [], "No meaningful words found after initial filtering."

    pos_tags = nltk.pos_tag(filtered_words)
    lemmatized_nouns = []
    for word, tag in pos_tags:
        if tag.startswith('NN'): 
            lemma = lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag))
            if len(lemma) > 2 and lemma not in all_stop_words : 
                 lemmatized_nouns.append(lemma)
    
    if not lemmatized_nouns:
        return [], "No nouns found after POS tagging and lemmatization."

    keyword_counts = Counter(lemmatized_nouns)
    min_frequency = 2 if len(keyword_counts) > 10 else 1
    top_keywords = [kw for kw, count in keyword_counts.most_common(20) if count >= min_frequency]
    final_keywords = sorted(list(set(top_keywords)), key=lambda x: keyword_counts[x], reverse=True)[:10]

    if not final_keywords:
        return [], "Keywords found but did not meet frequency or uniqueness criteria."
        
    return final_keywords, "Keywords extracted successfully using NLTK."


# --- API Endpoints ---
@app.post(
    "/analyze",
    response_model=KeywordsResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        403: {"model": ErrorResponse, "description": "API key missing or invalid"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable (e.g., NLP resources not loaded)"}
    },
    summary="Analyze text for keywords (NLTK)",
    description="Accepts text, returns potential keywords based on NLTK analysis (noun extraction, lemmatization)."
)
async def analyze_text_endpoint(
    payload: TextPayload,
    api_key: str = Depends(get_api_key)
):
    if not lemmatizer or not nltk_stop_words: 
        logger.error("Attempted to use /analyze endpoint but NLTK resources are not initialized.")
        raise HTTPException(status_code=503, detail="NLP service is not available. Resources not initialized.")

    try:
        logger.info(f"NLTK: Received analysis request for text starting with: {payload.text[:100]}...")
        
        keywords, message = await run_in_threadpool(process_text_with_nltk, payload.text)
        
        logger.info(f"NLTK Analysis complete. Keywords: {keywords}, Message: {message}")
        return KeywordsResponse(keywords=keywords, message=message)

    except RuntimeError as e: 
        logger.error(f"Runtime error during NLTK analysis: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error during NLTK text analysis: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@app.get("/health", summary="Health Check", description="Simple health check endpoint.")
async def health_check():
    if lemmatizer and nltk_stop_words:
        return {"status": "ok", "message": "API is running and NLTK resources are initialized."}
    else:
        missing_resources = []
        if not lemmatizer: missing_resources.append("Lemmatizer")
        if not nltk_stop_words: missing_resources.append("NLTK Stopwords")
        return {
            "status": "degraded", 
            "message": f"API is running, but some NLTK resources are not initialized: {', '.join(missing_resources)}."
        }

# --- How to run (for development) ---
# Run with Uvicorn: uvicorn main:app --reload --port 8000
