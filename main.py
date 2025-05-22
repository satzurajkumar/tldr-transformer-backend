# main.py
import os
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, EmailStr
from jose import JWTError, jwt
from passlib.context import CryptContext
from starlette.concurrency import run_in_threadpool
from collections import Counter

# --- Configuration & Initialization ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# JWT Configuration - GET THESE FROM ENVIRONMENT VARIABLES IN PRODUCTION
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-super-secret-jwt-key-please-change-me")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Password Hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Initialize FastAPI app
app = FastAPI(
    title="Topic Discoverer API (NLTK with Auth)",
    version="1.2.0",
    description="API for analyzing text to discover topics using NLP (NLTK), with JWT authentication."
)

# CORS Configuration
origins = [
    "chrome-extension://your_extension_id_here", # Replace with your actual extension ID
    "http://localhost", # For local testing if needed
    # Add other origins for development/production frontends if necessary
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Be more specific in production
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# --- NLTK Setup (same as before) ---
lemmatizer = None
nltk_stop_words = None
REQUIRED_NLTK_RESOURCES = {
    "tokenizers/punkt": "punkt", "corpora/stopwords": "stopwords",
    "taggers/averaged_perceptron_tagger": "averaged_perceptron_tagger",
    "corpora/wordnet": "wordnet", "corpora/omw-1.4": "omw-1.4"
}

@app.on_event("startup")
async def startup_event():
    global lemmatizer, nltk_stop_words
    logger.info("Initializing NLTK resources...")
    for path, resource_id in REQUIRED_NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
            logger.info(f"NLTK resource '{resource_id}' found.")
        except LookupError:
            logger.warning(f"NLTK resource '{resource_id}' not found. Downloading...")
            try:
                nltk.download(resource_id, quiet=True)
                logger.info(f"NLTK resource '{resource_id}' downloaded successfully.")
            except Exception as download_e:
                logger.error(f"Failed to download NLTK resource '{resource_id}': {download_e}")
    lemmatizer = WordNetLemmatizer()
    nltk_stop_words = set(stopwords.words('english'))
    logger.info("NLTK Lemmatizer and stopwords initialized.")

# --- User Data Store (In-memory for this example) ---
# In a real application, use a database (e.g., PostgreSQL, MySQL with SQLAlchemy)
fake_users_db: Dict[str, Dict[str, str]] = {} # username: {"hashed_password": "...", "email": "..."}

# --- Pydantic Models ---
class UserBase(BaseModel):
    username: str = Field(..., min_length=3)
    email: Optional[EmailStr] = None

class UserCreate(UserBase):
    password: str = Field(..., min_length=6)

class UserInDB(UserBase):
    hashed_password: str

class UserPublic(UserBase): # For returning user info without password
    pass

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class TextPayload(BaseModel):
    text: str = Field(..., min_length=50)

class KeywordsResponse(BaseModel):
    keywords: List[str]
    message: Optional[str] = None

class ErrorResponse(BaseModel): # Re-added for consistency if needed, though FastAPI handles many
    detail: str

# --- Authentication Utilities ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login") # Points to the login endpoint

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserPublic:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user_data = fake_users_db.get(token_data.username)
    if user_data is None:
        raise credentials_exception
    return UserPublic(username=token_data.username, email=user_data.get("email"))


# --- Authentication Endpoints ---
auth_router = FastAPI() # Using a sub-router for auth clarity

@auth_router.post("/register", response_model=UserPublic, status_code=status.HTTP_201_CREATED)
async def register_user(user: UserCreate):
    if user.username in fake_users_db:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user.password)
    fake_users_db[user.username] = {"hashed_password": hashed_password, "email": user.email}
    logger.info(f"User '{user.username}' registered successfully.")
    return UserPublic(username=user.username, email=user.email)

@auth_router.post("/login", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user_data = fake_users_db.get(form_data.username)
    if not user_data or not verify_password(form_data.password, user_data["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": form_data.username}, expires_delta=access_token_expires
    )
    logger.info(f"User '{form_data.username}' logged in successfully.")
    return {"access_token": access_token, "token_type": "bearer"}

@auth_router.get("/users/me", response_model=UserPublic)
async def read_users_me(current_user: UserPublic = Depends(get_current_user)):
    return current_user

app.include_router(auth_router, prefix="/auth", tags=["authentication"])


# --- NLTK Processing Logic (same as before) ---
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'): return wordnet.ADJ
    elif treebank_tag.startswith('V'): return wordnet.VERB
    elif treebank_tag.startswith('N'): return wordnet.NOUN
    elif treebank_tag.startswith('R'): return wordnet.ADV
    else: return wordnet.NOUN

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
        raise RuntimeError("NLP resources not available.")
    words = word_tokenize(text_to_analyze.lower())
    all_stop_words = nltk_stop_words.union(CUSTOM_STOP_WORDS)
    filtered_words = [w for w in words if w.isalpha() and w not in all_stop_words and len(w) > 2]
    if not filtered_words: return [], "No meaningful words after filtering."
    pos_tags = nltk.pos_tag(filtered_words)
    lemmatized_nouns = []
    for word, tag in pos_tags:
        if tag.startswith('NN'):
            lemma = lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag))
            if len(lemma) > 2 and lemma not in all_stop_words:
                lemmatized_nouns.append(lemma)
    if not lemmatized_nouns: return [], "No nouns after lemmatization."
    keyword_counts = Counter(lemmatized_nouns)
    min_freq = 2 if len(keyword_counts) > 10 else 1
    top_kws = [kw for kw, count in keyword_counts.most_common(20) if count >= min_freq]
    final_kws = sorted(list(set(top_kws)), key=lambda x: keyword_counts[x], reverse=True)[:10]
    if not final_kws: return [], "Keywords did not meet frequency/uniqueness criteria."
    return final_kws, "Keywords extracted successfully (NLTK)."

# --- Protected Analyze Endpoint ---
@app.post(
    "/analyze",
    response_model=KeywordsResponse,
    summary="Analyze text for keywords (NLTK, Auth Required)",
)
async def analyze_text_endpoint(
    payload: TextPayload,
    current_user: UserPublic = Depends(get_current_user) # This protects the endpoint
):
    if not lemmatizer or not nltk_stop_words:
        raise HTTPException(status_code=503, detail="NLP service not ready.")
    try:
        logger.info(f"User '{current_user.username}' requested analysis for text: {payload.text[:50]}...")
        keywords, message = await run_in_threadpool(process_text_with_nltk, payload.text)
        return KeywordsResponse(keywords=keywords, message=message)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error during NLTK analysis for user '{current_user.username}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error during analysis.")

# --- Health Check  ---
@app.get("/health", summary="Health Check")
async def health_check():
    if lemmatizer and nltk_stop_words:
        return {"status": "ok", "message": "API running, NLTK resources initialized."}
    else:
       
        return {"status": "degraded", "message": "API running, NLTK resources not fully initialized."}


