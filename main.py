from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import openai
import re
import json
import uuid
from datetime import datetime, timedelta
from collections import defaultdict
import random
from difflib import get_close_matches
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Halawa Wax AI Backend v2.0", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://hamzaahmad536.github.io"  # This covers all paths under the domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_name: Optional[str] = None

class ChatResponse(BaseModel):
    message: str
    product: Optional[Dict[str, Any]] = None
    intent: Optional[str] = None
    entities: Optional[Dict[str, Any]] = None
    actions: Optional[List[Dict[str, Any]]] = None
    escalation_needed: bool = False

# Knowledge Base
KNOWLEDGE_BASE = {
    "aftercare": {
        "waxing": [
            "Avoid hot showers, saunas, and swimming for 24 hours after waxing",
            "Don't apply makeup, lotions, or perfumes to the waxed area for 24 hours",
            "Exfoliate gently 2-3 days after waxing to prevent ingrown hairs",
            "Wear loose clothing to avoid irritation",
            "Apply aloe vera or tea tree oil for soothing relief"
        ]
    },
    "policies": {
        "cancellation": "Free cancellation up to 24 hours before appointment. Late cancellations may incur a 50% charge.",
        "rescheduling": "Reschedule anytime up to 2 hours before your appointment at no cost.",
        "refunds": "100% money-back guarantee if you're not satisfied with our service."
    }
}

# Product Database
PRODUCTS_DB = {
    "waxing_kits": [
        {
            "id": "wk001",
            "name": "Natural Honey Wax Kit",
            "price": 29.99,
            "features": "Natural honey wax, wooden applicators, pre-wax cleanser, aftercare oil",
            "benefits": "Gentle on skin, reduces irritation, easy to use at home, suitable for all skin types",
            "image_url": "https://via.placeholder.com/300x200?text=Honey+Wax+Kit",
            "product_link": "#",
            "skin_type": ["normal", "sensitive", "dry", "oily"],
            "concerns": ["irritation", "redness", "dryness"],
            "category": "waxing_kit"
        },
        {
            "id": "wk002", 
            "name": "Sensitive Skin Wax Kit",
            "price": 34.99,
            "features": "Hypoallergenic wax, soothing gel, aftercare cream, pre-wax oil",
            "benefits": "Specially formulated for sensitive skin, reduces redness, prevents irritation",
            "image_url": "https://via.placeholder.com/300x200?text=Sensitive+Wax+Kit",
            "product_link": "#",
            "skin_type": ["sensitive", "dry"],
            "concerns": ["sensitivity", "dryness", "redness"],
            "category": "waxing_kit"
        },
        {
            "id": "wk003",
            "name": "Professional Wax Kit",
            "price": 49.99,
            "features": "Professional-grade wax, metal applicators, wax warmer, strips",
            "benefits": "Salon-quality results, long-lasting smoothness, professional tools included",
            "image_url": "https://via.placeholder.com/300x200?text=Professional+Wax+Kit",
            "product_link": "#",
            "skin_type": ["normal", "oily"],
            "concerns": ["coarse_hair", "thick_hair"],
            "category": "waxing_kit"
        }
    ],
    "aftercare_products": [
        {
            "id": "ac001",
            "name": "Soothing Aloe Gel",
            "price": 19.99,
            "features": "Pure aloe vera, cooling effect, fast absorption, no sticky residue",
            "benefits": "Reduces redness, soothes irritation, promotes healing, hydrates skin",
            "image_url": "https://via.placeholder.com/300x200?text=Aloe+Gel",
            "product_link": "#",
            "skin_type": ["all"],
            "concerns": ["redness", "irritation", "dryness"],
            "category": "aftercare"
        },
        {
            "id": "ac002",
            "name": "Tea Tree Oil Solution",
            "price": 24.99,
            "features": "Pure tea tree oil, antibacterial properties, natural antiseptic",
            "benefits": "Prevents ingrown hairs, reduces inflammation, natural antibacterial protection",
            "image_url": "https://via.placeholder.com/300x200?text=Tea+Tree+Oil",
            "product_link": "#",
            "skin_type": ["normal", "oily"],
            "concerns": ["ingrown_hairs", "inflammation", "acne"],
            "category": "aftercare"
        },
        {
            "id": "ac003",
            "name": "Moisturizing Aftercare Cream",
            "price": 22.99,
            "features": "Rich moisturizing cream, vitamin E, shea butter, natural oils",
            "benefits": "Deep hydration, prevents dryness, nourishes skin, long-lasting moisture",
            "image_url": "https://via.placeholder.com/300x200?text=Aftercare+Cream",
            "product_link": "#",
            "skin_type": ["dry", "sensitive"],
            "concerns": ["dryness", "dehydration"],
            "category": "aftercare"
        }
    ],
    "pre_wax_products": [
        {
            "id": "pw001",
            "name": "Pre-Wax Cleanser",
            "price": 16.99,
            "features": "Gentle cleanser, removes oils and dirt, prepares skin for waxing",
            "benefits": "Ensures better wax adhesion, prevents irritation, prepares skin properly",
            "image_url": "https://via.placeholder.com/300x200?text=Pre-Wax+Cleanser",
            "product_link": "#",
            "skin_type": ["all"],
            "concerns": ["oily_skin", "dirt"],
            "category": "pre_wax"
        },
        {
            "id": "pw002",
            "name": "Exfoliating Scrub",
            "price": 18.99,
            "features": "Gentle exfoliation, removes dead skin cells, smooths skin surface",
            "benefits": "Prevents ingrown hairs, smoother waxing results, improves skin texture",
            "image_url": "https://via.placeholder.com/300x200?text=Exfoliating+Scrub",
            "product_link": "#",
            "skin_type": ["normal", "oily"],
            "concerns": ["ingrown_hairs", "rough_skin"],
            "category": "pre_wax"
        }
    ]
}

# Session Management
class SessionManager:
    def __init__(self):
        self.sessions = defaultdict(dict)
        self.appointments = {}
        self.orders = {}
    
    def get_session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "user_name": None,
                "preferences": {},
                "last_intent": None,
                "conversation_history": [],
                "frustration_level": 0,
                "current_booking": None
            }
        return self.sessions[session_id]
    
    def update_session(self, session_id, updates):
        session = self.get_session(session_id)
        session.update(updates)
        return session

# Intent Detection
class IntentDetector:
    def __init__(self):
        self.intent_patterns = {
            "booking": [r"book|appointment|schedule|reserve|make.*appointment"],
            "product_inquiry": [r"product|wax|kit|recommend|suggestion"],
            "service_details": [r"service|treatment|price|cost|how.*much"],
            "order_status": [r"order|tracking|delivery|shipped|where.*order"],
            "complaint": [r"problem|issue|wrong|bad|terrible|awful|not.*happy|dissatisfied|complaint"],
            "aftercare": [r"aftercare|care.*after|what.*do.*after|recovery|healing"],
            "casual_chat": [r"hello|hi|hey|how.*you|good.*morning|good.*evening|thanks|thank.*you|bye|goodbye|how.*are.*you|what.*up|nice.*meet.*you|pleasure"],
            "general_questions": [r"what.*is|how.*does|can.*you|tell.*me|explain|describe|what.*about|who.*are.*you|what.*can.*you.*do"],
            "business_hours": [r"hours|open|close|when.*open|business.*hours|working.*hours"],
            "location": [r"where.*are.*you|location|address|directions|near|far"],
            "pricing": [r"price|cost|how.*much|expensive|cheap|affordable|budget"]
        }
    
    def detect_intent(self, message):
        message_lower = message.lower()
        
        # Check for casual chat first (most common)
        for pattern in self.intent_patterns["casual_chat"]:
            if re.search(pattern, message_lower):
                return "casual_chat"
        
        # Check other intents
        for intent, patterns in self.intent_patterns.items():
            if intent == "casual_chat":  # Skip, already checked
                continue
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    return intent
        
        # If no specific intent found, check if it's a general question
        if any(word in message_lower for word in ["what", "how", "why", "when", "where", "who", "can", "tell", "explain"]):
            return "general_questions"
        
        return "casual_chat"  # Default to casual chat for unrecognized messages

# Entity Extraction
class EntityExtractor:
    def __init__(self):
        self.entity_patterns = {
            "date": [r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", r"(today|tomorrow|next week)"],
            "time": [r"(\d{1,2}:\d{2}\s*(?:am|pm)?)", r"(morning|afternoon|evening)"],
            "service_type": [r"(waxing|threading|facial)"],
            "contact": [r"(\d{3}[-.]?\d{3}[-.]?\d{4})"]
        }
    
    def extract_entities(self, message):
        entities = {}
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, message, re.IGNORECASE)
                if matches:
                    entities[entity_type] = matches[0] if isinstance(matches[0], str) else matches[0][0]
        return entities

# Helper to get absolute path for data files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HALAWA_DOCS_PATH = os.path.join(BASE_DIR, 'halawa_docs.csv')
HALAWA_PRODUCTS_PATH = os.path.join(BASE_DIR, 'halawawax_products.csv')
HALAWA_INDEX_PATH = os.path.join(BASE_DIR, 'halawa_index.faiss')

# Initialize components
try:
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct absolute paths for the files
    docs_path = HALAWA_DOCS_PATH
    index_path = HALAWA_INDEX_PATH
    
    print(f"Looking for files in: {script_dir}")
    df = pd.read_csv(docs_path)
    print("Successfully loaded halawa_docs.csv")
    index = faiss.read_index(index_path)
    print("Successfully loaded halawa_index.faiss")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    print("Successfully initialized SentenceTransformer")
    client = openai.OpenAI(
        api_key = os.getenv("OPENAI_API_KEY"),
        base_url='https://openrouter.ai/api/v1'
    )
    print("Successfully initialized OpenAI client")
except Exception as e:
    import traceback
    print(f"\nERROR initializing AI components:")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("Full traceback:")
    print(traceback.format_exc())
    print("\nSetting components to None as fallback...")
    df = None
    index = None
    embedder = None
    client = None

# Initialize instances
session_manager = SessionManager()
intent_detector = IntentDetector()
entity_extractor = EntityExtractor()

# Common misspellings and variations
SPELLING_VARIATIONS = {
    # Product types
    "wax": ["wacks", "waxx", "wacks", "waxing"],
    "waxing": ["waxin", "waxsing", "waxxing"],
    "kit": ["kitt", "kits", "kitt"],
    "cream": ["creme", "crem", "creams"],
    "gel": ["jell", "gels", "jel"],
    "oil": ["oill", "oils", "oyl"],
    "cleanser": ["cleaner", "cleansing", "clean"],
    "scrub": ["scrubb", "scrubs", "exfoliant"],
    "exfoliating": ["exfoliate", "exfoliation"],
    
    # Skin types
    "sensitive": ["sensative", "sensitve", "sensitive"],
    "normal": ["normall", "normel"],
    "dry": ["drie", "dryy", "dri"],
    "oily": ["oillly", "oilyy", "oill"],
    
    # Concerns
    "redness": ["red", "reddness", "rednesss"],
    "irritation": ["irritate", "irritated", "irritating"],
    "dryness": ["drynesss", "drieness"],
    "inflammation": ["inflamation", "inflamatory"],
    "ingrown": ["ingrow", "ingrowns", "ingrowne"],
    "acne": ["acnee", "acneee", "pimples"],
    
    # Product names
    "honey": ["honney", "honee", "honeyy"],
    "aloe": ["alo", "aloe", "aloe vera"],
    "tea tree": ["tea tree", "teatree", "tea-tree"],
    "shea": ["shea butter", "sheabutter", "shea"],
    
    # Common words
    "products": ["product", "produts", "producs"],
    "show": ["sho", "showw", "see"],
    "tell": ["tel", "telll"],
    "about": ["abut", "abou", "aboutt"],
    "more": ["mor", "moree", "mre"],
    "information": ["info", "informaton", "informatin"],
    "details": ["detail", "detals", "detaills"],
    "price": ["pric", "pricee", "cost"],
    "benefits": ["benefit", "benfits", "benifits"],
    "features": ["feature", "featurs", "featur"],
}

def normalize_text(text: str) -> str:
    """Normalize text by converting to lowercase and handling common variations"""
    text = text.lower().strip()
    
    # Replace common misspellings
    for correct, variations in SPELLING_VARIATIONS.items():
        for variation in variations:
            text = text.replace(variation, correct)
    
    return text

def fuzzy_match(query: str, target_list: List[str], threshold: float = 0.6) -> List[str]:
    """Find close matches for a query in a list of targets"""
    query = query.lower()
    matches = []
    
    for target in target_list:
        # Exact match
        if query == target.lower():
            matches.append(target)
            continue
        
        # Partial match
        if query in target.lower() or target.lower() in query:
            matches.append(target)
            continue
        
        # Fuzzy match using difflib
        similarity = get_close_matches(query, [target.lower()], n=1, cutoff=threshold)
        if similarity:
            matches.append(target)
    
    return list(set(matches))

def sprinkle_girlish_emojis(text: str) -> str:
    # Add girlish/beauty emojis to the start and end, and sprinkle in the middle for longer responses
    emojis = ["💖", "✨", "🌸", "👑", "💅🏼", "🧖‍♀️", "🌷", "💕", "🦋", "🌺"]
    import random
    def pick():
        return random.choice(emojis)
    # Add to start and end
    text = f"{pick()} {text.strip()} {pick()}"
    # Sprinkle in the middle for longer responses
    if len(text) > 80:
        parts = text.split('. ')
        for i in range(1, len(parts), 2):
            parts[i] = pick() + ' ' + parts[i]
        text = '. '.join(parts)
    return text

def get_ai_response(user_input: str, context: str = "", conversation_history: List[Dict] = None) -> str:
    """Get AI-powered response for complex queries"""
    try:
        # Prepare conversation history for context
        messages = [
            {
                "role": "system",
                "content": f"""
You are a helpful AI assistant for Halawa Wax, a natural hair removal brand.

Context about the business:
- We specialize in natural waxing products and services
- We offer waxing kits, aftercare products, and pre-wax products
- Our products are designed for different skin types (sensitive, normal, dry, oily, combination)
- We address concerns like redness, irritation, dryness, ingrown hairs, etc.
- We provide professional waxing services at our salon
- We offer full body waxing, facial waxing, and specialized treatments
- Our services typically take 30-60 minutes and cost $25-75
- We have products for pre-wax preparation, waxing, and post-wax care
- We're located in a salon setting and offer appointment booking
- We can help with order status, complaints, and general inquiries

Product Information:
- Natural Honey Wax Kit ($29.99): Natural honey wax, wooden applicators, pre-wax cleanser, aftercare oil
- Sensitive Skin Wax Kit ($34.99): Hypoallergenic wax, soothing gel, aftercare cream, pre-wax oil  
- Professional Wax Kit ($49.99): Professional-grade wax, metal applicators, wax warmer, strips
- Soothing Aloe Gel ($19.99): Pure aloe vera, cooling effect, reduces redness and irritation
- Tea Tree Oil Solution ($24.99): Pure tea tree oil, prevents ingrown hairs, antibacterial
- Moisturizing Aftercare Cream ($22.99): Rich moisturizing cream with vitamin E and shea butter
- Pre-Wax Cleanser ($16.99): Gentle cleanser that prepares skin for waxing
- Exfoliating Scrub ($18.99): Gentle exfoliation to prevent ingrown hairs

Guidelines:
- Aswer in a professional and girlish way because our audience is almost 90% girls , use emojis too.
- Dont give too long answers , try to give answers to the point, only provide details where it is needed
- Be friendly, professional, and helpful , your name is WAXBOT.
- Answer ANY question about waxing, hair removal, skin care, or our business
- If asked about products, services, or appointments, provide comprehensive information
- If you don't know specific details, suggest asking about our products or services
- Keep responses conversational and engaging
- If the query is about something we don't offer, politely redirect to what we do offer
- Always maintain a helpful and positive tone
- Provide practical advice when appropriate
- Be informative about waxing techniques, skin care, and hair removal
- If someone asks about alternatives to waxing, you can discuss them but emphasize our expertise in waxing
- For medical questions, suggest consulting a dermatologist but provide general information
- Be encouraging and supportive, especially for first-time waxing clients
- When asked about "wax" or "your wax", provide comprehensive information about our waxing products and services
- Include pricing when relevant
- Mention benefits and features of products when appropriate

---

IMPORTANT: After following all the above, ALWAYS return your response as HTML code (not plain text or markdown).
- Use <b> for bold, <i> for italics, <h2> or <h3> for headings, and <p> for short paragraphs.
- Never use bullet points, asterisks (*), dashes (-), or markdown formatting.
- Make the response visually appealing and easy to scan.
- If you mention a product or price, start with its name in a <b> tag, then a short summary in a <p>, and highlight key features in <i> tags.
- If you mention a guarantee or offer, use <b> tags.
- Never use more than 3 sentences per paragraph.
- If the user greets, respond with a warm, bold welcome and offer help in HTML.
- If the user asks for info about wax or products, summarize the most relevant product(s) with <b> names and <i> features/benefits.
- If the user asks about aftercare, summarize the most important tips in short, bolded HTML statements.
- Never use markdown links, just plain text for URLs if needed.
- Always keep the tone friendly and professional.
- Do not include <html>, <body>, or <head> tags—just the HTML for the content itself.

Current context: {context}
"""
            }
        ]
        
        # Add conversation history for context
        if conversation_history:
            for msg in conversation_history[-5:]:  # Last 5 messages for context
                if msg.get("role") == "user":
                    messages.append({"role": "user", "content": msg.get("content", "")})
                elif msg.get("role") == "assistant":
                    messages.append({"role": "assistant", "content": msg.get("content", "")})
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        # Use OpenRouter with the same model as app.py
        response = client.chat.completions.create(
            model="mistralai/mistral-small-3.1-24b-instruct:free",
            messages=messages,
            max_tokens=400,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error getting AI response: {e}")
        return "I'm having trouble processing that right now. Could you try rephrasing your question or ask about our products and services?"

def should_use_ai(user_input: str, intent: str, entities: Dict[str, Any]) -> bool:
    """Determine if we should use AI for this query"""
    user_input_lower = user_input.lower()
    
    # Use AI for ALMOST EVERYTHING - only use predefined responses for very specific cases
    
    # Only use predefined responses for these very specific cases
    specific_cases = [
        # Simple greetings only
        (intent == "casual_chat" and len(user_input.split()) <= 3 and any(word in user_input_lower for word in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"])),
        # Booking with complete information
        (intent == "booking" and entities.get("date") and entities.get("time") and entities.get("service_type")),
        # Product inquiry with very specific skin type/concerns
        (intent == "product_inquiry" and len(user_input.split()) <= 5 and any(word in user_input_lower for word in ["sensitive", "normal", "dry", "oily", "combination", "irritation", "redness", "dryness"])),
        # Order status with specific order ID
        (intent == "order_status" and entities.get("order_id") and len(user_input.split()) <= 8)
    ]
    
    # If it matches a very specific case, use predefined response
    if any(specific_cases):
        return False
    
    # For everything else, use AI (which is almost everything!)
    return True

def style_ai_response(text):
    # Remove asterisks and dashes used for lists
    text = re.sub(r"^[\*-]\s*", "", text, flags=re.MULTILINE)
    # Bold product names
    product_names = [
        "Natural Honey Wax Kit", "Sensitive Skin Wax Kit", "Professional Wax Kit",
        "Soothing Aloe Gel", "Tea Tree Oil Solution", "Moisturizing Aftercare Cream",
        "Pre-Wax Cleanser", "Exfoliating Scrub"
    ]
    for name in product_names:
        text = re.sub(rf"\b({re.escape(name)})\b", r"**\1**", text)
    # Bold headings (lines that look like headings)
    text = re.sub(r"^(\w[\w\s]+):$", r"**\1:**", text, flags=re.MULTILINE)
    # Italicize features/benefits keywords
    for word in ["gentle", "soothing", "cooling", "hydrating", "professional", "natural", "hypoallergenic"]:
        text = re.sub(rf"\b({word})\b", r"*\1*", text, flags=re.IGNORECASE)
    # Split long paragraphs into shorter ones
    text = re.sub(r"([.!?])\s+", r"\1\n\n", text)
    # Remove extra blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Add girlish emojis
    text = sprinkle_girlish_emojis(text)
    # Remove all asterisks from the final response
    text = text.replace('*', '')
    # Remove all occurrences of '\1' from the final response
    text = text.replace('\\1', '')
    text = text.replace('\1', '')
    return text.strip()

def generate_response(user_input: str, session_id: str, user_name: Optional[str] = None):
    """Generate comprehensive AI response with fallback to OpenAI"""
    
    # Get or create session
    session = session_manager.get_session(session_id)
    if user_name:
        session_manager.update_session(session_id, {"user_name": user_name})
    
    # Add user message to history
    session["conversation_history"].append({"role": "user", "content": user_input})
    
    # Detect intent and entities
    intent = intent_detector.detect_intent(user_input)
    entities = entity_extractor.extract_entities(user_input)
    
    # Check if we should use AI for this query (now much more aggressive)
    if should_use_ai(user_input, intent, entities):
        # Get context for AI
        context = f"Intent: {intent}, Entities: {entities}"
        if session.get("current_product"):
            context += f", Current product: {session['current_product']['name']}"
        
        # Get AI response
        ai_response = get_ai_response(user_input, context, session["conversation_history"])
        session["conversation_history"].append({"role": "assistant", "content": ai_response})
        styled_response = style_ai_response(ai_response)
        return {
            "response": styled_response,
            "product": None,
            "suggestions": [],
            "escalate": False,
            "intent": intent,
            "entities": entities,
            "actions": []
        }
    
    # Only handle very specific cases with predefined logic
    if intent == "casual_chat" and len(user_input.split()) <= 3 and any(word in user_input.lower() for word in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]):
        response, product, suggestions, escalate = handle_casual_chat(user_input, session)
    elif intent == "booking" and entities.get("date") and entities.get("time") and entities.get("service_type"):
        response, product, suggestions, escalate = handle_appointment_booking(entities, session)
    elif intent == "product_inquiry" and len(user_input.split()) <= 5 and any(word in user_input.lower() for word in ["sensitive", "normal", "dry", "oily", "combination", "irritation", "redness", "dryness"]):
        response, product, suggestions, escalate = handle_product_inquiry(user_input, session)
    elif intent == "order_status" and entities.get("order_id") and len(user_input.split()) <= 8:
        response, product, suggestions, escalate = handle_order_status(entities, session)
    else:
        # Use AI for everything else (which is almost everything!)
        ai_response = get_ai_response(user_input, f"Intent: {intent}, Entities: {entities}", session["conversation_history"])
        session["conversation_history"].append({"role": "assistant", "content": ai_response})
        response, product, suggestions, escalate = ai_response, None, [], False
    
    # Update session with current product if provided
    if product:
        session_manager.update_session(session_id, {"current_product": product})
    
    # Add response to history
    session["conversation_history"].append({"role": "assistant", "content": response})
    
    # Style the response with girlish emojis before returning
    styled_response = style_ai_response(response)
    
    return {
        "response": styled_response,
        "product": product,
        "suggestions": suggestions,
        "escalate": escalate,
        "intent": intent,
        "entities": entities,
        "actions": []
    }

def handle_appointment_booking(entities: Dict[str, Any], session: Dict[str, Any]):
    """Handle appointment booking requests"""
    # This would integrate with your existing appointment booking logic
    return "I'd be happy to help you book an appointment! What service are you interested in and when would you like to schedule it?", None, [], False

def handle_product_inquiry(user_input: str, session: Dict[str, Any]):
    """Handle product recommendations"""
    
    # Extract skin type and concerns from user input
    skin_types = ["sensitive", "normal", "dry", "oily", "combination"]
    concerns = ["irritation", "redness", "dryness", "sensitivity"]
    
    detected_skin_type = None
    detected_concerns = []
    
    for skin_type in skin_types:
        if skin_type in user_input.lower():
            detected_skin_type = skin_type
            break
    
    for concern in concerns:
        if concern in user_input.lower():
            detected_concerns.append(concern)
    
    # Recommend products
    recommended_products = []
    for category, products in PRODUCTS_DB.items():
        for product in products:
            if detected_skin_type and detected_skin_type in product["skin_type"]:
                recommended_products.append(product)
            elif detected_concerns and any(concern in product["concerns"] for concern in detected_concerns):
                recommended_products.append(product)
    
    if recommended_products:
        product = recommended_products[0]
        return f"Based on your needs, I recommend our {product['name']}. It's perfect for {detected_skin_type or 'your skin type'} and addresses {', '.join(detected_concerns) if detected_concerns else 'your concerns'}.", product, [], False
    else:
        return "I'd be happy to recommend products! Could you tell me about your skin type (sensitive, normal, dry, oily) and any specific concerns you have?", None, [], False

def handle_service_details(entities: Dict[str, Any]):
    """Handle service detail inquiries"""
    return "We offer a variety of waxing services including full body waxing, facial waxing, and specialized treatments. What specific service would you like to know more about?", None, [], False

def handle_order_status(entities: Dict[str, Any], session: Dict[str, Any]):
    """Handle order status inquiries"""
    return "I can help you check your order status. Could you provide your order number or the email address used for the order?", None, [], False

def handle_complaint(user_input: str, session: Dict[str, Any]):
    """Handle complaints and issues"""
    return "I'm sorry to hear you're having an issue. Let me help you resolve this. Could you tell me more about what happened?", None, [], True

def handle_aftercare(entities: Dict[str, Any]):
    """Handle aftercare questions"""
    aftercare_info = KNOWLEDGE_BASE["aftercare"]["waxing"]
    response = "Here are some important aftercare tips:\n\n"
    for tip in aftercare_info:
        response += f"• {tip}\n"
    response += "\nIs there anything specific about aftercare you'd like to know?"
    return response, None, [], False

def handle_casual_chat(user_input: str, session: Dict[str, Any]):
    """Handle casual conversation"""
    user_input_lower = user_input.lower()
    
    # Check if it's a simple greeting
    greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
    if any(greeting in user_input_lower for greeting in greetings):
        responses = [
            "Hello! Welcome to Halawa Wax! How can I help you today?",
            "Hi there! I'm here to help with any questions about our waxing services and products.",
            "Hey! Thanks for reaching out. What can I assist you with today?",
            "Hello! I'm your Halawa Wax assistant. How may I help you?"
        ]
    else:
        # For other casual conversation
        responses = [
            "That's interesting! I'm here to help with any questions about our waxing services and products.",
            "I appreciate you sharing that! Is there anything about our services I can help you with?",
            "That sounds great! While I'm here to help with waxing-related questions, I'm happy to chat briefly.",
            "Thanks for sharing! Have you tried any of our waxing products or services before?"
        ]
    
    return random.choice(responses), None, [], False

def handle_general_questions(user_input: str, session: Dict[str, Any]):
    """Handle general questions"""
    return "That's a great question! I can help you with information about our services, pricing, location, hours, and more. What specifically would you like to know?", None, [], False

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Handle chat messages from the React frontend"""
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())
    
    # Generate AI response
    ai_response = generate_response(request.message, session_id, request.user_name)
    
    # Prepare response
    response_data = {
        "message": ai_response["response"],
        "intent": ai_response.get("intent"),
        "entities": ai_response.get("entities"),
        "actions": ai_response.get("actions", []),
        "escalation_needed": ai_response.get("escalate", False)
    }
    
    if ai_response.get("product"):
        response_data["product"] = ai_response["product"]
    
    return ChatResponse(**response_data)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Halawa Wax AI Backend v2.0 is running"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to Halawa Wax AI Backend v2.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 