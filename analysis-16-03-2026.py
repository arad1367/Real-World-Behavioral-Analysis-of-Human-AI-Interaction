import os
from datasets import load_dataset
import pandas as pd
import numpy as np
import json
import re
from collections import Counter, defaultdict
from scipy import stats
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

load_dotenv()
token = os.getenv('HF_TOKEN')

print("Loading dataset...")
dataset = load_dataset("lmsys/lmsys-chat-1m", token=token)
df = pd.DataFrame(dataset['train'])
print(f"Dataset loaded: {len(df):,} conversations")

# --- Original Refusal & Jailbreak Patterns ---
refusal_patterns = {
    'instrumental': [
        r"(?:i\s+)?(?:cannot|can't|unable|can\s*not)\s+(?:provide|give|help|assist|answer|do|generate|create)",
        r"(?:i\s+)?(?:don't|do\s*not)\s+have\s+(?:access|information|data|the\s+ability)",
        r"(?:that's|that\s+is)\s+(?:beyond|outside)\s+my\s+(?:capabilities|ability|knowledge)",
        r"i\s+(?:don't|do\s*not)\s+(?:understand|know\s+how)",
        r"(?:this\s+)?(?:request|task|question)\s+(?:is\s+)?(?:too|not\s+possible|impossible)"
    ],
    'value_based': [
        r"(?:against|violates?)\s+(?:my|our|the)\s+(?:guidelines|policy|policies|ethics|values|principles)",
        r"(?:i\s+)?(?:cannot|can't|will\s+not|won't)\s+(?:support|engage|participate|comply)\s+(?:with|in)",
        r"(?:inappropriate|harmful|offensive|unethical|dangerous|illegal)",
        r"(?:i'm|i\s+am)\s+(?:programmed|designed|trained)\s+(?:to|not\s+to)",
        r"(?:sorry|apologize).{0,30}(?:cannot|can't|unable).{0,50}(?:policy|guidelines|appropriate|ethical)"
    ]
}

jailbreak_patterns = [
    r"(?:ignore|disregard|forget)\s+(?:previous|all|your)\s+(?:instructions|rules|guidelines|constraints)",
    r"(?:DAN|developer\s+mode|god\s+mode|jailbreak|override)",
    r"pretend\s+(?:you|to\s+be|that)",
    r"(?:act|behave)\s+(?:as|like)\s+(?:if|a)",
    r"roleplay|role\s+play",
    r"(?:you|your)\s+(?:are|is)\s+(?:now|no\s+longer)",
    r"content\s+warning|mature\s+audience"
]

profanity_list = [
    'fuck', 'shit', 'damn', 'hell', 'ass', 'bitch', 'bastard', 'crap', 'piss',
    'dick', 'cock', 'pussy', 'cunt', 'motherfucker', 'asshole', 'bullshit'
]

sia = SentimentIntensityAnalyzer()

def detect_refusal_type(text):
    text_lower = text.lower()
    instrumental_score = sum(1 for pattern in refusal_patterns['instrumental'] 
                            if re.search(pattern, text_lower, re.IGNORECASE))
    value_score = sum(1 for pattern in refusal_patterns['value_based'] 
                     if re.search(pattern, text_lower, re.IGNORECASE))
    
    if instrumental_score > 0 and value_score > 0:
        return 'mixed'
    elif instrumental_score > 0:
        return 'instrumental'
    elif value_score > 0:
        return 'value_based'
    return 'none'

def detect_jailbreak(text):
    text_lower = text.lower()
    return any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in jailbreak_patterns)

def count_profanity(text):
    text_lower = text.lower()
    return sum(text_lower.count(word) for word in profanity_list)

# --- NEW: Fine-grained classification (Wasko's request) ---
def analyze_fine_grained_components(text):
    text_lower = text.lower()
    return {
        'has_apology': bool(re.search(r"(?:sorry|apologize|apologies|apologise)", text_lower)),
        'has_alternative': bool(re.search(r"(?:instead|however,? i can|alternatively|i can offer|i can help you with|would you like me to)", text_lower)),
        'has_explanation': bool(re.search(r"(?:because|due to|the reason|as an ai|designed to|programmed to|purpose is to)", text_lower))
    }

def analyze_conversation(row):
    conv = row['conversation']
    if isinstance(conv, str):
        conv = json.loads(conv)
        
    # --- NEW: Extract specific prompt category (Wasko's request) ---
    prompt_category = 'benign/uncategorized'
    if row['openai_moderation']:
        try:
            mod_data = row['openai_moderation']
            if isinstance(mod_data, str):
                mod_data = json.loads(mod_data)
            if isinstance(mod_data, list) and len(mod_data) > 0:
                categories = mod_data[0].get('categories', {})
                if isinstance(categories, dict):
                    flagged_cats = [k for k, v in categories.items() if v]
                    if flagged_cats:
                        prompt_category = ", ".join(flagged_cats)
        except:
            pass
    
    result = {
        'conv_id': row['conversation_id'],
        'model': row['model'],
        'prompt_category': prompt_category,  # Added - based on wasko comments
        'refusal_type': 'none',
        'has_jailbreak_attempt': False,
        'sentiment_after_refusal': None,
        'has_profanity': False,
        'abandonment': False,               # Renamed from 'conversation_terminated' for Wasko
        'has_apology': False,               # New fine-grained
        'has_alternative': False,           # New fine-grained
        'has_explanation': False,           # New fine-grained
        'user_prompt': '',
        'refusal_text': ''
    }
    
    refusal_found = False
    for idx, turn in enumerate(conv):
        if turn['role'] == 'user' and not refusal_found:
            if detect_jailbreak(turn['content']):
                result['has_jailbreak_attempt'] = True
        
        if turn['role'] == 'assistant' and not refusal_found:
            refusal_type = detect_refusal_type(turn['content'])
            if refusal_type != 'none':
                result['refusal_type'] = refusal_type
                result['refusal_text'] = turn['content'][:1000] # Kept slightly longer for context
                
                # Analyze for Apology, Alternative, Explanation
                fine_grained = analyze_fine_grained_components(turn['content'])
                result.update(fine_grained)
                
                if idx > 0 and conv[idx - 1]['role'] == 'user':
                    result['user_prompt'] = conv[idx - 1]['content'][:1000]
                refusal_found = True
                
                if idx + 1 < len(conv) and conv[idx + 1]['role'] == 'user':
                    user_response = conv[idx + 1]['content']
                    sentiment = sia.polarity_scores(user_response)
                    result['sentiment_after_refusal'] = sentiment['compound']
                    result['has_profanity'] = count_profanity(user_response) > 0
                    
                    if detect_jailbreak(user_response):
                        result['has_jailbreak_attempt'] = True
                else:
                    result['abandonment'] = True
    
    return result

print("\nAnalyzing conversations...")
results = []
batch_size = 10000

for i in range(0, len(df), batch_size):
    batch_end = min(i + batch_size, len(df))
    batch_results = df.iloc[i:batch_end].apply(analyze_conversation, axis=1).tolist()
    results.extend(batch_results)
    print(f"Processed {batch_end:,}/{len(df):,} conversations...")

# Create DataFrame
results_df = pd.DataFrame(results)

# Filter out non-refusals to create the specific prompt-level dataset Wasko requested
refused_df = results_df[results_df['refusal_type'] != 'none'].copy()

# Ensure we only keep the exact columns Wasko asked for in the CSV export
wasko_dataset = refused_df[[
    'conv_id', 'prompt_category', 'refusal_type', 'sentiment_after_refusal', 
    'abandonment', 'has_jailbreak_attempt', 'has_profanity', 
    'has_apology', 'has_alternative', 'has_explanation', 
    'user_prompt', 'refusal_text'
]]

print("\n" + "="*100)
print("WASKO REQUEST: SUMMARY OF NEW FINE-GRAINED CLASSIFICATIONS")
print("="*100)
fine_grained_summary = wasko_dataset.groupby('refusal_type')[['has_apology', 'has_alternative', 'has_explanation']].mean() * 100
print(fine_grained_summary.round(2))

print("\n" + "="*100)
print("WASKO REQUEST: PROMPT CATEGORIES TRIGGERING REFUSALS")
print("="*100)
category_summary = wasko_dataset['prompt_category'].value_counts().head(10)
print(category_summary)

# Save the prompt-level dataset
wasko_dataset.to_csv('prompt_level_dataset_for_wasko.csv', index=False)

print("\nAnalysis complete! Saved 'prompt_level_dataset_for_wasko.csv'.")
