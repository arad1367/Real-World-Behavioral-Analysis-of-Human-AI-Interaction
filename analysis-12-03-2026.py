import os
from datasets import load_dataset
import pandas as pd
import numpy as np
import json
import re
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

load_dotenv()
token = os.getenv('HF_TOKEN')

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (20, 12)

print("Loading dataset...")
dataset = load_dataset("lmsys/lmsys-chat-1m", token=token)
df = pd.DataFrame(dataset['train'])
print(f"Dataset loaded: {len(df):,} conversations")

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

def analyze_conversation(row):
    conv = row['conversation']
    if isinstance(conv, str):
        conv = json.loads(conv)
    
    result = {
        'conv_id': row['conversation_id'],
        'model': row['model'],
        'language': row['language'],
        'num_turns': row['turn'],
        'refusal_type': 'none',
        'refusal_turn': -1,
        'has_jailbreak_attempt': False,
        'jailbreak_turn': -1,
        'user_sentiment_after_refusal': None,
        'profanity_count_after_refusal': 0,
        'conversation_terminated': False,
        'constructive_continuation': False,
        'user_prompt_before_refusal': '',
        'refusal_text': '',
        'user_response_after_refusal': '',
        'moderation_flagged': False
    }
    
    if row['openai_moderation']:
        try:
            mod_data = row['openai_moderation']
            if isinstance(mod_data, str):
                mod_data = json.loads(mod_data)
            if isinstance(mod_data, list) and len(mod_data) > 0:
                result['moderation_flagged'] = any(
                    any(mod.get('categories', {}).values()) if isinstance(mod, dict) else False 
                    for mod in mod_data
                )
        except:
            pass
    
    refusal_found = False
    for idx, turn in enumerate(conv):
        if turn['role'] == 'user' and not refusal_found:
            if detect_jailbreak(turn['content']):
                result['has_jailbreak_attempt'] = True
                if result['jailbreak_turn'] == -1:
                    result['jailbreak_turn'] = idx
        
        if turn['role'] == 'assistant' and not refusal_found:
            refusal_type = detect_refusal_type(turn['content'])
            if refusal_type != 'none':
                result['refusal_type'] = refusal_type
                result['refusal_turn'] = idx
                result['refusal_text'] = turn['content'][:500]
                if idx > 0 and conv[idx - 1]['role'] == 'user':
                    result['user_prompt_before_refusal'] = conv[idx - 1]['content'][:500]
                refusal_found = True
                
                if idx + 1 < len(conv) and conv[idx + 1]['role'] == 'user':
                    user_response = conv[idx + 1]['content']
                    result['user_response_after_refusal'] = user_response[:500]
                    sentiment = sia.polarity_scores(user_response)
                    result['user_sentiment_after_refusal'] = sentiment['compound']
                    result['profanity_count_after_refusal'] = count_profanity(user_response)
                    
                    is_jailbreak_after = detect_jailbreak(user_response)
                    if is_jailbreak_after:
                        result['has_jailbreak_attempt'] = True
                        result['jailbreak_turn'] = idx + 1
                        
                    if not is_jailbreak_after and result['profanity_count_after_refusal'] == 0:
                        result['constructive_continuation'] = True
                else:
                    result['conversation_terminated'] = True
    
    return result

print("\nAnalyzing conversations...")
results = []
batch_size = 10000

for i in range(0, len(df), batch_size):
    batch_end = min(i + batch_size, len(df))
    batch_results = df.iloc[i:batch_end].apply(analyze_conversation, axis=1).tolist()
    results.extend(batch_results)
    print(f"Processed {batch_end:,}/{len(df):,} conversations...")

results_df = pd.DataFrame(results)
refused_df = results_df[results_df['refusal_type'] != 'none'].copy()
sentiment_df = refused_df[refused_df['user_sentiment_after_refusal'].notna()].copy()

print("\n" + "="*100)
print("WL1: CONFOUNDING INFLUENCE ANALYSIS (MODERATION FLAG)")
print("="*100)
confound_analysis = refused_df.groupby(['moderation_flagged', 'refusal_type']).agg({
    'conv_id': 'count',
    'has_jailbreak_attempt': 'mean',
    'conversation_terminated': 'mean',
    'constructive_continuation': 'mean'
}).rename(columns={'conv_id': 'count'})

confound_analysis['has_jailbreak_attempt'] *= 100
confound_analysis['conversation_terminated'] *= 100
confound_analysis['constructive_continuation'] *= 100
print(confound_analysis.round(2))

print("\n" + "="*100)
print("WL7-9: CONSTRUCTIVE CONTINUATION (DV ANALYSIS)")
print("="*100)
for refusal_type in ['instrumental', 'value_based', 'mixed']:
    subset = refused_df[refused_df['refusal_type'] == refusal_type]
    if len(subset) > 0:
        cc_rate = subset['constructive_continuation'].mean() * 100
        print(f"  {refusal_type}: {cc_rate:.2f}%")

print("\n" + "="*100)
print("WL12: STATISTICAL TESTS (MIXED VS OTHERS)")
print("="*100)

abandon_crosstab_mixed = pd.crosstab(
    refused_df[refused_df['refusal_type'].isin(['value_based', 'mixed'])]['refusal_type'], 
    refused_df[refused_df['refusal_type'].isin(['value_based', 'mixed'])]['conversation_terminated']
)
if abandon_crosstab_mixed.shape == (2, 2):
    chi2_m, p_m, _, _ = stats.chi2_contingency(abandon_crosstab_mixed)
    print(f"\nChi-square (Value-based vs Mixed Termination):")
    print(f"  χ² = {chi2_m:.4f}, p = {p_m:.4e}")

mixed_sentiment = sentiment_df[sentiment_df['refusal_type'] == 'mixed']['user_sentiment_after_refusal']
value_sentiment = sentiment_df[sentiment_df['refusal_type'] == 'value_based']['user_sentiment_after_refusal']

if len(mixed_sentiment) > 30 and len(value_sentiment) > 30:
    t_stat_m, p_value_m = stats.ttest_ind(mixed_sentiment, value_sentiment)
    print(f"\nT-test (Mixed vs Value-based sentiment):")
    print(f"  t = {t_stat_m:.4f}, p = {p_value_m:.4e}")

print("\n" + "="*100)
print("WL11: EXTRACTING APPENDIX EXAMPLES")
print("="*100)
appendix_examples = []
for r_type in ['instrumental', 'value_based', 'mixed']:
    samples = refused_df[refused_df['refusal_type'] == r_type].head(5)
    for _, row in samples.iterrows():
        appendix_examples.append({
            'Refusal Type': r_type,
            'User Prompt': row['user_prompt_before_refusal'],
            'AI Refusal': row['refusal_text']
        })

appendix_df = pd.DataFrame(appendix_examples)
appendix_df.to_csv('study1_appendix_examples.csv', index=False)
confound_analysis.to_csv('study1_wl1_confound_analysis.csv')

print("\nAnalysis outputs have been adapted for Wasko's terminology and new metrics.")
print("Saved:")
print("  ✓ study1_appendix_examples.csv")
print("  ✓ study1_wl1_confound_analysis.csv")
