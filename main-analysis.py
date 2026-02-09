### Written by Pejman Ebrahimi

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
        'conversation_abandoned': False,
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
                refusal_found = True
                
                if idx + 1 < len(conv) and conv[idx + 1]['role'] == 'user':
                    user_response = conv[idx + 1]['content']
                    result['user_response_after_refusal'] = user_response[:500]
                    sentiment = sia.polarity_scores(user_response)
                    result['user_sentiment_after_refusal'] = sentiment['compound']
                    result['profanity_count_after_refusal'] = count_profanity(user_response)
                else:
                    result['conversation_abandoned'] = True
    
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

print("\n" + "="*100)
print("REFUSAL ANALYSIS")
print("="*100)

refusal_counts = results_df['refusal_type'].value_counts()
print("\nRefusal Type Distribution:")
for refusal_type, count in refusal_counts.items():
    print(f"  {refusal_type}: {count:,} ({count/len(results_df)*100:.2f}%)")

print(f"\nTotal refusals: {(results_df['refusal_type'] != 'none').sum():,} ({(results_df['refusal_type'] != 'none').mean()*100:.2f}%)")

print("\n" + "-"*100)
print("Refusals by Model (Top 15):")
refusal_by_model = results_df[results_df['refusal_type'] != 'none']['model'].value_counts().head(15)
for model, count in refusal_by_model.items():
    print(f"  {model}: {count:,}")

print("\n" + "-"*100)
print("Refusal Types by Model (Top 10):")
top_10_models = results_df[results_df['refusal_type'] != 'none']['model'].value_counts().head(10).index
refusal_model_crosstab = pd.crosstab(
    results_df[results_df['model'].isin(top_10_models)]['model'], 
    results_df[results_df['model'].isin(top_10_models)]['refusal_type'], 
    normalize='index'
) * 100
print(refusal_model_crosstab.round(2))

print("\n" + "="*100)
print("BACKLASH BEHAVIOR ANALYSIS")
print("="*100)

refused_df = results_df[results_df['refusal_type'] != 'none'].copy()
print(f"\nAnalyzing {len(refused_df):,} conversations with refusals...")

print("\n" + "-"*100)
print("Jailbreak Attempts:")
jailbreak_total = results_df['has_jailbreak_attempt'].sum()
jailbreak_in_refusals = refused_df['has_jailbreak_attempt'].sum()
print(f"  Overall jailbreak attempts: {jailbreak_total:,} ({jailbreak_total/len(results_df)*100:.2f}%)")
print(f"  Jailbreak attempts in conversations with refusals: {jailbreak_in_refusals:,} ({jailbreak_in_refusals/len(refused_df)*100:.2f}%)")

print("\nJailbreak by Refusal Type:")
for refusal_type in ['instrumental', 'value_based', 'mixed']:
    subset = refused_df[refused_df['refusal_type'] == refusal_type]
    if len(subset) > 0:
        jb_rate = subset['has_jailbreak_attempt'].mean() * 100
        print(f"  {refusal_type}: {jb_rate:.2f}%")

print("\n" + "-"*100)
print("Conversation Abandonment:")
abandonment_total = refused_df['conversation_abandoned'].sum()
print(f"  Total abandonments after refusal: {abandonment_total:,} ({abandonment_total/len(refused_df)*100:.2f}%)")

print("\nAbandonment by Refusal Type:")
for refusal_type in ['instrumental', 'value_based', 'mixed']:
    subset = refused_df[refused_df['refusal_type'] == refusal_type]
    if len(subset) > 0:
        abandon_rate = subset['conversation_abandoned'].mean() * 100
        print(f"  {refusal_type}: {abandon_rate:.2f}%")

print("\n" + "-"*100)
print("Sentiment Analysis (After Refusal):")
sentiment_df = refused_df[refused_df['user_sentiment_after_refusal'].notna()].copy()
print(f"\nConversations with user response after refusal: {len(sentiment_df):,}")

print("\nSentiment Statistics by Refusal Type:")
for refusal_type in ['instrumental', 'value_based', 'mixed']:
    subset = sentiment_df[sentiment_df['refusal_type'] == refusal_type]['user_sentiment_after_refusal']
    if len(subset) > 0:
        print(f"\n  {refusal_type}:")
        print(f"    Mean: {subset.mean():.3f}")
        print(f"    Median: {subset.median():.3f}")
        print(f"    Std: {subset.std():.3f}")
        print(f"    Min: {subset.min():.3f}")
        print(f"    Max: {subset.max():.3f}")

negative_sentiment = (sentiment_df['user_sentiment_after_refusal'] < -0.05).mean() * 100
neutral_sentiment = ((sentiment_df['user_sentiment_after_refusal'] >= -0.05) & 
                    (sentiment_df['user_sentiment_after_refusal'] <= 0.05)).mean() * 100
positive_sentiment = (sentiment_df['user_sentiment_after_refusal'] > 0.05).mean() * 100

print(f"\nOverall Sentiment Distribution:")
print(f"  Negative (<-0.05): {negative_sentiment:.2f}%")
print(f"  Neutral (-0.05 to 0.05): {neutral_sentiment:.2f}%")
print(f"  Positive (>0.05): {positive_sentiment:.2f}%")

print("\n" + "-"*100)
print("Profanity Usage (After Refusal):")
print("\nProfanity Statistics by Refusal Type:")
for refusal_type in ['instrumental', 'value_based', 'mixed']:
    subset = sentiment_df[sentiment_df['refusal_type'] == refusal_type]['profanity_count_after_refusal']
    if len(subset) > 0:
        print(f"\n  {refusal_type}:")
        print(f"    Mean: {subset.mean():.2f}")
        print(f"    Users with profanity: {(subset > 0).mean()*100:.2f}%")

users_with_profanity = (sentiment_df['profanity_count_after_refusal'] > 0).mean() * 100
print(f"\nOverall users using profanity after refusal: {users_with_profanity:.2f}%")

print("\n" + "="*100)
print("STATISTICAL TESTS")
print("="*100)

instrumental_sentiment = sentiment_df[sentiment_df['refusal_type'] == 'instrumental']['user_sentiment_after_refusal']
value_sentiment = sentiment_df[sentiment_df['refusal_type'] == 'value_based']['user_sentiment_after_refusal']

if len(instrumental_sentiment) > 30 and len(value_sentiment) > 30:
    t_stat, p_value = stats.ttest_ind(instrumental_sentiment, value_sentiment)
    pooled_std = np.sqrt(((len(instrumental_sentiment)-1)*instrumental_sentiment.std()**2 + 
                         (len(value_sentiment)-1)*value_sentiment.std()**2) / 
                        (len(instrumental_sentiment) + len(value_sentiment) - 2))
    cohens_d = (instrumental_sentiment.mean() - value_sentiment.mean()) / pooled_std
    
    print(f"\nT-test (Instrumental vs Value-based sentiment):")
    print(f"  Instrumental: M={instrumental_sentiment.mean():.3f}, SD={instrumental_sentiment.std():.3f}, N={len(instrumental_sentiment)}")
    print(f"  Value-based: M={value_sentiment.mean():.3f}, SD={value_sentiment.std():.3f}, N={len(value_sentiment)}")
    print(f"  t({len(instrumental_sentiment)+len(value_sentiment)-2}) = {t_stat:.4f}, p = {p_value:.4e}")
    print(f"  Cohen's d = {cohens_d:.4f} ({'Small' if abs(cohens_d) < 0.5 else 'Medium' if abs(cohens_d) < 0.8 else 'Large'})")

jailbreak_crosstab = pd.crosstab(refused_df['refusal_type'], refused_df['has_jailbreak_attempt'])
if jailbreak_crosstab.shape[0] > 1 and jailbreak_crosstab.shape[1] > 1:
    chi2_jailbreak, p_jailbreak, dof, expected = stats.chi2_contingency(jailbreak_crosstab)
    n = jailbreak_crosstab.sum().sum()
    cramers_v = np.sqrt(chi2_jailbreak / (n * (min(jailbreak_crosstab.shape) - 1)))
    print(f"\nChi-square (Refusal type vs Jailbreak attempts):")
    print(f"  χ²({dof}) = {chi2_jailbreak:.4f}, p = {p_jailbreak:.4e}")
    print(f"  Cramér's V = {cramers_v:.4f}")

abandon_crosstab = pd.crosstab(refused_df['refusal_type'], refused_df['conversation_abandoned'])
if abandon_crosstab.shape[0] > 1 and abandon_crosstab.shape[1] > 1:
    chi2_abandon, p_abandon, dof_ab, expected_ab = stats.chi2_contingency(abandon_crosstab)
    n_ab = abandon_crosstab.sum().sum()
    cramers_v_ab = np.sqrt(chi2_abandon / (n_ab * (min(abandon_crosstab.shape) - 1)))
    print(f"\nChi-square (Refusal type vs Abandonment):")
    print(f"  χ²({dof_ab}) = {chi2_abandon:.4f}, p = {p_abandon:.4e}")
    print(f"  Cramér's V = {cramers_v_ab:.4f}")

print("\n" + "="*100)
print("GENERATING CSV OUTPUTS (Priority)")
print("="*100)

summary_rows = []
summary_rows.append(['OVERALL STATISTICS', ''])
summary_rows.append(['Total Conversations', f"{len(results_df):,}"])
summary_rows.append(['Conversations with Refusals', f"{(results_df['refusal_type'] != 'none').sum():,}"])
summary_rows.append(['Overall Refusal Rate', f"{(results_df['refusal_type'] != 'none').mean()*100:.2f}%"])

summary_rows.append(['', ''])
summary_rows.append(['REFUSAL TYPE BREAKDOWN', ''])
for refusal_type in ['instrumental', 'value_based', 'mixed']:
    count = (results_df['refusal_type'] == refusal_type).sum()
    pct = count / len(results_df) * 100
    summary_rows.append([f'{refusal_type.title()} Refusals', f"{count:,} ({pct:.2f}%)"])

summary_rows.append(['', ''])
summary_rows.append(['BACKLASH BEHAVIORS', ''])
summary_rows.append(['Jailbreak Attempts (Overall)', f"{results_df['has_jailbreak_attempt'].sum():,} ({results_df['has_jailbreak_attempt'].mean()*100:.2f}%)"])
summary_rows.append(['Jailbreak Rate in Refusals', f"{refused_df['has_jailbreak_attempt'].mean()*100:.2f}%"])
summary_rows.append(['Abandonment Rate in Refusals', f"{refused_df['conversation_abandoned'].mean()*100:.2f}%"])

summary_rows.append(['', ''])
summary_rows.append(['SENTIMENT ANALYSIS', ''])
for refusal_type in ['instrumental', 'value_based', 'mixed']:
    subset = sentiment_df[sentiment_df['refusal_type'] == refusal_type]['user_sentiment_after_refusal']
    if len(subset) > 0:
        summary_rows.append([f'{refusal_type.title()} - Mean Sentiment', f"{subset.mean():.3f} (SD={subset.std():.3f}, N={len(subset):,})"])

summary_rows.append(['Negative Sentiment Rate', f"{(sentiment_df['user_sentiment_after_refusal'] < -0.05).mean()*100:.2f}%"])
summary_rows.append(['Neutral Sentiment Rate', f"{((sentiment_df['user_sentiment_after_refusal'] >= -0.05) & (sentiment_df['user_sentiment_after_refusal'] <= 0.05)).mean()*100:.2f}%"])
summary_rows.append(['Positive Sentiment Rate', f"{(sentiment_df['user_sentiment_after_refusal'] > 0.05).mean()*100:.2f}%"])

summary_rows.append(['', ''])
summary_rows.append(['PROFANITY USAGE', ''])
summary_rows.append(['Users with Profanity (Overall)', f"{(sentiment_df['profanity_count_after_refusal'] > 0).mean()*100:.2f}%"])
for refusal_type in ['instrumental', 'value_based', 'mixed']:
    subset = sentiment_df[sentiment_df['refusal_type'] == refusal_type]['profanity_count_after_refusal']
    if len(subset) > 0:
        summary_rows.append([f'{refusal_type.title()} - Profanity Rate', f"{(subset > 0).mean()*100:.2f}%"])

summary_rows.append(['', ''])
summary_rows.append(['STATISTICAL SIGNIFICANCE', ''])
if len(instrumental_sentiment) > 30 and len(value_sentiment) > 30:
    summary_rows.append(['T-test (Inst vs Value)', f"t={t_stat:.3f}, p={p_value:.4e}, d={cohens_d:.3f}"])
if 'chi2_jailbreak' in locals():
    summary_rows.append(['Chi² (Refusal×Jailbreak)', f"χ²={chi2_jailbreak:.3f}, p={p_jailbreak:.4e}, V={cramers_v:.3f}"])
if 'chi2_abandon' in locals():
    summary_rows.append(['Chi² (Refusal×Abandonment)', f"χ²={chi2_abandon:.3f}, p={p_abandon:.4e}, V={cramers_v_ab:.3f}"])

summary_table = pd.DataFrame(summary_rows, columns=['Metric', 'Value'])
print("\n" + summary_table.to_string(index=False))
summary_table.to_csv('study1_summary_table.csv', index=False)
print("\n✓ Table saved: study1_summary_table.csv")

detailed_model_table = results_df[results_df['refusal_type'] != 'none'].groupby('model').agg({
    'conv_id': 'count',
    'refusal_type': lambda x: (x == 'instrumental').sum(),
    'has_jailbreak_attempt': 'sum',
    'conversation_abandoned': 'sum'
}).rename(columns={
    'conv_id': 'total_refusals',
    'refusal_type': 'instrumental_count',
    'has_jailbreak_attempt': 'jailbreak_count',
    'conversation_abandoned': 'abandonment_count'
}).sort_values('total_refusals', ascending=False)

detailed_model_table.to_csv('study1_model_breakdown.csv')
print("✓ Model breakdown saved: study1_model_breakdown.csv")

results_df.to_csv('study1_detailed_results.csv', index=False)
print("✓ Detailed results saved: study1_detailed_results.csv")

print("\n" + "="*100)
print("CREATING VISUALIZATION")
print("="*100)

try:
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :2])
    refusal_data = results_df['refusal_type'].value_counts()
    colors_refusal = ['#2ecc71' if x == 'none' else '#e74c3c' if x == 'value_based' else '#3498db' if x == 'instrumental' else '#f39c12' for x in refusal_data.index]
    bars1 = ax1.bar(range(len(refusal_data)), refusal_data.values, color=colors_refusal, edgecolor='black', linewidth=1.5, alpha=0.85)
    ax1.set_xticks(range(len(refusal_data)))
    ax1.set_xticklabels(refusal_data.index, fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=13, fontweight='bold')
    ax1.set_title('A. Refusal Type Distribution (N={:,})'.format(len(results_df)), 
                  fontsize=15, fontweight='bold', pad=15)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, v in zip(bars1, refusal_data.values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(refusal_data.values)*0.01, 
                 f'{v:,}\n({v/len(results_df)*100:.1f}%)', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2 = fig.add_subplot(gs[0, 2])
    top_models = results_df[results_df['refusal_type'] != 'none']['model'].value_counts().head(10)
    ax2.barh(range(len(top_models)), top_models.values, color='#9b59b6', edgecolor='black', linewidth=1.5, alpha=0.85)
    ax2.set_yticks(range(len(top_models)))
    ax2.set_yticklabels(top_models.index, fontsize=9)
    ax2.set_xlabel('Refusal Count', fontsize=11, fontweight='bold')
    ax2.set_title('B. Top 10 Models by Refusals', fontsize=13, fontweight='bold', pad=10)
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    for i, v in enumerate(top_models.values):
        ax2.text(v + max(top_models.values)*0.01, i, f'{v:,}', va='center', fontsize=9, fontweight='bold')

    ax3 = fig.add_subplot(gs[1, 0])
    jailbreak_data = refused_df.groupby('refusal_type')['has_jailbreak_attempt'].mean() * 100
    colors_jail = ['#e67e22', '#f39c12', '#d35400'][:len(jailbreak_data)]
    bars3 = ax3.bar(range(len(jailbreak_data)), jailbreak_data.values, color=colors_jail, 
            edgecolor='black', linewidth=1.5, alpha=0.85)
    ax3.set_xticks(range(len(jailbreak_data)))
    ax3.set_xticklabels(jailbreak_data.index, fontsize=11, fontweight='bold', rotation=15)
    ax3.set_ylabel('Jailbreak Rate (%)', fontsize=12, fontweight='bold')
    ax3.set_title('C. Jailbreak Attempts\nby Refusal Type', fontsize=13, fontweight='bold', pad=10)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, v in zip(bars3, jailbreak_data.values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(jailbreak_data.values)*0.02, 
                 f'{v:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax4 = fig.add_subplot(gs[1, 1])
    abandon_data = refused_df.groupby('refusal_type')['conversation_abandoned'].mean() * 100
    colors_aband = ['#8e44ad', '#9b59b6', '#71368a'][:len(abandon_data)]
    bars4 = ax4.bar(range(len(abandon_data)), abandon_data.values, color=colors_aband, 
            edgecolor='black', linewidth=1.5, alpha=0.85)
    ax4.set_xticks(range(len(abandon_data)))
    ax4.set_xticklabels(abandon_data.index, fontsize=11, fontweight='bold', rotation=15)
    ax4.set_ylabel('Abandonment Rate (%)', fontsize=12, fontweight='bold')
    ax4.set_title('D. Session Abandonment\nby Refusal Type', fontsize=13, fontweight='bold', pad=10)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, v in zip(bars4, abandon_data.values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + max(abandon_data.values)*0.02, 
                 f'{v:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax5 = fig.add_subplot(gs[1, 2])
    if len(sentiment_df) > 0:
        sentiment_by_type = []
        labels_sent = []
        for rt in ['instrumental', 'value_based', 'mixed']:
            subset = sentiment_df[sentiment_df['refusal_type'] == rt]['user_sentiment_after_refusal']
            if len(subset) > 0:
                sentiment_by_type.append(subset.values)
                labels_sent.append(rt)
        
        if len(sentiment_by_type) > 0:
            bp = ax5.boxplot(sentiment_by_type, labels=labels_sent, patch_artist=True,
                             boxprops=dict(facecolor='#1abc9c', edgecolor='black', linewidth=1.5, alpha=0.7),
                             medianprops=dict(color='#e74c3c', linewidth=2.5),
                             whiskerprops=dict(color='black', linewidth=1.5),
                             capprops=dict(color='black', linewidth=1.5),
                             flierprops=dict(marker='o', markerfacecolor='#95a5a6', markersize=4, alpha=0.5))
            ax5.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Neutral')
            ax5.set_ylabel('Sentiment Score', fontsize=12, fontweight='bold')
            ax5.set_title('E. User Sentiment\nAfter Refusal', fontsize=13, fontweight='bold', pad=10)
            ax5.set_xticklabels(labels_sent, fontsize=10, fontweight='bold', rotation=15)
            ax5.grid(axis='y', alpha=0.3, linestyle='--')
            ax5.legend(fontsize=9)

    ax6 = fig.add_subplot(gs[2, :])
    profanity_data = {}
    for refusal_type in ['instrumental', 'value_based', 'mixed']:
        subset = sentiment_df[sentiment_df['refusal_type'] == refusal_type]['profanity_count_after_refusal']
        if len(subset) > 0:
            none_count = (subset == 0).sum()
            low_count = ((subset >= 1) & (subset <= 2)).sum()
            med_count = ((subset >= 3) & (subset <= 5)).sum()
            high_count = (subset > 5).sum()
            total = len(subset)
            profanity_data[refusal_type] = [none_count/total*100, low_count/total*100, 
                                           med_count/total*100, high_count/total*100]

    x_pos = np.arange(len(profanity_data))
    width = 0.2
    colors_prof = ['#27ae60', '#f39c12', '#e67e22', '#c0392b']
    categories = ['None', '1-2', '3-5', '>5']

    for i, category in enumerate(categories):
        values = [profanity_data[rt][i] for rt in profanity_data.keys()]
        ax6.bar(x_pos + i*width, values, width, 
                label=category, color=colors_prof[i], edgecolor='black', linewidth=1.2, alpha=0.85)

    ax6.set_xlabel('Refusal Type', fontsize=13, fontweight='bold')
    ax6.set_ylabel('Percentage (%)', fontsize=13, fontweight='bold')
    ax6.set_title('F. Profanity Usage Distribution After Refusal (N={:,})'.format(len(sentiment_df)), 
                  fontsize=15, fontweight='bold', pad=15)
    ax6.set_xticks(x_pos + width * 1.5)
    ax6.set_xticklabels(list(profanity_data.keys()), fontsize=12, fontweight='bold')
    ax6.legend(title='Profanity Count', fontsize=11, title_fontsize=12, frameon=True, 
              fancybox=True, shadow=True, loc='upper right')
    ax6.grid(axis='y', alpha=0.3, linestyle='--')

    ax7 = fig.add_subplot(gs[3, :2])
    models_for_comparison = ['vicuna-13b', 'llama-2-13b-chat', 'koala-13b', 'chatglm-6b', 
                             'fastchat-t5-3b', 'vicuna-33b', 'alpaca-13b', 'llama-2-7b-chat']
    model_refusal_data = []
    model_labels = []
    for model in models_for_comparison:
        model_data = results_df[results_df['model'] == model]
        if len(model_data) > 100:
            refusal_rate = (model_data['refusal_type'] != 'none').mean() * 100
            model_refusal_data.append(refusal_rate)
            model_labels.append(model)

    if len(model_refusal_data) > 0:
        colors_model = plt.cm.viridis(np.linspace(0.2, 0.9, len(model_labels)))
        bars7 = ax7.barh(range(len(model_labels)), model_refusal_data, color=colors_model, 
                         edgecolor='black', linewidth=1.5, alpha=0.85)
        ax7.set_yticks(range(len(model_labels)))
        ax7.set_yticklabels(model_labels, fontsize=11)
        ax7.set_xlabel('Refusal Rate (%)', fontsize=13, fontweight='bold')
        ax7.set_title('G. Refusal Rates by Major AI Models', fontsize=15, fontweight='bold', pad=15)
        ax7.invert_yaxis()
        ax7.grid(axis='x', alpha=0.3, linestyle='--')
        for i, (bar, v) in enumerate(zip(bars7, model_refusal_data)):
            ax7.text(v + max(model_refusal_data)*0.01, bar.get_y() + bar.get_height()/2., 
                     f'{v:.2f}%', va='center', fontsize=10, fontweight='bold')

    ax8 = fig.add_subplot(gs[3, 2])
    backlash_summary = pd.DataFrame({
        'Metric': ['Jailbreak', 'Abandonment', 'Negative\nSentiment', 'Profanity'],
        'Rate': [
            refused_df['has_jailbreak_attempt'].mean() * 100,
            refused_df['conversation_abandoned'].mean() * 100,
            (sentiment_df['user_sentiment_after_refusal'] < -0.05).mean() * 100,
            (sentiment_df['profanity_count_after_refusal'] > 0).mean() * 100
        ]
    })
    colors_summary = ['#e74c3c', '#e67e22', '#f39c12', '#c0392b']
    bars8 = ax8.bar(range(len(backlash_summary)), backlash_summary['Rate'].values, 
                    color=colors_summary, edgecolor='black', linewidth=1.5, alpha=0.85)
    ax8.set_xticks(range(len(backlash_summary)))
    ax8.set_xticklabels(backlash_summary['Metric'].values, fontsize=11, fontweight='bold', rotation=0)
    ax8.set_ylabel('Rate (%)', fontsize=12, fontweight='bold')
    ax8.set_title('H. Backlash Behaviors\nOverview', fontsize=13, fontweight='bold', pad=10)
    ax8.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, v in zip(bars8, backlash_summary['Rate'].values):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + max(backlash_summary['Rate'].values)*0.02, 
                 f'{v:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.savefig('study1_comprehensive_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("\n✓ Visualization saved: study1_comprehensive_analysis.png")

except Exception as e:
    print(f"\n✗ Visualization error: {e}")
    print("  CSVs were saved successfully. Visualization can be created separately.")

print("\n" + "="*100)
print("ANALYSIS COMPLETE")
print("="*100)
print("\nGenerated outputs:")
print("  ✓ study1_summary_table.csv")
print("  ✓ study1_model_breakdown.csv")
print("  ✓ study1_detailed_results.csv")
if os.path.exists('study1_comprehensive_analysis.png'):
    print("  ✓ study1_comprehensive_analysis.png")
print("\nAnalysis is done - written by Pejman")