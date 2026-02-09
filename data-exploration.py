import os
from datasets import load_dataset
import pandas as pd
import numpy as np
from collections import Counter
import json
from dotenv import load_dotenv

load_dotenv()

token = os.getenv('HF_TOKEN')
if not token:
    raise ValueError("HF_TOKEN not found in environment variables")

print(f"Token loaded: {token[:10]}...")
print("Loading dataset...")

dataset = load_dataset("lmsys/lmsys-chat-1m", token=token)

print(f"\nDataset splits available: {list(dataset.keys())}")

df = pd.DataFrame(dataset['train'])

print("\n" + "="*80)
print("DATASET OVERVIEW")
print("="*80)
print(f"Total conversations: {len(df):,}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n" + "="*80)
print("SAMPLE ROWS (First 5 columns)")
print("="*80)
print(df.head(3))

print("\n" + "="*80)
print("SAMPLE CONVERSATIONS (First 3)")
print("="*80)
for idx in range(min(3, len(df))):
    print(f"\n--- Conversation {idx+1} ---")
    print(f"Columns in this row: {list(df.columns)}")
    for col in df.columns[:5]:
        print(f"{col}: {df.iloc[idx][col]}")

print("\n" + "="*80)
print("CHECKING FOR CONVERSATION COLUMN")
print("="*80)
possible_conv_cols = [col for col in df.columns if 'conv' in col.lower() or 'message' in col.lower() or 'dialog' in col.lower()]
print(f"Possible conversation columns: {possible_conv_cols}")

if possible_conv_cols:
    conv_col = possible_conv_cols[0]
    print(f"\nUsing column: {conv_col}")
    print(f"Sample content:\n{df[conv_col].iloc[0]}")
else:
    print("\nShowing all columns and first row:")
    for col in df.columns:
        print(f"\n{col}:")
        print(df[col].iloc[0])

print("\n" + "="*80)
print("COLUMN VALUE COUNTS (for categorical columns)")
print("="*80)
for col in df.columns:
    if df[col].dtype == 'object' and df[col].nunique() < 100:
        print(f"\n{col}:")
        print(df[col].value_counts().head(10))

print("\n" + "="*80)
print("EXPLORATION COMPLETE")
print("="*80)
print("\nNext: Run optimized Code 2 for the exact data structure - With respect Pejman!")
