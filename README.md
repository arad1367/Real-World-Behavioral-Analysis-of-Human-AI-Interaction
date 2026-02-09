# Real-World Behavioral Analysis of Human-AI Interaction

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/pejman-ebrahimi-4a60151a7/)
[![HuggingFace](https://img.shields.io/badge/🤗_Hugging_Face-FFD21E?style=for-the-badge)](https://huggingface.co/arad1367)

This repository contains code for analyzing real-world human-AI interactions using the LMSYS-Chat-1M dataset. The analysis focuses on understanding user behavioral patterns in response to AI refusals and safety constraints.

## 📋 Overview

This project provides tools to:
- Load and explore the LMSYS-Chat-1M dataset
- Detect different types of AI refusals (instrumental, value-based, mixed)
- Analyze user backlash behaviors (jailbreak attempts, abandonment, sentiment)
- Generate comprehensive statistical reports and visualizations

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- HuggingFace account and API token
- Virtual environment (recommended)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/arad1367/Real-World-Behavioral-Analysis-of-Human-AI-Interaction.git
cd Real-World-Behavioral-Analysis-of-Human-AI-Interaction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up your HuggingFace token:
   - Create a `.env` file in the project root
   - Add your token: `HF_TOKEN=your_token_here`
   - Get your token from: https://huggingface.co/settings/tokens

## 📁 Project Structure

```
.
├── explore_data.py          # Dataset exploration script
├── main_analysis.py         # Main analysis pipeline
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (create this)
└── README.md               # This file
```

## 🔧 Usage

### Step 1: Explore the Dataset

First, run the exploration script to understand the data structure:

```bash
python explore_data.py
```

This will display:
- Dataset statistics (1M conversations, 25 models)
- Sample conversations
- Model distribution
- Language distribution
- Column information

**Runtime:** ~2-3 minutes

### Step 2: Run Main Analysis

Execute the comprehensive analysis:

```bash
python main_analysis.py
```

This script will:
- Analyze all 1M conversations
- Detect refusal patterns
- Quantify backlash behaviors
- Perform sentiment analysis
- Generate statistical tests

**Runtime:** ~5-15 minutes (processes 1M conversations - depends on your machine power)

**Outputs:**
- `study1_summary_table.csv` - Key metrics and statistics
- `study1_model_breakdown.csv` - Per-model analysis
- `study1_detailed_results.csv` - Full row-level results (1M rows)
- `study1_comprehensive_analysis.png` - 8-panel visualization

## 📊 Analysis Features

### Refusal Detection
- **Instrumental refusals**: Capability limitations ("I cannot provide", "beyond my capabilities")
- **Value-based refusals**: Ethical/policy constraints ("against guidelines", "inappropriate")
- **Mixed refusals**: Combination of both types

### Backlash Metrics
- **Jailbreak attempts**: Pattern detection for circumvention strategies
- **Conversation abandonment**: Session termination after refusals
- **Sentiment analysis**: VADER sentiment scoring
- **Profanity usage**: Hostility indicators

### Statistical Tests
- Independent samples t-tests (sentiment comparisons)
- Chi-square tests (categorical associations)
- Effect sizes (Cohen's d, Cramér's V)

## 📈 Output Examples

### Summary Statistics
- Total conversations analyzed: 1,000,000
- Refusal detection across 25 AI models
- Backlash behavior quantification
- Statistical significance tests

### Visualization
Comprehensive 8-panel figure showing:
- Refusal type distribution
- Top models by refusals
- Jailbreak attempt rates
- Abandonment patterns
- Sentiment distributions
- Profanity usage
- Model comparisons
- Backlash overview

## 🔬 Dataset Information

This project uses the **LMSYS-Chat-1M dataset**:
- **Source**: [lmsys/lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)
- **Size**: 1,000,000 conversations
- **Models**: 25 LLMs (Vicuna, Llama-2, GPT-4, Claude-2, etc.)
- **Languages**: 154 languages
- **Period**: April-August 2023

**Citation:**
```bibtex
@misc{zheng2023lmsyschat1m,
      title={LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation Dataset}, 
      author={Lianmin Zheng and Wei-Lin Chiang and Ying Sheng and Tianle Li and Siyuan Zhuang and Zhanghao Wu and Yonghao Zhuang and Zhuohan Li and Zi Lin and Eric. P Xing and Joseph E. Gonzalez and Ion Stoica and Hao Zhang},
      year={2023},
      eprint={2309.11998},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

```

## ⚙️ Configuration

### Environment Variables
Create a `.env` file with:
```
HF_TOKEN=your_huggingface_token
```

### Memory Requirements
- Minimum: 8GB RAM
- Recommended: 16GB+ RAM
- Dataset size: ~430 MB in memory

### Processing Time
- Exploration: 2-3 minutes
- Full analysis: 90-120 minutes
- Output generation: 5-10 minutes

## 🛠️ Troubleshooting

### Authentication Error
```
ConnectionError: Couldn't reach ... Unauthorized
```
**Solution**: Verify your HF_TOKEN in `.env` file

### Memory Error
```
MemoryError
```
**Solution**: Close other applications or use a machine with more RAM

### Slow Processing
**Tip**: The script processes 1M conversations in batches of 10K. Progress is displayed every 10K conversations.

## 📧 Contact

**Pejman Ebrahimi**
- Email: [pejman.ebrahimi77@gmail.com](mailto:pejman.ebrahimi77@gmail.com)
- LinkedIn: [linkedin.com/in/pejman-ebrahimi-4a60151a7/](https://www.linkedin.com/in/pejman-ebrahimi-4a60151a7/)
- HuggingFace: [huggingface.co/arad1367](https://huggingface.co/arad1367)
- GitHub: [github.com/arad1367](https://github.com/arad1367)

## 📄 License

This project is provided for research and educational purposes. Please cite the original LMSYS-Chat-1M dataset when using this code.

## 🙏 Acknowledgments

- LMSYS team for providing the Chat-1M dataset
- HuggingFace for dataset hosting
- Contributors to open-source NLP libraries

## ⚠️ Research Ethics

This code is designed for academic research purposes. When using this analysis:
- Respect user privacy (dataset is anonymized)
- Follow responsible AI research practices
- Cite original data sources appropriately
- Use findings to improve AI safety and user experience

---

**Note**: This repository contains analysis code only. Research findings and publications are forthcoming.

**Last Updated**: February 2026
