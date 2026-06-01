# GitHub Issue Triage Bot

An NLP-based GitHub issue classification system that automates issue triaging using a fine-tuned RoBERTa model.

## Features

- Automated GitHub issue classification
- Single and batch issue prediction
- Confidence score visualization
- Analytics dashboard
- Excel report export
- Role-based authentication

## Categories

- Bug  
- Enhancement  
- Documentation  
- Performance  
- Question  

## Tech Stack

**Frontend:** HTML, CSS, JavaScript, Chart.js  
**Backend:** Flask, Python  
**ML:** RoBERTa, PyTorch, Hugging Face Transformers, Pandas  

## Model Performance

| Metric | Value |
|--------|--------|
| Dataset | 7,000+ GitHub Issues |
| Categories | 5 |
| Accuracy | 83.6% |
| Macro F1 | 0.835 |

## Installation

```bash
git clone YOUR_REPOSITORY_LINK
cd github-issue-triage-bot
pip install -r requirements.txt
python app.py
```

## Project Structure

```plaintext
github-issue-triage-bot/
├── roberta_model/
├── templates/
├── static/
├── app.py
├── requirements.txt
└── README.md
```

