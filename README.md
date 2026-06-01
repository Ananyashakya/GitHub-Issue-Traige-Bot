# GitHub Issue Triage Bot

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/Ananyashakya/GitHub-Issue-Traige-Bot)
[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![HTML](https://img.shields.io/badge/Frontend-HTML%2FCSs%2FJS-orange)](https://developer.mozilla.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](#license)

An intelligent, NLP-powered GitHub issue classification system that automates issue triaging using a fine-tuned RoBERTa model. This bot learns from existing GitHub issues and automatically categorizes new issues with high accuracy.

## Features

- **Automated Issue Classification** - Intelligently categorize GitHub issues without manual intervention
- **Single & Batch Predictions** - Classify one issue or process multiple issues at once
- **Confidence Score Visualization** - View prediction confidence with interactive charts
- **Analytics Dashboard** - Comprehensive statistics and insights about classified issues
- **Excel Report Export** - Generate and download detailed classification reports
- **Role-Based Authentication** - Secure access with user roles and permissions
- **Real-Time Processing** - Quick predictions powered by RoBERTa neural network

## Issue Categories

The bot classifies issues into 5 categories:

| Category | Description |
|----------|-------------|
| **Bug** | Code defects and issues |
| **Enhancement** | New features and improvements |
| **Documentation** | Documentation updates and guides |
| **Performance** | Performance optimizations and improvements |
| **Question** | User questions and discussions |

## Model Performance

Trained on a comprehensive dataset with excellent accuracy:

| Metric | Value |
|--------|-------|
| Training Dataset | 7,000+ GitHub Issues |
| Classification Categories | 5 |
| Overall Accuracy | **83.6%** |
| Macro F1-Score | **0.835** |

## Tech Stack

### Frontend
- **HTML** - Markup and structure
- **CSS** - Styling and responsive design
- **JavaScript** - Interactive features and client-side logic
- **Chart.js** - Data visualization and charts

### Backend
- **Flask** - Web framework
- **Python** - Core application logic

### Machine Learning
- **RoBERTa** - Pre-trained transformer model
- **PyTorch** - Deep learning framework
- **Hugging Face Transformers** - NLP model library
- **Pandas** - Data processing and analysis

## Quick Start

### Prerequisites
- Python 3.7+
- pip (Python package manager)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ananyashakya/GitHub-Issue-Traige-Bot.git
   cd GitHub-Issue-Traige-Bot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Access the web interface**
   - Open your browser and navigate to `http://localhost:5000`
   - The dashboard will be available for issue classification

## Project Structure

```
GitHub-Issue-Traige-Bot/
├── roberta_model/              # Pre-trained RoBERTa model files
├── templates/                  # HTML templates for web interface
│   ├── index.html             # Main dashboard page
│   ├── single_predict.html    # Single issue prediction page
│   ├── batch_predict.html     # Batch prediction page
│   └── ...                    # Other template files
├── static/                     # Static assets
│   ├── css/                   # Stylesheets
│   ├── js/                    # JavaScript files
│   └── images/                # Image assets
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Git ignore rules
```

## Usage

### Single Issue Prediction
1. Navigate to the "Single Prediction" section
2. Enter the issue title and description
3. Click "Predict" to get the classification
4. View the predicted category and confidence score

### Batch Prediction
1. Go to the "Batch Prediction" section
2. Upload a CSV or JSON file containing issues
3. The bot will classify all issues automatically
4. Download the results as an Excel report

### Analytics Dashboard
- View statistics about classified issues
- Monitor classification accuracy
- Analyze category distribution
- Track prediction confidence trends

## Authentication

The bot includes role-based access control:
- **Admin** - Full access to all features
- **User** - Can make predictions and view results
- **Viewer** - Read-only access to analytics

## API Documentation

### Classification Endpoint
```bash
POST /api/predict
Content-Type: application/json

{
  "title": "Issue title",
  "description": "Detailed issue description"
}
```

**Response:**
```json
{
  "category": "Bug",
  "confidence": 0.92,
  "scores": {
    "bug": 0.92,
    "enhancement": 0.04,
    "documentation": 0.02,
    "performance": 0.01,
    "question": 0.01
  }
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Known Issues & Limitations

- Model accuracy varies based on issue quality and description completeness
- Performance may depend on the complexity of issue descriptions
- The model is trained on English-language issues

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Ananyashakya**
- GitHub: [@Ananyashakya](https://github.com/Ananyashakya)

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the RoBERTa model
- [PyTorch](https://pytorch.org/) for the deep learning framework
- [Flask](https://flask.palletsprojects.com/) for the web framework

## Support

If you encounter any issues or have questions, please:
1. Check the existing [GitHub Issues](https://github.com/Ananyashakya/GitHub-Issue-Traige-Bot/issues)
2. Create a new issue with detailed information
3. Include reproduction steps and error messages

---

Made with care by Ananyashakya
