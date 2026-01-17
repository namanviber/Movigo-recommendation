# Movigo Recommendation Engine

A high-performance recommendation API that provides personalized movie suggestions using Collaborative Filtering and SVD algorithms.

## 🚀 Overview

Movigo-recommendation is a Flask-based microservice designed to handle movie recommendation requests. It utilizes machine learning models trained on user rating data to predict and suggest movies that a user is likely to enjoy based on their past interactions and similar user profiles.

## ✨ Key Features

- **Personalized Recommendations**: Uses Single Value Decomposition (SVD) for accurate rating predictions.
- **User Similarity**: Implements Cosine Similarity to find users with similar tastes.
- **Microservice Architecture**: Exposes a RESTful API endpoint for seamless integration with frontend applications.
- **Efficient Processing**: Leverages pre-trained models (Pickle files) for fast inference.

## 🛠️ Project Structure

```text
Movigo-recommendation/
├── templates/              # HTML templates for the API landing page
├── main.py                 # Flask application and recommendation logic
├── model.ipynb             # Jupyter notebook used for model training
├── Scaled_ratings.pkl      # Pre-processed rating data for similarity calculations
├── newModel.pkl            # Trained SVD model weights
├── pivot.pkl               # User-Item interaction matrix
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 💻 Tech Stack

- **Framework**: Flask (Python)
- **ML Libraries**: Scikit-learn, Surprise (Surprise-Lib), Pandas, NumPy
- **Algorithm**: SVD (Collaborative Filtering), Cosine Similarity
- **Data Handling**: Pickle for model serialization

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/namanviber/Movigo-recommendation.git
   cd Movigo-recommendation
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the API server**:
   ```bash
   python main.py
   ```

## 📡 API Usage

- **Endpoint**: `/predict`
- **Method**: `POST`
- **Payload**: JSON array of user ratings (e.g., `[{"tmdbId": 123, "rating": 5.0}, ...]`)
- **Response**: List of recommended `tmdbId`s.
