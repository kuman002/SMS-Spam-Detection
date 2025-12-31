# SMS Spam Detection

A machine learning-based web application to detect whether an SMS message is **Spam** or **Ham** (not spam). This project uses a **Linear Support Vector Classifier (LinearSVC)** trained on the SMS Spam Collection dataset to classify messages with high accuracy.

## 🚀 Features

- **Real-time Prediction**: Instantly classify SMS messages as Spam or Ham.
- **Web Interface**: User-friendly web UI built with Flask, HTML, CSS, and JavaScript.
- **Text Preprocessing**: Includes robust text cleaning (removing URLs, numbers, special characters) ensuring the model focuses on relevant text features.
- **Model Pipeline**: deeply integrated `scikit-learn` pipeline with `TfidfVectorizer` and `LinearSVC`.
- **Logging & Exception Handling**: Comprehensive logging for debugging and custom exception handling.

## 📂 Project Structure

```
├── artifacts/          # Stores the trained model (best_spam_pipeline.pkl)
├── data/               # Dataset directory (spam.csv)
├── logs/               # Application logs
├── research/           # Jupyter notebooks for experimentation and analysis
├── src/                # Source code for model training and preprocessing
│   ├── exception.py    # Custom exception handling
│   ├── logger.py       # Logging configuration
│   ├── model_preprocess.py # Text preprocessing logic
│   └── model_training.py   # Script to train and save the model
├── templates/          # HTML templates for the Flask app
│   └── index.html      # Main user interface
├── app.py              # Main Flask application entry point
├── requirements.txt    # Python dependencies
└── setup.py            # Package setup script
```

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kuman002/SMS-Spam-Detection.git
    cd SMS-Spam-Detection
    ```

2.  **Create and activate a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    # For Windows:
    venv\Scripts\activate
    # For macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Usage

1.  **Start the Flask application:**
    ```bash
    python app.py
    ```

2.  **Access the application:**
    Open your web browser and navigate to `http://127.0.0.1:5000`.

3.  **Predict:**
    - Enter an SMS message in the text area.
    - Click **Analyze Message**.
    - The result (Spam or Ham) will be displayed instantly.

## 🧠 Model Details

The model is built using `scikit-learn`.

-   **Preprocessing**:
    -   Lowercasing
    -   URL removal
    -   Digit and special character removal
-   **Vectorization**: `TfidfVectorizer` (English stop words removed, N-grams: 1-2)
-   **Classifier**: `LinearSVC` (Balanced class weights)

To retrain the model, run:
```bash
python src/model_training.py
```
This will process the data in `data/spam.csv` and save the new model to `artifacts/best_spam_pipeline.pkl`.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 👤 Author

**Kuman02**