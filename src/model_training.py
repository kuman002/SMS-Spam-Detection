import os
import sys
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.logger import logging
from src.exception import CustomException
from src.model_preprocess import TextPreprocessor

class ModelTrainer:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "best_spam_pipeline.pkl")
        self.data_path = "data/spam.csv"
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

    def initiate_model_training(self):
        try:
            logging.info("Initiating model training")
            
            # Load Data
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"File {self.data_path} not found.")
                
            msg = pd.read_csv(self.data_path, encoding="latin-1")
            
            # Drop unnecessary columns
            msg = msg.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
            msg.columns = ["labels", "text"]
            
            logging.info("Read and cleaned dataframe structure")

            # Preprocessing
            preprocessor = TextPreprocessor()
            msg['text'] = msg['text'].apply(preprocessor.preprocess)
            
            logging.info("Preprocessing completed")

            # Split Data
            x_train, x_test, y_train, y_test = train_test_split(
                msg["text"], msg["labels"], test_size=0.2, random_state=42
            )

            # Create Pipeline
            pipe = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=2)),
                ('clf', LinearSVC(class_weight='balanced', C=1.0, max_iter=5000))
            ])
            # Note: Notebook used GridSearchCV to find these params, so we use them directly.
            
            logging.info("Fitting the model")
            pipe.fit(x_train, y_train)

            # Evaluate
            pred = pipe.predict(x_test)
            report = classification_report(y_test, pred)
            logging.info(f"Classification Report:\n{report}")
            print(report)

            # Save Model
            joblib.dump(pipe, self.model_path)
            logging.info(f"Model saved at {self.model_path}")

            return self.model_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.initiate_model_training()
