import re
from src.logger import logging
from src.exception import CustomException
import sys

class TextPreprocessor:
    def __init__(self):
        pass

    def preprocess(self, text):
        try:
            logging.info("Started text preprocessing")
            t = text.lower()  # Lowercase the text
            
            # Fix: The original notebook likely intended to remove URLs. 
            # Using a more robust regex for URLs.
            t = re.sub(r'https?://\S+|www\.\S+', "", t) 
            
            t = re.sub(r'\d+', "", t) # Removes the digits or numbers
            t = re.sub(r'[^\w\s]', "", t) # Remove everything other than text
            
            logging.info("Completed text preprocessing")
            return t
        except Exception as e:
            raise CustomException(e, sys)
