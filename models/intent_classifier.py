import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

class IntentClassifier:
    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Initialize the intent classifier with a sentence transformer model
        
        Args:
            model_name (str): The name of the sentence transformer model
        """
        self.embedding_model = SentenceTransformer(model_name)
        self.classifier = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', multi_class='multinomial')
        self.intents = []
        self.model_path = 'models/intent_classifier.pkl'
        
    def train(self, data_path, test_size=0.2, random_state=42):
        """
        Train the intent classifier
        
        Args:
            data_path (str): Path to the CSV file with intents data
            test_size (float): Test set size for evaluation
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Dictionary with performance metrics
        """
        # Load the dataset
        df = pd.read_csv(data_path)
        
        # Check that we have both 'Intent' and 'Text' columns
        if 'Intent' not in df.columns or 'Text' not in df.columns:
            raise ValueError("The dataset must contain 'Intent' and 'Text' columns")
        
        # Get unique intents and store them
        self.intents = df['Intent'].unique().tolist()
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            df['Text'].values, 
            df['Intent'].values, 
            test_size=test_size, 
            random_state=random_state,
            stratify=df['Intent']
        )
        
        # Generate embeddings for the texts
        X_train_embeddings = self._get_embeddings(X_train)
        X_test_embeddings = self._get_embeddings(X_test)
        
        # Train the classifier
        self.classifier.fit(X_train_embeddings, y_train)
        
        # Make predictions on the test set
        y_pred = self.classifier.predict(X_test_embeddings)
        
        # Calculate performance metrics
        metrics = {
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # Save the model
        self.save_model()
        
        return metrics, X_test.tolist(), y_test.tolist(), y_pred.tolist()
    
    def predict(self, text):
        """
        Predict the intent of a text
        
        Args:
            text (str): The text to predict
            
        Returns:
            str: The predicted intent
        """
        # Generate embeddings for the text
        embedding = self._get_embeddings([text])
        
        # Make prediction
        intent_idx = self.classifier.predict(embedding)[0]
        
        return intent_idx
    
    def _get_embeddings(self, texts):
        """
        Generate embeddings for a list of texts
        
        Args:
            texts (list): List of texts
            
        Returns:
            numpy.ndarray: Array of embeddings
        """
        return self.embedding_model.encode(texts)
    
    def save_model(self):
        """Save the model to disk"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'classifier': self.classifier,
                'intents': self.intents
            }, f)
        print(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load the model from disk"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.classifier = model_data['classifier']
                self.intents = model_data['intents']
            print(f"Model loaded from {self.model_path}")
            return True
        else:
            print(f"Model file {self.model_path} not found")
            return False 