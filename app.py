import os
import subprocess
import sys
from data.download_dataset import download_health_chatbot_dataset

def main():
    """
    Main function to run the Health Chatbot Streamlit application.
    Checks for the dataset and starts the Streamlit app.
    """
    print("Health Chatbot Application")
    print("-------------------------")
    
    # Download dataset if needed
    print("Checking dataset...")
    download_health_chatbot_dataset()
    
    # Run the Streamlit app
    print("\nStarting Streamlit app...")
    subprocess.run(["streamlit", "run", "app/streamlit_app.py"])

if __name__ == "__main__":
    main() 