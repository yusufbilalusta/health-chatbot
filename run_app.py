import os
import subprocess

def main():
    """
    Run the Streamlit app
    """
    print("Starting Sağlık Bilgilendirme Chatbot...")
    
    # Check if the app directory exists
    if not os.path.exists("app/streamlit_app.py"):
        print("Error: app/streamlit_app.py not found!")
        return
    
    # Run the Streamlit app
    subprocess.run(["streamlit", "run", "app/streamlit_app.py"])

if __name__ == "__main__":
    main() 