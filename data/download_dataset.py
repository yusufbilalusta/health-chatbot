import os
import kagglehub
import shutil

def download_health_chatbot_dataset():
    """
    Download the health chatbot dataset from KaggleHub if data directories are empty.
    """
    # Check if data directories are empty
    rag_corpus_dir = os.path.join("data", "rag_corpus")
    intents_dataset_dir = os.path.join("data", "intents_dataset")
    
    # Create directories if they don't exist
    os.makedirs(rag_corpus_dir, exist_ok=True)
    os.makedirs(intents_dataset_dir, exist_ok=True)
    
    # Check if the directories are empty
    rag_corpus_empty = len(os.listdir(rag_corpus_dir)) == 0
    intents_dataset_empty = len(os.listdir(intents_dataset_dir)) == 0
    
    if rag_corpus_empty or intents_dataset_empty:
        print("Data directories are empty. Downloading dataset from KaggleHub...")
        
        # Download latest version
        kaggle_path = kagglehub.dataset_download("yusufbilalusta/health-chatbot-dataset")
        print("Dataset downloaded to:", kaggle_path)
        
        # Copy files directly from the KaggleHub cache to the project's data directories
        kaggle_rag_corpus_dir = os.path.join(kaggle_path, "rag_corpus")
        kaggle_intents_dataset_dir = os.path.join(kaggle_path, "intents_dataset")
        
        # Check if the directories exist in the downloaded dataset
        if os.path.exists(kaggle_rag_corpus_dir) and rag_corpus_empty:
            print(f"Copying RAG corpus files from {kaggle_rag_corpus_dir} to {rag_corpus_dir}")
            for file_name in os.listdir(kaggle_rag_corpus_dir):
                src_file = os.path.join(kaggle_rag_corpus_dir, file_name)
                dst_file = os.path.join(rag_corpus_dir, file_name)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file)
                    print(f"Copied {file_name} to {rag_corpus_dir}")
        
        if os.path.exists(kaggle_intents_dataset_dir) and intents_dataset_empty:
            print(f"Copying intents dataset files from {kaggle_intents_dataset_dir} to {intents_dataset_dir}")
            for file_name in os.listdir(kaggle_intents_dataset_dir):
                src_file = os.path.join(kaggle_intents_dataset_dir, file_name)
                dst_file = os.path.join(intents_dataset_dir, file_name)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file)
                    print(f"Copied {file_name} to {intents_dataset_dir}")
        
        print("Dataset files copied successfully.")
        
        # Verify files were copied correctly
        print("\nVerifying copied files:")
        print(f"RAG corpus directory ({rag_corpus_dir}) contains {len(os.listdir(rag_corpus_dir))} files.")
        print(f"Intents dataset directory ({intents_dataset_dir}) contains {len(os.listdir(intents_dataset_dir))} files.")
    else:
        print("Data directories already contain files. No need to download.")

if __name__ == "__main__":
    download_health_chatbot_dataset() 