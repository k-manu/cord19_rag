#!/usr/bin/env python3
"""
Script to upload the Chroma vectorstore to Hugging Face Hub as a dataset.
Run this once to upload your vectorstore files.
"""

import os
import shutil
from huggingface_hub import HfApi, create_repo
from pathlib import Path

def upload_vectorstore_to_hf():
    # Configuration
    REPO_NAME = "chroma_cord19"  # Change this to your preferred name
    HF_USERNAME = input("Enter your Hugging Face username: ").strip()
    REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"
    
    # Check if vectorstore exists
    vectorstore_path = Path("chroma_cord19")
    if not vectorstore_path.exists():
        print("❌ chroma_cord19 directory not found!")
        return
    
    print(f"📂 Found vectorstore at: {vectorstore_path}")
    print(f"🚀 Uploading to: https://huggingface.co/datasets/{REPO_ID}")
    
    try:
        # Initialize HF API
        api = HfApi()
        
        # Create repository (this will fail gracefully if it already exists)
        try:
            create_repo(
                repo_id=REPO_ID,
                repo_type="dataset",
                private=False,  # Set to True if you want a private dataset
                exist_ok=True
            )
            print(f"✅ Repository created/verified: {REPO_ID}")
        except Exception as e:
            print(f"⚠️  Repository might already exist: {e}")
        
        # Upload the entire vectorstore directory
        print("📤 Uploading vectorstore files...")
        api.upload_folder(
            folder_path=str(vectorstore_path),
            repo_id=REPO_ID,
            repo_type="dataset",
            path_in_repo="vectorstore"
        )
        
        print(f"🎉 Successfully uploaded vectorstore to: https://huggingface.co/datasets/{REPO_ID}")
        print(f"📝 Update your app.py with REPO_ID: '{REPO_ID}'")
        
        # Create a simple README for the dataset
        readme_content = f"""# COVID-19 CORD-19 Vectorstore

This dataset contains a Chroma vectorstore with embeddings for COVID-19 research papers from the CORD-19 dataset.

## Usage

This vectorstore is used by a Streamlit RAG application for COVID-19 research Q&A.

## Contents

- `vectorstore/`: Chroma vectorstore files including embeddings and metadata
- Papers: ~2000 recent COVID-19 research papers
- Embeddings: OpenAI embeddings

## Citation

If you use this dataset, please cite the original CORD-19 dataset.
"""
        
        with open("README_dataset.md", "w") as f:
            f.write(readme_content)
        
        api.upload_file(
            path_or_fileobj="README_dataset.md",
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="dataset"
        )
        
        # Clean up
        os.remove("README_dataset.md")
        
    except Exception as e:
        print(f"❌ Error uploading to Hugging Face: {e}")
        print("💡 Make sure you're logged in with `huggingface-cli login`")

if __name__ == "__main__":
    print("🤗 Hugging Face Vectorstore Uploader")
    print("=" * 40)
    
    # Check if user is logged in
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"👤 Logged in as: {user_info['name']}")
    except Exception:
        print("❌ Not logged in to Hugging Face!")
        print("Please run: huggingface-cli login")
        exit(1)
    
    upload_vectorstore_to_hf() 