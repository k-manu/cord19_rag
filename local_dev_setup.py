#!/usr/bin/env python3
"""
Local development setup checker for COVID-19 RAG app.
Run this to verify your local environment is ready for development or deployment.
"""

import os
from pathlib import Path

def check_local_setup():
    """Check if the local development environment is properly configured"""
    print("🔍 Checking Local Development Setup")
    print("=" * 40)
    
    issues = []
    success = []
    
    # Check 1: Environment variables
    if os.getenv("OPENAI_API_KEY"):
        success.append("✅ OPENAI_API_KEY found in environment")
    else:
        issues.append("❌ OPENAI_API_KEY not found in environment")
        print("   💡 Create .env file with: OPENAI_API_KEY=your-key-here")
    
    # Check 2: Vectorstore
    vectorstore_path = Path("chroma_cord19")
    if vectorstore_path.exists():
        # Check if it has content
        files = list(vectorstore_path.glob("**/*"))
        if len(files) > 0:
            success.append(f"✅ Local vectorstore found with {len(files)} files")
        else:
            success.append("⚠️  Empty vectorstore directory (will download from HF)")
    else:
        success.append("ℹ️  No local vectorstore (will download from HF on first run)")
        print("   💡 This is the expected setup for deployment")
    
    # Check 3: Required packages
    try:
        import langchain
        import openai
        import chromadb
        import huggingface_hub
        success.append("✅ All required packages installed")
    except ImportError as e:
        issues.append(f"❌ Missing package: {e}")
        print("   💡 Run: pip install -r requirements.txt")
    
    # Check 4: Deployment files
    if Path("upload_to_hf.py").exists():
        success.append("✅ HuggingFace upload script ready")
    else:
        issues.append("❌ upload_to_hf.py not found")
    
    if Path("DEPLOYMENT.md").exists():
        success.append("✅ Deployment guide available")
    else:
        issues.append("❌ DEPLOYMENT.md not found")
    
    # Check 5: Git status
    if Path(".git").exists():
        success.append("✅ Git repository initialized")
        if Path(".gitignore").exists():
            with open(".gitignore", "r") as f:
                content = f.read()
                if "chroma_cord19/" in content:
                    success.append("✅ Vectorstore excluded from git")
                else:
                    issues.append("❌ Vectorstore not excluded from git")
        else:
            issues.append("❌ .gitignore not found")
    else:
        issues.append("❌ Not a git repository")
        print("   💡 Run: git init && git add . && git commit -m 'Initial commit'")
    
    # Summary
    print("\n📊 Setup Summary")
    print("-" * 20)
    
    for item in success:
        print(item)
    
    if issues:
        print("\n⚠️  Issues to resolve:")
        for item in issues:
            print(item)
        print(f"\n🔧 {len(issues)} issue(s) found. Please resolve them before deployment.")
    else:
        print("\n🎉 Everything looks good! You're ready to deploy.")
        print("\n📋 Next steps:")
        print("1. Run: python upload_to_hf.py (if not done already)")
        print("2. Push to GitHub: git push origin main")
        print("3. Deploy on Streamlit Cloud")
        print("4. Configure secrets in Streamlit dashboard")
    
    return len(issues) == 0

if __name__ == "__main__":
    success = check_local_setup()
    exit(0 if success else 1) 