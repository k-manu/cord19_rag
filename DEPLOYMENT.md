# 🚀 Deployment Guide: COVID-19 RAG Chatbot

This guide will help you deploy your RAG application using Hugging Face for vectorstore hosting and Streamlit Community Cloud for the web app.

## Prerequisites

- Hugging Face account ([sign up here](https://huggingface.co/join))
- GitHub account
- OpenAI API key
- Your local vectorstore files in `chroma_cord19/` directory

## Step 1: Upload Vectorstore to Hugging Face

### 1.1 Login to Hugging Face

```bash
# Install Hugging Face CLI if not already installed
pip install huggingface_hub

# Login to your account
huggingface-cli login
```

### 1.2 Run the Upload Script

```bash
python upload_to_hf.py
```

- Enter your Hugging Face username when prompted
- The script will create a dataset called `covid19-cord19-vectorstore`
- Your vectorstore will be uploaded to: `https://huggingface.co/datasets/YOUR_USERNAME/covid19-cord19-vectorstore`

### 1.3 Update Configuration

After successful upload, update the `HF_DATASET_ID` in one of these ways:

**Option A: Direct code update (simple)**
```python
# In app.py, line ~14, replace:
HF_DATASET_ID = "YOUR_USERNAME/covid19-cord19-vectorstore"
# With your actual dataset ID:
HF_DATASET_ID = "your-actual-username/covid19-cord19-vectorstore"
```

**Option B: Use Streamlit secrets (recommended for deployment)**
- Keep the code as-is and configure via Streamlit secrets (see Step 2.3)

## Step 2: Deploy to Streamlit Community Cloud

### 2.1 Remove Large Files and Push to GitHub

```bash
# Remove vectorstore from Git (too large for Streamlit deployment)
git rm --cached -r chroma_cord19

# Add all files except vectorstore (in .gitignore)
git add .
git commit -m "Remove large files, deploy with HF vectorstore download"
git push origin main
```

**Note**: Streamlit Community Cloud has size limits, so we exclude the vectorstore files and download them from Hugging Face at runtime instead.

### 2.2 Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub repository
4. Set:
   - **Repository**: `your-username/rag-app`
   - **Branch**: `main`
   - **Main file path**: `app.py`

### 2.3 Configure Secrets

In the Streamlit Cloud dashboard, add these secrets:

```toml
[openai]
OPENAI_API_KEY = "sk-your-actual-openai-api-key"

[huggingface]
HF_DATASET_ID = "your-username/covid19-cord19-vectorstore"
```

## Step 3: First Run

1. Your app will be available at `https://your-app-name.streamlit.app`
2. On first run, the app will:
   - Download the vectorstore from Hugging Face (~2-5 minutes)
   - Initialize the RAG chain
   - Be ready for questions!

## Architecture Overview

```
┌─────────────────┐    ┌───────────────────┐    ┌──────────────────┐
│  Streamlit App  │───▶│  Hugging Face     │───▶│  OpenAI API      │
│  (UI + Logic)   │    │  (Vectorstore)    │    │  (Embeddings +   │
│                 │    │                   │    │   LLM)           │
└─────────────────┘    └───────────────────┘    └──────────────────┘
```

## Benefits of This Setup

✅ **Scalable**: Large vectorstore files hosted on HF  
✅ **Fast**: Downloads cached, only happens once  
✅ **Free**: Both HF datasets and Streamlit Community Cloud are free  
✅ **Reliable**: Automatic scaling and uptime  
✅ **Maintainable**: Easy to update vectorstore separately  

## Troubleshooting

### "Error downloading vectorstore"
- Check your `HF_DATASET_ID` is correct
- Ensure the dataset is public (not private)
- Verify Hugging Face dataset exists

### "OpenAI API Key not found"
- Check Streamlit secrets configuration
- Ensure the secret key name is exactly `OPENAI_API_KEY`

### "Failed to initialize RAG chain"
- Check OpenAI API key has sufficient credits
- Ensure vectorstore downloaded successfully
- Check Streamlit logs for specific error details

### App takes long to start
- First run downloads vectorstore (~2-5 min)
- Subsequent runs should be fast (cached)
- Check Streamlit Cloud resource limits

## Cost Estimates

- **Hugging Face**: Free for public datasets up to 10GB
- **Streamlit Cloud**: Free tier with reasonable limits
- **OpenAI API**: Pay per usage (~$0.01-0.10 per conversation)

## Updates and Maintenance

### Updating the Vectorstore
1. Update your local `chroma_cord19/` directory
2. Run `python upload_to_hf.py` again
3. Restart your Streamlit app to download new version

### Updating the App
1. Push changes to GitHub
2. Streamlit Cloud will auto-deploy

## Security Notes

- Keep your OpenAI API key secure
- Consider using private HF datasets for proprietary data
- Monitor API usage to prevent unexpected costs

---

🎉 **You're all set!** Your RAG chatbot should now be live and accessible to users worldwide! 