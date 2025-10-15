---
title: arXiv Research Assistant
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: mit
---

# 🔬 arXiv Research Assistant

An intelligent AI-powered chatbot for exploring computer science research papers from the arXiv dataset. Built with advanced NLP techniques for semantic search, summarization, and concept explanation.

## 🌟 Features

- **💬 Intelligent Chat Interface**: Ask complex questions about research topics
- **🔍 Semantic Search**: Find relevant papers using transformer-based embeddings
- **📝 Automatic Summarization**: Generate concise summaries of research papers
- **🎯 Concept Extraction**: Identify and visualize key concepts from papers
- **📊 Interactive Visualizations**: Explore dataset statistics and trends
- **🤖 Multiple AI Models**: Combines 4+ state-of-the-art models for comprehensive assistance

## 🚀 Quick Start

1. **Upload the arXiv dataset** (JSON format) using the sidebar
2. **Load AI models** by clicking the "Load AI Models" button
3. **Start exploring** through four powerful interfaces:
   - Chat Assistant
   - Paper Search
   - Paper Summaries
   - Visualizations

## 📊 Supported Capabilities

### Question Answering
Ask questions like:
- "What are transformers in deep learning?"
- "Explain attention mechanisms"
- "What's the difference between supervised and unsupervised learning?"

### Paper Search
- Semantic search across thousands of papers
- Relevance scoring with similarity metrics
- Category-based filtering

### Summarization
- Automatic abstract summarization
- Key concept extraction
- Multi-paper comparison

### Visualizations
- Category distribution charts
- Concept frequency analysis
- Dataset statistics

## 🤖 AI Models Used

| Model | Purpose | Provider |
|-------|---------|----------|
| all-MiniLM-L6-v2 | Semantic embeddings | Sentence Transformers |
| facebook/bart-large-cnn | Text summarization | Meta AI |
| deepset/roberta-base-squad2 | Question answering | deepset |
| facebook/opt-350m | Text generation | Meta AI |

## 📥 Dataset

This application works with the [arXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv) from Kaggle.

**Download Instructions:**
1. Visit the [dataset page](https://www.kaggle.com/datasets/Cornell-University/arxiv)
2. Download the JSON file
3. Upload it through the app interface

**Dataset Coverage:**
- Computer Science papers (cs.*)
- Configurable sample size (1K - 50K papers)
- Papers from multiple CS subcategories

## 🎯 Use Cases

### For Researchers
- Quickly find relevant papers in your field
- Get summaries of papers to decide what to read
- Explore related work and citations

### For Students
- Understand complex concepts with AI explanations
- Find papers for literature reviews
- Learn about research trends

### For Educators
- Curate reading lists for courses
- Find examples for teaching concepts
- Track research developments

## 🛠️ Technical Details

**Frontend:** Streamlit  
**Backend:** Python 3.10+  
**ML Framework:** PyTorch, Transformers  
**Vector Search:** Sentence Transformers + Cosine Similarity  

**Architecture:**