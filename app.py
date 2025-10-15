import streamlit as st
import pandas as pd
import json
import re
from typing import List, Dict
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModel,
    pipeline
)
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="arXiv Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .paper-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .concept-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'papers_data' not in st.session_state:
    st.session_state.papers_data = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

@st.cache_resource
def load_models():
    """Load all required models"""
    try:
        # Embedding model for semantic search
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Summarization model
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Question answering model
        qa_model = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Text generation model (smaller model for faster inference)
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
        gen_model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
        
        if torch.cuda.is_available():
            gen_model = gen_model.to('cuda')
        
        return {
            'embedding': embedding_model,
            'summarizer': summarizer,
            'qa': qa_model,
            'generator': gen_model,
            'tokenizer': tokenizer
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

def load_arxiv_data(uploaded_file, sample_size: int = 10000):
    """Load and preprocess arXiv dataset"""
    try:
        # Read the uploaded file
        papers = []
        content = uploaded_file.getvalue().decode('utf-8')
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if i >= sample_size or not line.strip():
                break
            try:
                paper = json.loads(line)
                # Filter for computer science papers
                if 'categories' in paper and 'cs.' in paper.get('categories', '').lower():
                    papers.append({
                        'id': paper.get('id', ''),
                        'title': paper.get('title', ''),
                        'abstract': paper.get('abstract', ''),
                        'categories': paper.get('categories', ''),
                        'authors': paper.get('authors', ''),
                        'update_date': paper.get('update_date', '')
                    })
            except json.JSONDecodeError:
                continue
        
        df = pd.DataFrame(papers)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def create_embeddings(texts: List[str], model):
    """Create embeddings for text data"""
    try:
        embeddings = model.encode(texts, show_progress_bar=True)
        return embeddings
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")
        return None

def semantic_search(query: str, embeddings, texts: List[str], model, top_k: int = 5):
    """Perform semantic search"""
    try:
        query_embedding = model.encode([query])
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'index': int(idx),
                'similarity': float(similarities[idx]),
                'text': texts[idx]
            })
        return results
    except Exception as e:
        st.error(f"Error in semantic search: {e}")
        return []

def summarize_text(text: str, summarizer, max_length: int = 150):
    """Summarize text using transformer model"""
    try:
        # Clean and truncate text
        text = text.replace('\n', ' ').strip()
        if len(text) < 50:
            return text
        
        # BART can handle up to 1024 tokens
        text = text[:3000]
        
        summary = summarizer(
            text,
            max_length=max_length,
            min_length=30,
            do_sample=False
        )
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error summarizing: {str(e)}"

def answer_question(question: str, context: str, qa_model):
    """Answer questions based on context"""
    try:
        result = qa_model(question=question, context=context)
        return result['answer'], result['score']
    except Exception as e:
        return f"Error: {str(e)}", 0.0

def generate_explanation(topic: str, context: str, models):
    """Generate explanation for a concept"""
    try:
        tokenizer = models['tokenizer']
        model = models['generator']
        
        prompt = f"Explain the concept of {topic} in computer science:\n{context[:500]}\n\nExplanation:"
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
        
        explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the explanation part
        if "Explanation:" in explanation:
            explanation = explanation.split("Explanation:")[1].strip()
        
        return explanation
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

def extract_key_concepts(text: str, top_n: int = 10):
    """Extract key concepts from text"""
    # Simple keyword extraction based on frequency
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    
    # Filter out common words
    stop_words = {'this', 'that', 'with', 'from', 'have', 'been', 'which', 
                  'their', 'there', 'these', 'those', 'would', 'could', 'should'}
    words = [w for w in words if w not in stop_words]
    
    word_freq = Counter(words)
    return word_freq.most_common(top_n)

def visualize_concepts(concepts: List[tuple]):
    """Visualize key concepts"""
    if not concepts:
        return None
    
    words, frequencies = zip(*concepts)
    
    fig = go.Figure(data=[
        go.Bar(x=list(words), y=list(frequencies), marker_color='#1f77b4')
    ])
    
    fig.update_layout(
        title="Key Concepts Frequency",
        xaxis_title="Concepts",
        yaxis_title="Frequency",
        height=400,
        template="plotly_white"
    )
    
    return fig

def visualize_paper_categories(papers_df):
    """Visualize paper categories distribution"""
    if papers_df is None or len(papers_df) == 0:
        return None
    
    # Extract main categories
    categories = []
    for cats in papers_df['categories']:
        if isinstance(cats, str):
            main_cat = cats.split()[0] if cats else 'unknown'
            categories.append(main_cat)
    
    cat_counts = Counter(categories).most_common(10)
    
    if not cat_counts:
        return None
    
    labels, values = zip(*cat_counts)
    
    fig = go.Figure(data=[
        go.Pie(labels=labels, values=values, hole=0.3)
    ])
    
    fig.update_layout(
        title="Top Paper Categories",
        height=400,
        template="plotly_white"
    )
    
    return fig

# Main UI
st.markdown('<p class="main-header">🔬 arXiv Research Assistant</p>', unsafe_allow_html=True)
st.markdown("### Your AI-powered guide to computer science research papers")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload arXiv dataset (JSON)",
        type=['json', 'jsonl'],
        help="Upload the arXiv dataset JSON file"
    )
    
    sample_size = st.slider(
        "Number of papers to load",
        min_value=1000,
        max_value=50000,
        value=10000,
        step=1000
    )
    
    if uploaded_file is not None:
        if st.button("🔄 Load Data"):
            with st.spinner("Loading dataset..."):
                st.session_state.papers_data = load_arxiv_data(uploaded_file, sample_size)
                
                if st.session_state.papers_data is not None:
                    st.success(f"✅ Loaded {len(st.session_state.papers_data)} papers!")
    
    st.divider()
    
    # Model loading
    if not st.session_state.models_loaded:
        if st.button("🚀 Load AI Models"):
            with st.spinner("Loading AI models... This may take a few minutes."):
                models = load_models()
                if models:
                    st.session_state.models = models
                    st.session_state.models_loaded = True
                    st.success("✅ Models loaded!")
                    st.rerun()
    else:
        st.success("✅ Models ready!")
    
    st.divider()
    
    # Dataset info
    if st.session_state.papers_data is not None:
        st.metric("Total Papers", len(st.session_state.papers_data))
        
        if st.button("📊 Show Dataset Statistics"):
            st.session_state.show_stats = True

# Main content area
if st.session_state.papers_data is not None and st.session_state.models_loaded:
    
    # Create embeddings if not already done
    if st.session_state.embeddings is None:
        with st.spinner("Creating embeddings for semantic search..."):
            combined_texts = (
                st.session_state.papers_data['title'] + " " + 
                st.session_state.papers_data['abstract']
            ).tolist()
            st.session_state.embeddings = create_embeddings(
                combined_texts,
                st.session_state.models['embedding']
            )
            st.success("✅ Embeddings ready!")
    
    # Tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Chat Assistant",
        "🔍 Paper Search",
        "📚 Paper Summaries",
        "📊 Visualizations"
    ])
    
    with tab1:
        st.subheader("Ask me anything about computer science research!")
        
        # Chat interface
        user_question = st.text_input(
            "Your question:",
            placeholder="e.g., What are the latest trends in deep learning?"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            ask_button = st.button("🚀 Ask", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Clear History", use_container_width=True)
        
        if clear_button:
            st.session_state.chat_history = []
            st.rerun()
        
        if ask_button and user_question:
            with st.spinner("Thinking..."):
                # Search relevant papers
                combined_texts = (
                    st.session_state.papers_data['title'] + " " + 
                    st.session_state.papers_data['abstract']
                ).tolist()
                
                search_results = semantic_search(
                    user_question,
                    st.session_state.embeddings,
                    combined_texts,
                    st.session_state.models['embedding'],
                    top_k=3
                )
                
                # Prepare context from top results
                context = "\n\n".join([r['text'][:500] for r in search_results])
                
                # Get answer
                answer, confidence = answer_question(
                    user_question,
                    context,
                    st.session_state.models['qa']
                )
                
                # Generate explanation
                explanation = generate_explanation(
                    user_question,
                    context,
                    st.session_state.models
                )
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'question': user_question,
                    'answer': answer,
                    'explanation': explanation,
                    'confidence': confidence,
                    'papers': [
                        st.session_state.papers_data.iloc[r['index']] 
                        for r in search_results
                    ]
                })
        
        # Display chat history
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.container():
                st.markdown(f"**🙋 Question:** {chat['question']}")
                st.markdown(f"**🤖 Answer:** {chat['answer']}")
                
                with st.expander("📖 Detailed Explanation"):
                    st.write(chat['explanation'])
                    st.metric("Confidence", f"{chat['confidence']:.2%}")
                
                with st.expander("📄 Related Papers"):
                    for paper in chat['papers'][:3]:
                        st.markdown(f"""
                        <div class="paper-card">
                            <strong>{paper['title']}</strong><br>
                            <small>{paper['categories']}</small>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.divider()
    
    with tab2:
        st.subheader("🔍 Search Research Papers")
        
        search_query = st.text_input(
            "Search for papers:",
            placeholder="e.g., neural networks, machine learning, computer vision"
        )
        
        num_results = st.slider("Number of results", 3, 20, 10)
        
        if st.button("🔍 Search"):
            if search_query:
                with st.spinner("Searching..."):
                    combined_texts = (
                        st.session_state.papers_data['title'] + " " + 
                        st.session_state.papers_data['abstract']
                    ).tolist()
                    
                    results = semantic_search(
                        search_query,
                        st.session_state.embeddings,
                        combined_texts,
                        st.session_state.models['embedding'],
                        top_k=num_results
                    )
                    
                    st.success(f"Found {len(results)} relevant papers")
                    
                    for i, result in enumerate(results, 1):
                        paper = st.session_state.papers_data.iloc[result['index']]
                        
                        with st.expander(f"#{i} - {paper['title']} (Similarity: {result['similarity']:.2%})"):
                            st.markdown(f"**Categories:** {paper['categories']}")
                            st.markdown(f"**Authors:** {paper['authors']}")
                            st.markdown(f"**Abstract:**")
                            st.write(paper['abstract'])
                            
                            # Generate summary
                            if st.button(f"📝 Summarize", key=f"sum_{i}"):
                                summary = summarize_text(
                                    paper['abstract'],
                                    st.session_state.models['summarizer']
                                )
                                st.markdown("**Summary:**")
                                st.info(summary)
    
    with tab3:
        st.subheader("📚 Paper Summaries")
        
        paper_idx = st.selectbox(
            "Select a paper:",
            range(min(100, len(st.session_state.papers_data))),
            format_func=lambda x: st.session_state.papers_data.iloc[x]['title'][:100]
        )
        
        if st.button("📝 Generate Summary"):
            paper = st.session_state.papers_data.iloc[paper_idx]
            
            st.markdown(f"### {paper['title']}")
            st.markdown(f"**Categories:** {paper['categories']}")
            st.markdown(f"**Authors:** {paper['authors']}")
            
            with st.spinner("Generating summary..."):
                summary = summarize_text(
                    paper['abstract'],
                    st.session_state.models['summarizer']
                )
                
                concepts = extract_key_concepts(paper['abstract'])
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("### 📄 Summary")
                    st.info(summary)
                    
                    st.markdown("### 📖 Full Abstract")
                    st.write(paper['abstract'])
                
                with col2:
                    st.markdown("### 🔑 Key Concepts")
                    for concept, freq in concepts:
                        st.markdown(f"- **{concept}** ({freq})")
    
    with tab4:
        st.subheader("📊 Dataset Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = visualize_paper_categories(st.session_state.papers_data)
            if fig1:
                st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Sample paper for concept visualization
            if len(st.session_state.papers_data) > 0:
                sample_paper = st.session_state.papers_data.iloc[0]
                concepts = extract_key_concepts(sample_paper['abstract'])
                fig2 = visualize_concepts(concepts)
                if fig2:
                    st.plotly_chart(fig2, use_container_width=True)
        
        # Dataset statistics
        if hasattr(st.session_state, 'show_stats') and st.session_state.show_stats:
            st.markdown("### 📈 Dataset Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Papers", len(st.session_state.papers_data))
            
            with col2:
                unique_cats = st.session_state.papers_data['categories'].nunique()
                st.metric("Unique Categories", unique_cats)
            
            with col3:
                avg_abstract_len = st.session_state.papers_data['abstract'].str.len().mean()
                st.metric("Avg Abstract Length", f"{avg_abstract_len:.0f}")
            
            with col4:
                total_authors = st.session_state.papers_data['authors'].str.split(',').str.len().sum()
                st.metric("Total Authors", f"{total_authors:.0f}")

else:
    # Welcome screen
    st.info("👆 Please upload the arXiv dataset and load the AI models from the sidebar to get started!")
    
    st.markdown("""
    ## Features:
    
    - 💬 **Intelligent Chat Assistant**: Ask complex questions about computer science research
    - 🔍 **Semantic Paper Search**: Find relevant papers using advanced NLP
    - 📝 **Automatic Summarization**: Get concise summaries of research papers
    - 🎯 **Concept Extraction**: Identify key concepts and topics
    - 📊 **Visualizations**: Explore dataset statistics and trends
    - 🤖 **Open-Source Models**: Powered by state-of-the-art transformers
    
    ## How to use:
    
    1. Download the arXiv dataset from [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv)
    2. Upload the JSON file in the sidebar
    3. Click "Load Data" and "Load AI Models"
    4. Start exploring research papers!
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    Built with Streamlit 🎈 | Powered by HuggingFace 🤗 | arXiv Dataset 📚
</div>
""", unsafe_allow_html=True)