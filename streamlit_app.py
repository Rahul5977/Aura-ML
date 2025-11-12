"""
Aura ML Pipeline - Streamlit Educational Interface
==================================================
A transparent, educational interface for the Aura conversational ML pipeline.

Features:
- Real-time audio transcription (STT with Whisper)
- Speech emotion recognition (SER with Wav2Vec2)
- Named entity recognition (NER with spaCy)
- Contextual embeddings (COMET)
- Knowledge graph visualization
- Real-time pipeline visualization for educational transparency
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import base64
from typing import Dict, Any, List

# Page configuration
st.set_page_config(
    page_title="Aura ML Pipeline",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
BACKEND_URL = "http://localhost:8000"
MAX_AUDIO_DURATION = 300  # 5 minutes

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        max-width: 100%;
    }
    .pipeline-step {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
        background-color: #f5f5f5;
    }
    .pipeline-step.active {
        border-left-color: #2196F3;
        background-color: #e3f2fd;
    }
    .pipeline-step.complete {
        border-left-color: #4CAF50;
        background-color: #e8f5e9;
    }
    .pipeline-step.error {
        border-left-color: #f44336;
        background-color: #ffebee;
    }
    .metric-card {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 10px 0;
    }
    .entity-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 5px;
        margin: 5px;
        font-weight: bold;
    }
    .entity-PERSON { background-color: #ffeb3b; color: #000; }
    .entity-ORG { background-color: #2196f3; color: #fff; }
    .entity-GPE { background-color: #4caf50; color: #fff; }
    .entity-DATE { background-color: #ff9800; color: #fff; }
    .entity-TIME { background-color: #9c27b0; color: #fff; }
    .entity-EVENT { background-color: #f44336; color: #fff; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = f"conv_{int(time.time())}"
if 'speaker_id' not in st.session_state:
    st.session_state.speaker_id = "speaker_001"
if 'pipeline_logs' not in st.session_state:
    st.session_state.pipeline_logs = []

# Helper functions
def check_backend_health() -> bool:
    """Check if backend is running and healthy"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_models_status() -> Dict[str, Any]:
    """Get status of all ML models"""
    try:
        response = requests.get(f"{BACKEND_URL}/models/status", timeout=10)
        if response.status_code == 200:
            return response.json()
        return {"error": "Failed to get models status"}
    except Exception as e:
        return {"error": str(e)}

def process_audio_pipeline(audio_bytes: bytes, conversation_id: str, speaker_id: str) -> Dict[str, Any]:
    """Process audio through the complete ML pipeline"""
    try:
        files = {'file': ('audio.wav', audio_bytes, 'audio/wav')}
        data = {
            'conversation_id': conversation_id,
            'speaker_id': speaker_id,
            'include_graph': 'true'
        }
        
        response = requests.post(
            f"{BACKEND_URL}/orchestrate/analyze-audio",
            files=files,
            data=data,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}

def process_text_pipeline(text: str, conversation_id: str, speaker_id: str) -> Dict[str, Any]:
    """Process text through the NER and COMET pipeline"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/analyze/text",
            params={
                'text': text,
                'conversation_id': conversation_id,
                'speaker_id': speaker_id,
                'include_graph': 'true'
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}

def render_pipeline_visualization(stages: List[str], current_stage: str = None):
    """Render the pipeline visualization"""
    colors = []
    for stage in stages:
        if stage == current_stage:
            colors.append('#2196F3')  # Blue for current
        elif stages.index(stage) < stages.index(current_stage) if current_stage in stages else False:
            colors.append('#4CAF50')  # Green for complete
        else:
            colors.append('#9E9E9E')  # Grey for pending
    
    fig = go.Figure(data=[go.Bar(
        x=stages,
        y=[1] * len(stages),
        marker_color=colors,
        text=stages,
        textposition='inside',
        insidetextanchor='middle',
        hoverinfo='text',
        hovertext=[f"Stage: {s}" for s in stages]
    )])
    
    fig.update_layout(
        title="ML Pipeline Stages",
        showlegend=False,
        height=200,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False, range=[0, 1.2])
    )
    
    return fig

def render_entity_badges(entities: List[Dict[str, Any]]):
    """Render entity badges with colors"""
    html = "<div style='margin: 10px 0;'>"
    for entity in entities:
        label = entity.get('label', 'UNKNOWN')
        text = entity.get('text', '')
        html += f'<span class="entity-badge entity-{label}">{text} ({label})</span>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def render_emotion_chart(emotions: Dict[str, float]):
    """Render emotion scores as a bar chart"""
    if not emotions:
        return
    
    df = pd.DataFrame([
        {"Emotion": k, "Confidence": v}
        for k, v in emotions.items()
    ]).sort_values("Confidence", ascending=False)
    
    fig = px.bar(
        df,
        x="Confidence",
        y="Emotion",
        orientation='h',
        title="Speech Emotion Recognition Scores",
        color="Confidence",
        color_continuous_scale="Viridis"
    )
    
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_comet_scores(comet_data: Dict[str, Any]):
    """Render COMET embedding information"""
    if not comet_data or 'error' in comet_data:
        return
    
    st.markdown("### 🧠 COMET Contextual Embeddings")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Embedding Dimensions", comet_data.get('dimensions', 'N/A'))
    with col2:
        st.metric("Model", comet_data.get('model', 'COMET'))
    
    if 'embedding_preview' in comet_data:
        with st.expander("View Embedding Preview (first 10 dimensions)"):
            st.code(str(comet_data['embedding_preview']))

def add_to_conversation_history(text: str, speaker: str, result: Dict[str, Any]):
    """Add a conversation turn to history"""
    st.session_state.conversation_history.append({
        'timestamp': datetime.now().strftime("%H:%M:%S"),
        'speaker': speaker,
        'text': text,
        'result': result
    })

def render_conversation_history():
    """Render the conversation history"""
    if not st.session_state.conversation_history:
        st.info("No conversation history yet. Start by uploading audio or entering text!")
        return
    
    st.markdown("### 💬 Conversation History")
    
    for i, turn in enumerate(reversed(st.session_state.conversation_history[-5:])):
        with st.expander(f"🕐 {turn['timestamp']} - {turn['speaker']}", expanded=(i==0)):
            st.markdown(f"**Text:** {turn['text']}")
            
            if 'emotions' in turn['result']:
                st.markdown("**Detected Emotion:**")
                emotions = turn['result']['emotions']
                if emotions:
                    top_emotion = max(emotions.items(), key=lambda x: x[1])
                    st.markdown(f"🎭 {top_emotion[0].title()} ({top_emotion[1]:.2%})")
            
            if 'entities' in turn['result'] and turn['result']['entities']:
                st.markdown("**Detected Entities:**")
                render_entity_badges(turn['result']['entities'])

# Main app
def main():
    # Header
    st.title("🎤 Aura ML Pipeline - Educational Interface")
    st.markdown("**Real-time conversational AI analysis with transparent ML pipeline visualization**")
    
    # Sidebar for configuration and status
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Backend status
        is_healthy = check_backend_health()
        if is_healthy:
            st.success("✅ Backend Connected")
        else:
            st.error("❌ Backend Offline")
            st.markdown("**Start backend:**")
            st.code("cd aura-backend && uvicorn main:app --reload", language="bash")
            st.stop()
        
        # Models status
        with st.expander("🤖 ML Models Status", expanded=False):
            models_status = get_models_status()
            if 'error' not in models_status:
                for model, status in models_status.items():
                    if isinstance(status, dict):
                        loaded = status.get('loaded', False)
                        icon = "✅" if loaded else "⏳"
                        st.markdown(f"{icon} **{model}**: {status.get('status', 'Unknown')}")
                    else:
                        st.markdown(f"📊 **{model}**: {status}")
            else:
                st.error(f"Error: {models_status['error']}")
        
        st.markdown("---")
        
        # Session configuration
        st.header("📋 Session Info")
        st.session_state.conversation_id = st.text_input(
            "Conversation ID",
            value=st.session_state.conversation_id,
            help="Unique identifier for this conversation session"
        )
        st.session_state.speaker_id = st.text_input(
            "Speaker ID",
            value=st.session_state.speaker_id,
            help="Identifier for the current speaker"
        )
        
        if st.button("🔄 Reset Session"):
            st.session_state.conversation_history = []
            st.session_state.conversation_id = f"conv_{int(time.time())}"
            st.session_state.pipeline_logs = []
            st.success("Session reset!")
            st.rerun()
        
        st.markdown("---")
        
        # Documentation
        st.header("📚 Pipeline Overview")
        st.markdown("""
        **ML Pipeline Stages:**
        1. 🎤 **STT**: Speech-to-Text (Whisper)
        2. 🎭 **SER**: Speech Emotion Recognition (Wav2Vec2)
        3. 🏷️ **NER**: Named Entity Recognition (spaCy)
        4. 🧠 **COMET**: Contextual Embeddings
        5. 🕸️ **Knowledge Graph**: Neo4j Storage
        """)
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["🎤 Audio Analysis", "📝 Text Analysis", "📊 Pipeline Monitor", "🕸️ Knowledge Graph"])
    
    # Tab 1: Audio Analysis
    with tab1:
        st.header("🎤 Audio Analysis Pipeline")
        st.markdown("Upload or record audio to analyze through the complete ML pipeline")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            audio_source = st.radio("Audio Source:", ["Upload File", "Record Audio"], horizontal=True)
            
            audio_bytes = None
            
            if audio_source == "Upload File":
                audio_file = st.file_uploader(
                    "Upload audio file (WAV, MP3, M4A)",
                    type=['wav', 'mp3', 'm4a', 'flac', 'ogg'],
                    help="Upload an audio file to process through the ML pipeline"
                )
                if audio_file:
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/wav')
            else:
                audio_bytes = st.audio_input("Record your voice")
                
            if st.button("🚀 Process Audio", type="primary", disabled=not audio_bytes):
                if audio_bytes:
                    # Show pipeline visualization
                    st.markdown("### 🔄 Processing Pipeline")
                    pipeline_stages = ["STT (Whisper)", "SER (Wav2Vec2)", "NER (spaCy)", "COMET", "Knowledge Graph"]
                    
                    pipeline_viz = st.empty()
                    status_text = st.empty()
                    
                    # Initial pipeline view
                    pipeline_viz.plotly_chart(
                        render_pipeline_visualization(pipeline_stages, "STT (Whisper)"),
                        use_container_width=True
                    )
                    
                    status_text.info("🔄 Processing audio through ML pipeline...")
                    
                    # Process audio
                    with st.spinner("Processing..."):
                        result = process_audio_pipeline(
                            audio_bytes,
                            st.session_state.conversation_id,
                            st.session_state.speaker_id
                        )
                    
                    # Update pipeline to show completion
                    pipeline_viz.plotly_chart(
                        render_pipeline_visualization(pipeline_stages, None),
                        use_container_width=True
                    )
                    
                    if 'error' in result:
                        status_text.error(f"❌ Error: {result['error']}")
                    else:
                        status_text.success("✅ Pipeline completed successfully!")
                        
                        # Display results
                        st.markdown("---")
                        st.markdown("### 📊 Analysis Results")
                        
                        # Transcription
                        if 'transcription' in result:
                            st.markdown("#### 📝 Transcription (Whisper STT)")
                            st.info(result['transcription'])
                            text = result['transcription']
                        else:
                            text = "No transcription available"
                        
                        # Emotions
                        if 'emotions' in result and result['emotions']:
                            st.markdown("#### 🎭 Speech Emotion Recognition (Wav2Vec2)")
                            render_emotion_chart(result['emotions'])
                        
                        # Entities
                        if 'entities' in result and result['entities']:
                            st.markdown("#### 🏷️ Named Entities (spaCy NER)")
                            render_entity_badges(result['entities'])
                            
                            # Show entities table
                            entities_df = pd.DataFrame(result['entities'])
                            st.dataframe(entities_df, use_container_width=True)
                        
                        # COMET embeddings
                        if 'comet_embedding' in result:
                            render_comet_scores(result['comet_embedding'])
                        
                        # Knowledge graph
                        if 'graph_summary' in result:
                            st.markdown("#### 🕸️ Knowledge Graph Update")
                            graph_summary = result['graph_summary']
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Nodes Created", graph_summary.get('nodes_created', 0))
                            with col2:
                                st.metric("Relationships", graph_summary.get('relationships_created', 0))
                            with col3:
                                st.metric("Entities Linked", len(result.get('entities', [])))
                        
                        # Add to history
                        add_to_conversation_history(text, st.session_state.speaker_id, result)
                        
                        # Show raw JSON
                        with st.expander("🔍 View Raw API Response"):
                            st.json(result)
        
        with col2:
            st.markdown("### 📖 How It Works")
            st.markdown("""
            **Pipeline Stages:**
            
            1️⃣ **Speech-to-Text**
            - Model: Whisper (OpenAI)
            - Converts speech to text
            
            2️⃣ **Emotion Recognition**
            - Model: Wav2Vec2
            - Detects emotional tone
            
            3️⃣ **Entity Recognition**
            - Model: spaCy NER
            - Extracts people, places, dates
            
            4️⃣ **Embeddings**
            - Model: COMET
            - Contextual understanding
            
            5️⃣ **Knowledge Graph**
            - Database: Neo4j
            - Stores relationships
            """)
    
    # Tab 2: Text Analysis
    with tab2:
        st.header("📝 Text Analysis Pipeline")
        st.markdown("Analyze text through NER, COMET, and Knowledge Graph stages")
        
        text_input = st.text_area(
            "Enter text to analyze:",
            height=150,
            placeholder="Example: I met Sarah Johnson at Google headquarters in Mountain View on March 15th to discuss the AI project.",
            help="Enter text to extract entities, generate embeddings, and update the knowledge graph"
        )
        
        col1, col2 = st.columns([1, 3])
        with col1:
            process_text_btn = st.button("🚀 Analyze Text", type="primary", disabled=not text_input)
        
        if process_text_btn and text_input:
            # Show pipeline visualization
            st.markdown("### 🔄 Processing Pipeline")
            pipeline_stages = ["NER (spaCy)", "COMET", "Knowledge Graph"]
            
            pipeline_viz = st.empty()
            status_text = st.empty()
            
            # Initial pipeline view
            pipeline_viz.plotly_chart(
                render_pipeline_visualization(pipeline_stages, "NER (spaCy)"),
                use_container_width=True
            )
            
            status_text.info("🔄 Processing text through ML pipeline...")
            
            # Process text
            with st.spinner("Processing..."):
                result = process_text_pipeline(
                    text_input,
                    st.session_state.conversation_id,
                    st.session_state.speaker_id
                )
            
            # Update pipeline to show completion
            pipeline_viz.plotly_chart(
                render_pipeline_visualization(pipeline_stages, None),
                use_container_width=True
            )
            
            if 'error' in result:
                status_text.error(f"❌ Error: {result['error']}")
            else:
                status_text.success("✅ Analysis completed successfully!")
                
                # Display results
                st.markdown("---")
                st.markdown("### 📊 Analysis Results")
                
                # Entities
                if 'entities' in result and result['entities']:
                    st.markdown("#### 🏷️ Named Entities (spaCy NER)")
                    render_entity_badges(result['entities'])
                    
                    # Show entities table
                    entities_df = pd.DataFrame(result['entities'])
                    st.dataframe(entities_df, use_container_width=True)
                else:
                    st.info("No entities detected in the text")
                
                # COMET embeddings
                if 'comet_embedding' in result:
                    render_comet_scores(result['comet_embedding'])
                
                # Knowledge graph
                if 'graph_summary' in result:
                    st.markdown("#### 🕸️ Knowledge Graph Update")
                    graph_summary = result['graph_summary']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Nodes Created", graph_summary.get('nodes_created', 0))
                    with col2:
                        st.metric("Relationships", graph_summary.get('relationships_created', 0))
                    with col3:
                        st.metric("Entities Linked", len(result.get('entities', [])))
                
                # Add to history
                add_to_conversation_history(text_input, st.session_state.speaker_id, result)
                
                # Show raw JSON
                with st.expander("🔍 View Raw API Response"):
                    st.json(result)
    
    # Tab 3: Pipeline Monitor
    with tab3:
        st.header("📊 ML Pipeline Monitor")
        st.markdown("Real-time monitoring of the ML pipeline and conversation analysis")
        
        # Conversation history
        render_conversation_history()
        
        # Statistics
        if st.session_state.conversation_history:
            st.markdown("---")
            st.markdown("### 📈 Session Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Messages", len(st.session_state.conversation_history))
            
            with col2:
                total_entities = sum(
                    len(turn['result'].get('entities', []))
                    for turn in st.session_state.conversation_history
                )
                st.metric("Entities Extracted", total_entities)
            
            with col3:
                # Count unique entity types
                entity_types = set()
                for turn in st.session_state.conversation_history:
                    for entity in turn['result'].get('entities', []):
                        entity_types.add(entity.get('label', ''))
                st.metric("Entity Types", len(entity_types))
            
            with col4:
                st.metric("Active Conversation", st.session_state.conversation_id.split('_')[-1][:8])
            
            # Entity distribution
            if total_entities > 0:
                st.markdown("### 🏷️ Entity Distribution")
                
                entity_counts = {}
                for turn in st.session_state.conversation_history:
                    for entity in turn['result'].get('entities', []):
                        label = entity.get('label', 'UNKNOWN')
                        entity_counts[label] = entity_counts.get(label, 0) + 1
                
                df = pd.DataFrame([
                    {"Entity Type": k, "Count": v}
                    for k, v in entity_counts.items()
                ]).sort_values("Count", ascending=False)
                
                fig = px.pie(
                    df,
                    values="Count",
                    names="Entity Type",
                    title="Entity Types Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Knowledge Graph
    with tab4:
        st.header("🕸️ Knowledge Graph Explorer")
        st.markdown("Explore the knowledge graph built from conversation analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Get Graph Summary"):
                with st.spinner("Fetching graph summary..."):
                    try:
                        response = requests.get(f"{BACKEND_URL}/knowledge-graph/summary", timeout=10)
                        if response.status_code == 200:
                            summary = response.json()
                            
                            st.markdown("### 📈 Graph Statistics")
                            
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Total Nodes", summary.get('total_nodes', 0))
                            with col_b:
                                st.metric("Total Relationships", summary.get('total_relationships', 0))
                            with col_c:
                                st.metric("Conversations", summary.get('conversations', 0))
                            
                            if 'node_types' in summary:
                                st.markdown("#### Node Types Distribution")
                                st.json(summary['node_types'])
                        else:
                            st.error(f"Error fetching summary: HTTP {response.status_code}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        with col2:
            if st.button("📥 Export Graph Data"):
                with st.spinner("Exporting graph data..."):
                    try:
                        response = requests.get(
                            f"{BACKEND_URL}/knowledge-graph/export",
                            params={'format': 'json'},
                            timeout=30
                        )
                        if response.status_code == 200:
                            graph_data = response.json()
                            
                            # Create download button
                            st.download_button(
                                label="💾 Download JSON",
                                data=json.dumps(graph_data, indent=2),
                                file_name=f"knowledge_graph_{st.session_state.conversation_id}.json",
                                mime="application/json"
                            )
                            
                            st.success(f"Exported {len(graph_data.get('nodes', []))} nodes and {len(graph_data.get('relationships', []))} relationships")
                        else:
                            st.error(f"Error exporting data: HTTP {response.status_code}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        st.markdown("---")
        
        # Get conversation context
        st.markdown("### 🔍 Conversation Context")
        
        conv_id_input = st.text_input(
            "Conversation ID to retrieve:",
            value=st.session_state.conversation_id,
            key="conv_id_retrieve"
        )
        
        if st.button("🔎 Get Context"):
            with st.spinner("Retrieving conversation context..."):
                try:
                    response = requests.get(
                        f"{BACKEND_URL}/analyze/conversation/{conv_id_input}",
                        timeout=10
                    )
                    if response.status_code == 200:
                        context = response.json()
                        st.json(context)
                    else:
                        st.warning(f"No context found for conversation: {conv_id_input}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
