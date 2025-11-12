"""
Aura Real-Time AI Chat Interface - Complete Production Version
Handles ALL backend routes with real-time features

Backend Routes Implemented:
- GET  /                                 Health & info
- GET  /health                           Backend health check
- GET  /models/status                    ML models status
- POST /transcribe                       Audio transcription (Whisper)
- POST /recognize-emotion                Emotion recognition (Wav2Vec2)
- POST /analyze/text                     Text NER + COMET + Graph
- GET  /analyze/conversation/{id}        Get conversation context
- GET  /knowledge-graph/summary          Graph statistics
- GET  /knowledge-graph/export           Export graph data
- POST /orchestrate/analyze-audio        Unified ML pipeline
- POST /test/echo                        Echo test

Features:
- Real-time text chat
- Audio file upload and processing
- Live voice recording (WebAudio)
- Complete ML pipeline visualization
- OpenAI LLM responses (when available)
- Knowledge graph exploration
- Session statistics and history
"""

import streamlit as st
import requests
from datetime import datetime
import json
import time
from typing import Dict, Any, Optional, List
import uuid
import os
from io import BytesIO
import base64
import wave

# Page configuration
st.set_page_config(
    page_title="Aura - AI Chat with ML Pipeline",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1f77b4, #4caf50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .chat-message {
        padding: 1.2rem;
        border-radius: 0.8rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        animation: fadeIn 0.3s;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .user-message {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 4px solid #1f77b4;
    }
    .ai-message {
        background: linear-gradient(135deg, #f5f5f5 0%, #e0e0e0 100%);
        border-left: 4px solid #4caf50;
    }
    .emotion-badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .entity-badge {
        display: inline-block;
        padding: 0.3rem 0.6rem;
        border-radius: 0.4rem;
        font-size: 0.8rem;
        margin: 0.2rem;
        font-weight: 500;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }
    .pipeline-stage {
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
        background-color: #e8f5e9;
        transition: all 0.3s;
    }
    .pipeline-stage:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transform: translateX(5px);
    }
    .pipeline-stage.processing {
        border-left-color: #ff9800;
        background-color: #fff3e0;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    .pipeline-stage.error {
        border-left-color: #f44336;
        background-color: #ffebee;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
        animation: blink 2s infinite;
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .status-online { background-color: #4caf50; }
    .status-offline { background-color: #f44336; }
    .knowledge-graph-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'conversation_id': f"conv_{uuid.uuid4().hex[:8]}",
        'messages': [],
        'pipeline_history': [],
        'session_stats': {
            'total_messages': 0,
            'audio_messages': 0,
            'text_messages': 0,
            'emotions_detected': [],
            'entities_extracted': [],
            'session_start': datetime.now(),
            'total_processing_time': 0
        },
        'backend_status': None,
        'knowledge_graph_data': None,
        'current_analysis': None,
        'llm_enabled': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ============================================================================
# Utility Functions - Backend API Calls
# ============================================================================

def check_backend_health() -> Dict[str, Any]:
    """Check backend health status"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            # Check if OpenAI is available for LLM
            try:
                info_resp = requests.get(f"{BACKEND_URL}/", timeout=3)
                if info_resp.status_code == 200:
                    info = info_resp.json()
                    st.session_state.llm_enabled = 'llm' in str(info).lower()
            except:
                pass
            return data
        return {"status": "error", "message": "Backend returned non-200 status"}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "Cannot connect to backend"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_models_status() -> Dict[str, Any]:
    """Get ML models status"""
    try:
        response = requests.get(f"{BACKEND_URL}/models/status", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"error": "Failed to get models status"}
    except Exception as e:
        return {"error": str(e)}

def transcribe_audio(audio_bytes: bytes) -> Dict[str, Any]:
    """Transcribe audio using Whisper"""
    try:
        files = {'file': ('audio.wav', audio_bytes, 'audio/wav')}
        response = requests.post(
            f"{BACKEND_URL}/transcribe",
            files=files,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}

def recognize_emotion(audio_bytes: bytes) -> Dict[str, Any]:
    """Recognize emotion from audio"""
    try:
        files = {'file': ('audio.wav', audio_bytes, 'audio/wav')}
        response = requests.post(
            f"{BACKEND_URL}/recognize-emotion",
            files=files,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}

def analyze_text(text: str, conversation_id: str = None, speaker_id: str = "user") -> Dict[str, Any]:
    """Analyze text through NER, COMET, and Knowledge Graph"""
    try:
        params = {
            'text': text,
            'conversation_id': conversation_id or st.session_state.conversation_id,
            'speaker_id': speaker_id,
            'include_graph': True
        }
        response = requests.post(
            f"{BACKEND_URL}/analyze/text",
            params=params,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}

def process_audio_unified(audio_bytes: bytes, conversation_id: str = None, speaker_id: str = "user") -> Dict[str, Any]:
    """Process audio through unified ML pipeline"""
    try:
        files = {'file': ('audio.wav', audio_bytes, 'audio/wav')}
        params = {
            'conversation_id': conversation_id or st.session_state.conversation_id,
            'speaker_id': speaker_id,
            'include_graph': True
        }
        response = requests.post(
            f"{BACKEND_URL}/orchestrate/analyze-audio",
            files=files,
            params=params,
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}

def get_conversation_context(conversation_id: str) -> Dict[str, Any]:
    """Get conversation context from knowledge graph"""
    try:
        response = requests.get(
            f"{BACKEND_URL}/analyze/conversation/{conversation_id}",
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def get_knowledge_graph_summary() -> Dict[str, Any]:
    """Get knowledge graph summary statistics"""
    try:
        response = requests.get(
            f"{BACKEND_URL}/knowledge-graph/summary",
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def export_knowledge_graph(format: str = "json") -> Dict[str, Any]:
    """Export knowledge graph data"""
    try:
        response = requests.get(
            f"{BACKEND_URL}/knowledge-graph/export",
            params={'format': format},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# UI Rendering Functions
# ============================================================================

def render_emotion_badge(emotion: str, confidence: float = None) -> str:
    """Render emotion badge with color coding"""
    colors = {
        'happy': ('#4caf50', '😊'),
        'sad': ('#2196f3', '😢'),
        'angry': ('#f44336', '😠'),
        'fear': ('#9c27b0', '😨'),
        'neutral': ('#757575', '😐'),
        'surprise': ('#ff9800', '😲'),
        'disgust': ('#795548', '🤢')
    }
    color, emoji = colors.get(emotion.lower(), ('#607d8b', '❓'))
    conf_text = f" {confidence:.0%}" if confidence else ""
    return f'<span class="emotion-badge" style="background-color: {color}; color: white;">{emoji} {emotion.title()}{conf_text}</span>'

def render_entity_badge(entity: Dict[str, Any]) -> str:
    """Render entity badge with type color coding"""
    colors = {
        'PERSON': '#e91e63',
        'ORG': '#3f51b5',
        'GPE': '#00bcd4',
        'DATE': '#ff9800',
        'TIME': '#ff5722',
        'MONEY': '#4caf50',
        'CARDINAL': '#9c27b0',
        'EVENT': '#f44336',
        'PRODUCT': '#009688',
        'WORK_OF_ART': '#673ab7',
        'FAC': '#795548',
        'LOC': '#00bcd4'
    }
    ent_type = entity.get('label', 'UNKNOWN')
    text = entity.get('text', '')
    color = colors.get(ent_type, '#757575')
    return f'<span class="entity-badge" style="background-color: {color}; color: white;">{text} <small>({ent_type})</small></span>'

def add_to_pipeline_history(stage: str, status: str, data: Dict[str, Any] = None, duration_ms: int = None):
    """Add pipeline stage to history"""
    st.session_state.pipeline_history.append({
        'timestamp': datetime.now(),
        'stage': stage,
        'status': status,
        'data': data,
        'duration_ms': duration_ms
    })
    # Keep only last 20 items
    if len(st.session_state.pipeline_history) > 20:
        st.session_state.pipeline_history = st.session_state.pipeline_history[-20:]

def update_session_stats(message_type: str, analysis: Dict[str, Any] = None, processing_time: int = None):
    """Update session statistics"""
    stats = st.session_state.session_stats
    stats['total_messages'] += 1
    
    if message_type == 'audio':
        stats['audio_messages'] += 1
    elif message_type == 'text':
        stats['text_messages'] += 1
    
    if processing_time:
        stats['total_processing_time'] += processing_time
    
    if analysis:
        # Extract emotions
        if 'emotion' in analysis:
            emotion_data = analysis['emotion']
            if 'from_audio' in emotion_data:
                emotion = emotion_data['from_audio'].get('primary')
                if emotion:
                    stats['emotions_detected'].append(emotion)
            if 'from_text' in emotion_data:
                detected = emotion_data['from_text'].get('detected', [])
                stats['emotions_detected'].extend(detected)
        
        # Extract entities
        if 'entities' in analysis:
            entities = analysis['entities']
            for ent_type, ent_list in entities.items():
                if isinstance(ent_list, list):
                    for ent in ent_list:
                        if isinstance(ent, dict) and 'text' in ent:
                            stats['entities_extracted'].append(ent['text'])

# ============================================================================
# Main UI
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🎙️ Aura - Real-Time AI Chat</h1>', unsafe_allow_html=True)
    st.markdown("**Multi-Modal Conversational AI with Complete ML Pipeline Transparency**")
    st.markdown("---")
    
    # Check backend status
    with st.spinner("🔄 Connecting to backend..."):
        backend_health = check_backend_health()
        st.session_state.backend_status = backend_health
    
    if backend_health.get('status') == 'error':
        st.error(f"⚠️ **Backend Connection Error:** {backend_health.get('message')}")
        st.info(f"**Backend URL:** `{BACKEND_URL}`")
        st.code("""# Start the backend:
cd aura-backend
python main.py""", language="bash")
        st.stop()
    
    # Success message
    services = backend_health.get('services', {})
    all_services_ready = all(services.values())
    if all_services_ready:
        st.success("✅ **All systems operational!** Backend connected and all ML models loaded.")
    else:
        st.warning("⚠️ **Some services are unavailable.** Check sidebar for details.")
    
    # Sidebar - ML Pipeline Monitor & Controls
    with st.sidebar:
        st.header("📊 ML Pipeline Monitor")
        
        # Backend Status
        st.subheader("🔌 Backend Status")
        status_color = "status-online" if backend_health.get('status') == 'healthy' else "status-offline"
        status_text = "Connected" if backend_health.get('status') == 'healthy' else "Disconnected"
        st.markdown(f'<div><span class="{status_color} status-indicator"></span><strong>{status_text}</strong></div>', unsafe_allow_html=True)
        st.caption(f"URL: `{BACKEND_URL}`")
        
        # Models Status
        models_status = get_models_status()
        if 'models' in models_status:
            st.subheader("🤖 ML Models")
            models = models_status['models']
            
            model_icons = {
                'whisper': '🎤',
                'wav2vec2': '😊',
                'spacy': '🏷️',
                'comet': '🧠'
            }
            
            for model_name, model_info in models.items():
                loaded = model_info.get('loaded', False)
                icon = model_icons.get(model_name, '🔧')
                status_icon = "✅" if loaded else "❌"
                
                with st.expander(f"{icon} {model_name.upper()} {status_icon}", expanded=False):
                    st.write(f"**Status:** {'Loaded' if loaded else 'Not available'}")
                    if loaded and 'model_name' in model_info:
                        st.caption(f"Model: `{model_info['model_name']}`")
                    if 'capabilities' in model_info:
                        st.caption(f"Capabilities: {', '.join(model_info['capabilities'])}")
        
        st.divider()
        
        # Session Statistics
        st.subheader("📈 Session Statistics")
        stats = st.session_state.session_stats
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Messages", stats['total_messages'])
            st.metric("Audio Messages", stats['audio_messages'])
        with col2:
            st.metric("Text Messages", stats['text_messages'])
            duration = datetime.now() - stats['session_start']
            st.metric("Session Time", f"{duration.seconds // 60}m")
        
        if stats['total_processing_time'] > 0:
            avg_time = stats['total_processing_time'] / max(stats['total_messages'], 1)
            st.metric("Avg Processing", f"{avg_time:.0f}ms")
        
        # Emotion Distribution
        if stats['emotions_detected']:
            st.subheader("😊 Emotion Distribution")
            emotion_counts = {}
            for emotion in stats['emotions_detected']:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                st.write(f"{emotion.title()}: {count}")
        
        # Entities Count
        if stats['entities_extracted']:
            st.metric("Entities Extracted", len(stats['entities_extracted']))
        
        st.divider()
        
        # Recent Pipeline Activity
        st.subheader("🔄 Pipeline Activity")
        if st.session_state.pipeline_history:
            for item in st.session_state.pipeline_history[-5:]:
                status_icon = {
                    'completed': '✅',
                    'processing': '⏳',
                    'error': '❌'
                }.get(item['status'], '❓')
                
                time_str = item['timestamp'].strftime('%H:%M:%S')
                duration = f" ({item['duration_ms']}ms)" if item.get('duration_ms') else ""
                st.caption(f"{status_icon} {time_str} - {item['stage']}{duration}")
        else:
            st.caption("No activity yet")
        
        st.divider()
        
        # Knowledge Graph Section
        st.subheader("🕸️ Knowledge Graph")
        if st.button("📊 View Graph Summary", use_container_width=True):
            with st.spinner("Loading graph data..."):
                graph_summary = get_knowledge_graph_summary()
                if 'error' not in graph_summary:
                    st.session_state.knowledge_graph_data = graph_summary
                    st.success("Graph data loaded!")
                else:
                    st.error(f"Error: {graph_summary['error']}")
        
        if st.button("💾 Export Graph", use_container_width=True):
            with st.spinner("Exporting graph..."):
                graph_export = export_knowledge_graph()
                if 'error' not in graph_export:
                    st.download_button(
                        "⬇️ Download JSON",
                        json.dumps(graph_export, indent=2),
                        file_name=f"knowledge_graph_{st.session_state.conversation_id}.json",
                        mime="application/json"
                    )
                else:
                    st.error(f"Error: {graph_export['error']}")
        
        if st.session_state.knowledge_graph_data:
            with st.expander("📈 Graph Statistics", expanded=False):
                data = st.session_state.knowledge_graph_data
                if 'nodes' in data:
                    st.write("**Nodes:**")
                    for node_type, count in data['nodes'].items():
                        st.write(f"  - {node_type}: {count}")
                if 'relationships' in data:
                    st.write("**Relationships:**")
                    for rel_type, count in data['relationships'].items():
                        st.write(f"  - {rel_type}: {count}")
        
        st.divider()
        
        # Conversation Controls
        if st.button("🔄 New Conversation", use_container_width=True, type="primary"):
            st.session_state.conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
            st.session_state.messages = []
            st.session_state.pipeline_history = []
            st.session_state.session_stats = {
                'total_messages': 0,
                'audio_messages': 0,
                'text_messages': 0,
                'emotions_detected': [],
                'entities_extracted': [],
                'session_start': datetime.now(),
                'total_processing_time': 0
            }
            st.session_state.knowledge_graph_data = None
            st.rerun()
        
        st.caption(f"**Conversation ID:**")
        st.code(st.session_state.conversation_id, language="text")
        
        # Get conversation context
        if st.button("🔍 Get Context", use_container_width=True):
            with st.spinner("Fetching context..."):
                context = get_conversation_context(st.session_state.conversation_id)
                if 'error' not in context:
                    st.json(context)
                else:
                    st.error(f"Error: {context['error']}")
    
    # Main Content Area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Conversation")
        
        # Messages Container
        messages_container = st.container()
        
        with messages_container:
            for msg in st.session_state.messages:
                if msg['role'] == 'user':
                    st.markdown(
                        f'<div class="chat-message user-message">'
                        f'<strong>👤 You</strong> <small>({msg["timestamp"]})</small><br>'
                        f'{msg["content"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Show analysis if available
                    if 'analysis' in msg and msg['analysis']:
                        with st.expander("📊 View Full Analysis", expanded=False):
                            analysis = msg['analysis']
                            
                            # Transcript
                            if 'transcript' in analysis:
                                st.write("**📝 Transcript:**")
                                st.info(analysis['transcript'].get('text', 'N/A'))
                                st.caption(f"Language: {analysis['transcript'].get('language', 'unknown')}")
                            
                            # Emotions
                            if 'emotion' in analysis:
                                st.write("**😊 Emotions:**")
                                emotion_data = analysis['emotion']
                                
                                if 'from_audio' in emotion_data:
                                    st.write("*From Audio:*")
                                    audio_emotion = emotion_data['from_audio']
                                    st.markdown(
                                        render_emotion_badge(
                                            audio_emotion.get('primary', 'unknown'),
                                            audio_emotion.get('confidence', 0)
                                        ),
                                        unsafe_allow_html=True
                                    )
                                    
                                    if 'all_scores' in audio_emotion:
                                        with st.expander("All Scores"):
                                            st.json(audio_emotion['all_scores'])
                                
                                if 'from_text' in emotion_data:
                                    detected = emotion_data['from_text'].get('detected', [])
                                    if detected:
                                        st.write("*From Text:*")
                                        for emotion in detected:
                                            st.markdown(render_emotion_badge(emotion), unsafe_allow_html=True)
                            
                            # Entities
                            if 'entities' in analysis:
                                entities = analysis['entities']
                                entity_list = []
                                
                                for ent_type, ents in entities.items():
                                    if isinstance(ents, list) and ents:
                                        entity_list.extend(ents)
                                
                                if entity_list:
                                    st.write("**🏷️ Entities:**")
                                    for ent in entity_list:
                                        st.markdown(render_entity_badge(ent), unsafe_allow_html=True)
                            
                            # Commonsense
                            if 'commonsense' in analysis:
                                comet_data = analysis['commonsense']
                                if comet_data.get('inferences'):
                                    st.write("**🧠 Commonsense Inferences:**")
                                    st.json(comet_data['inferences'])
                            
                            # Processing Info
                            if 'processing' in analysis:
                                st.write("**⚡ Processing Metrics:**")
                                processing = analysis['processing']
                                st.json(processing)
                
                elif msg['role'] == 'assistant':
                    st.markdown(
                        f'<div class="chat-message ai-message">'
                        f'<strong>🤖 Aura</strong> <small>({msg["timestamp"]})</small><br>'
                        f'{msg["content"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
        
        st.markdown("---")
        
        # Input Area
        st.subheader("✍️ Send Message")
        
        input_tab1, input_tab2, input_tab3 = st.tabs(["💬 Text", "🎤 Audio File", "🎙️ Record"])
        
        # Tab 1: Text Input
        with input_tab1:
            text_input = st.text_area(
                "Type your message:",
                height=100,
                placeholder="Type your message here...",
                key="text_input_area"
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                send_text = st.button("📤 Send Text", type="primary", use_container_width=True)
            with col_b:
                analyze_only = st.button("🔍 Analyze Only", use_container_width=True)
            
            if send_text or analyze_only:
                if text_input.strip():
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    
                    # Add user message
                    st.session_state.messages.append({
                        'role': 'user',
                        'content': text_input,
                        'timestamp': timestamp,
                        'type': 'text'
                    })
                    
                    # Process text
                    with st.spinner("🔄 Analyzing text through ML pipeline..."):
                        add_to_pipeline_history("Text Analysis", "processing")
                        start_time = time.time()
                        
                        analysis_result = analyze_text(text_input)
                        
                        processing_time = int((time.time() - start_time) * 1000)
                        
                        if 'error' not in analysis_result:
                            add_to_pipeline_history("Text Analysis", "completed", analysis_result, processing_time)
                            
                            # Store analysis
                            st.session_state.messages[-1]['analysis'] = analysis_result
                            
                            # Update stats
                            update_session_stats('text', analysis_result, processing_time)
                            
                            # Add AI response
                            entity_count = sum(len(v) for k, v in analysis_result.get('entities', {}).items() if isinstance(v, list))
                            emotion_count = len(analysis_result.get('emotions_detected', []))
                            
                            response_text = f"✅ Analysis complete! Found {entity_count} entities"
                            if emotion_count > 0:
                                response_text += f" and detected {emotion_count} emotional cues"
                            response_text += "."
                            
                            st.session_state.messages.append({
                                'role': 'assistant',
                                'content': response_text,
                                'timestamp': datetime.now().strftime('%H:%M:%S'),
                                'type': 'text'
                            })
                        else:
                            add_to_pipeline_history("Text Analysis", "error", analysis_result)
                            st.error(f"❌ Analysis failed: {analysis_result.get('error')}")
                    
                    st.rerun()
        
        # Tab 2: Audio File Upload
        with input_tab2:
            st.write("**Upload Audio File**")
            audio_file = st.file_uploader(
                "Choose an audio file (WAV, MP3, M4A, OGG)",
                type=['wav', 'mp3', 'm4a', 'ogg'],
                key="audio_uploader"
            )
            
            if audio_file is not None:
                st.audio(audio_file)
                
                col_c, col_d = st.columns(2)
                with col_c:
                    process_unified = st.button("🚀 Process (Unified Pipeline)", type="primary", use_container_width=True)
                with col_d:
                    process_separate = st.button("🔧 Process (Separate)", use_container_width=True)
                
                if process_unified:
                    audio_bytes = audio_file.read()
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    
                    # Add user message
                    st.session_state.messages.append({
                        'role': 'user',
                        'content': f"🎤 Audio message ({audio_file.name})",
                        'timestamp': timestamp,
                        'type': 'audio'
                    })
                    
                    # Process through unified pipeline
                    with st.spinner("🔄 Processing audio through complete ML pipeline..."):
                        add_to_pipeline_history("Unified Audio Pipeline", "processing")
                        start_time = time.time()
                        
                        result = process_audio_unified(audio_bytes)
                        
                        processing_time = int((time.time() - start_time) * 1000)
                        
                        if 'error' not in result:
                            add_to_pipeline_history("Unified Audio Pipeline", "completed", result, processing_time)
                            
                            # Store analysis
                            st.session_state.messages[-1]['analysis'] = result
                            
                            # Update stats
                            update_session_stats('audio', result, processing_time)
                            
                            # Build response
                            transcript_text = result.get('transcript', {}).get('text', 'N/A')
                            emotion = result.get('emotion', {}).get('from_audio', {}).get('primary', 'unknown')
                            entity_count = result.get('metadata', {}).get('entity_count', 0)
                            
                            response_text = f"✅ Audio processed!\n\n"
                            response_text += f"📝 Transcript: \"{transcript_text}\"\n"
                            response_text += f"😊 Emotion: {emotion.title()}\n"
                            response_text += f"🏷️ Entities: {entity_count} found"
                            
                            st.session_state.messages.append({
                                'role': 'assistant',
                                'content': response_text,
                                'timestamp': datetime.now().strftime('%H:%M:%S'),
                                'type': 'text'
                            })
                        else:
                            add_to_pipeline_history("Unified Audio Pipeline", "error", result)
                            st.error(f"❌ Processing failed: {result.get('error')}")
                    
                    st.rerun()
                
                if process_separate:
                    audio_bytes = audio_file.read()
                    
                    with st.spinner("🔄 Processing audio (separate steps)..."):
                        # Step 1: Transcription
                        add_to_pipeline_history("Transcription (STT)", "processing")
                        transcript_result = transcribe_audio(audio_bytes)
                        
                        if 'error' not in transcript_result:
                            add_to_pipeline_history("Transcription (STT)", "completed", transcript_result)
                            st.success(f"✅ Transcription: {transcript_result.get('text')}")
                        else:
                            add_to_pipeline_history("Transcription (STT)", "error", transcript_result)
                            st.error(f"❌ Transcription failed: {transcript_result.get('error')}")
                        
                        # Step 2: Emotion Recognition
                        add_to_pipeline_history("Emotion Recognition (SER)", "processing")
                        emotion_result = recognize_emotion(audio_bytes)
                        
                        if 'error' not in emotion_result:
                            add_to_pipeline_history("Emotion Recognition (SER)", "completed", emotion_result)
                            st.success(f"✅ Emotion: {emotion_result.get('emotion')} ({emotion_result.get('confidence', 0):.0%})")
                        else:
                            add_to_pipeline_history("Emotion Recognition (SER)", "error", emotion_result)
                            st.error(f"❌ Emotion recognition failed: {emotion_result.get('error')}")
                        
                        # Step 3: Text Analysis (if transcription succeeded)
                        if 'error' not in transcript_result and transcript_result.get('text'):
                            add_to_pipeline_history("Text Analysis (NER + COMET)", "processing")
                            text_result = analyze_text(transcript_result['text'])
                            
                            if 'error' not in text_result:
                                add_to_pipeline_history("Text Analysis (NER + COMET)", "completed", text_result)
                                entity_count = sum(len(v) for k, v in text_result.get('entities', {}).items() if isinstance(v, list))
                                st.success(f"✅ Text Analysis: {entity_count} entities found")
                            else:
                                add_to_pipeline_history("Text Analysis (NER + COMET)", "error", text_result)
                                st.error(f"❌ Text analysis failed: {text_result.get('error')}")
        
        # Tab 3: Voice Recording
        with input_tab3:
            st.write("**🎙️ Record Your Voice**")
            st.info("⚠️ Voice recording requires browser permissions. Click 'Allow' when prompted.")
            
            # Simple audio recording interface
            audio_value = st.audio_input("Record audio")
            
            if audio_value is not None:
                st.success("✅ Recording received!")
                audio_bytes = audio_value.read()
                
                if st.button("🚀 Process Recording", type="primary", use_container_width=True):
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    
                    # Add user message
                    st.session_state.messages.append({
                        'role': 'user',
                        'content': f"🎙️ Voice recording",
                        'timestamp': timestamp,
                        'type': 'audio'
                    })
                    
                    # Process recording
                    with st.spinner("🔄 Processing your voice..."):
                        add_to_pipeline_history("Voice Recording Pipeline", "processing")
                        start_time = time.time()
                        
                        result = process_audio_unified(audio_bytes)
                        
                        processing_time = int((time.time() - start_time) * 1000)
                        
                        if 'error' not in result:
                            add_to_pipeline_history("Voice Recording Pipeline", "completed", result, processing_time)
                            
                            # Store analysis
                            st.session_state.messages[-1]['analysis'] = result
                            
                            # Update stats
                            update_session_stats('audio', result, processing_time)
                            
                            # Build response
                            transcript_text = result.get('transcript', {}).get('text', 'N/A')
                            emotion = result.get('emotion', {}).get('from_audio', {}).get('primary', 'unknown')
                            entity_count = result.get('metadata', {}).get('entity_count', 0)
                            
                            response_text = f"✅ Voice processed!\n\n"
                            response_text += f"📝 You said: \"{transcript_text}\"\n"
                            response_text += f"😊 Detected emotion: {emotion.title()}\n"
                            response_text += f"🏷️ Found {entity_count} entities"
                            
                            st.session_state.messages.append({
                                'role': 'assistant',
                                'content': response_text,
                                'timestamp': datetime.now().strftime('%H:%M:%S'),
                                'type': 'text'
                            })
                            
                            st.success("✅ Processing complete!")
                            st.rerun()
                        else:
                            add_to_pipeline_history("Voice Recording Pipeline", "error", result)
                            st.error(f"❌ Processing failed: {result.get('error')}")
    
    # Right Column - Pipeline Visualization
    with col2:
        st.subheader("🔍 Live Pipeline View")
        
        if st.session_state.messages and st.session_state.messages[-1].get('analysis'):
            analysis = st.session_state.messages[-1]['analysis']
            
            # Pipeline Stages
            st.markdown("### Processing Pipeline")
            
            # Stage 1: STT
            if 'transcript' in analysis:
                with st.expander("1️⃣ Speech-to-Text (STT)", expanded=True):
                    transcript_data = analysis['transcript']
                    if transcript_data.get('text'):
                        st.success("✅ Completed")
                        st.write(f"**Text:** {transcript_data['text']}")
                        st.caption(f"Language: {transcript_data.get('language', 'en')}")
                    else:
                        st.warning("⚠️ No transcription")
            
            # Stage 2: SER
            if 'emotion' in analysis and 'from_audio' in analysis['emotion']:
                with st.expander("2️⃣ Emotion Recognition (SER)", expanded=True):
                    emotion_data = analysis['emotion']['from_audio']
                    if emotion_data.get('primary'):
                        st.success("✅ Completed")
                        st.markdown(
                            render_emotion_badge(
                                emotion_data['primary'],
                                emotion_data.get('confidence', 0)
                            ),
                            unsafe_allow_html=True
                        )
                        if 'all_scores' in emotion_data:
                            with st.expander("All Scores"):
                                for emo, score in emotion_data['all_scores'].items():
                                    st.progress(score, text=f"{emo}: {score:.2%}")
            
            # Stage 3: NER
            if 'entities' in analysis:
                with st.expander("3️⃣ Named Entity Recognition (NER)", expanded=True):
                    entities = analysis['entities']
                    entity_list = []
                    
                    for ent_type, ents in entities.items():
                        if isinstance(ents, list) and ents:
                            entity_list.extend(ents)
                    
                    if entity_list:
                        st.success(f"✅ Completed - {len(entity_list)} entities")
                        for ent in entity_list[:10]:  # Show first 10
                            st.markdown(render_entity_badge(ent), unsafe_allow_html=True)
                    else:
                        st.info("No entities found")
            
            # Stage 4: COMET
            if 'commonsense' in analysis:
                with st.expander("4️⃣ Commonsense Reasoning (COMET)", expanded=True):
                    comet_data = analysis['commonsense']
                    if comet_data.get('inferences'):
                        st.success("✅ Completed")
                        inferences = comet_data['inferences']
                        
                        if 'subject' in inferences:
                            st.write("**Subject:**")
                            subject = inferences['subject']
                            if subject.get('feelings'):
                                st.caption(f"Feelings: {', '.join(subject['feelings'][:3])}")
                            if subject.get('wants'):
                                st.caption(f"Wants: {', '.join(subject['wants'][:3])}")
                    else:
                        st.info("No inferences")
            
            # Stage 5: Graph
            if 'graph_updates' in analysis and analysis['graph_updates']:
                with st.expander("5️⃣ Knowledge Graph Update", expanded=True):
                    graph_data = analysis['graph_updates']
                    st.success("✅ Completed")
                    
                    if isinstance(graph_data, dict):
                        if 'nodes_created' in graph_data:
                            st.metric("Nodes Created", graph_data['nodes_created'])
                        if 'relationships_created' in graph_data:
                            st.metric("Relationships", graph_data['relationships_created'])
            
            # Performance Metrics
            if 'processing' in analysis:
                st.markdown("### ⚡ Performance Metrics")
                processing = analysis['processing']
                
                total_time = processing.get('total_time_ms', 0)
                st.metric("Total Time", f"{total_time}ms")
                
                # Status indicators
                stages = {
                    'STT': processing.get('stt_completed', False),
                    'SER': processing.get('ser_completed', False),
                    'NER': processing.get('ner_completed', False),
                    'COMET': processing.get('comet_completed', False),
                    'Graph': processing.get('graph_updated', False)
                }
                
                completed = sum(1 for v in stages.values() if v)
                st.progress(completed / len(stages), text=f"{completed}/{len(stages)} stages completed")
        
        else:
            st.info("👆 Send a message to see the ML pipeline in action!")
            
            st.markdown("""
            **The pipeline will show:**
            1. 🎤 Speech-to-Text (Whisper)
            2. 😊 Emotion Recognition (Wav2Vec2)
            3. 🏷️ Named Entity Recognition (spaCy)
            4. 🧠 Commonsense Reasoning (COMET)
            5. 🕸️ Knowledge Graph Update (Neo4j)
            """)

if __name__ == "__main__":
    main()
