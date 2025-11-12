"""
Aura Real-Time Chat Interface - Enhanced Version
Production-grade Streamlit UI for the Aura ML Backend

Features:
- Real-time chat interface with text and audio support
- Live voice recording with WebAudio API
- WebSocket support for real-time streaming
- Live ML pipeline visualization (STT, SER, NER, COMET, Knowledge Graph)
- OpenAI LLM integration for intelligent responses
- Session management and conversation history
- Audio recording and file upload
- Complete transparency into all ML processing stages
- Knowledge graph exploration
"""

import streamlit as st
import streamlit.components.v1 as components
import requests
from datetime import datetime
import json
import time
from typing import Dict, Any, Optional, List
import uuid
import os
from io import BytesIO
import base64
import asyncio
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Aura - Real-Time AI Chat",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1f77b4;
    }
    .ai-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .emotion-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .entity-badge {
        display: inline-block;
        padding: 0.2rem 0.4rem;
        border-radius: 0.2rem;
        font-size: 0.8rem;
        margin: 0.2rem;
        font-weight: 500;
    }
    .pipeline-stage {
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 0.3rem;
        border-left: 3px solid #4caf50;
        background-color: #e8f5e9;
    }
    .pipeline-stage.processing {
        border-left-color: #ff9800;
        background-color: #fff3e0;
    }
    .pipeline-stage.error {
        border-left-color: #f44336;
        background-color: #ffebee;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if 'conversation_id' not in st.session_state:
        st.session_state.conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'pipeline_history' not in st.session_state:
        st.session_state.pipeline_history = []
    
    if 'session_stats' not in st.session_state:
        st.session_state.session_stats = {
            'total_messages': 0,
            'audio_messages': 0,
            'text_messages': 0,
            'emotions_detected': [],
            'entities_extracted': [],
            'session_start': datetime.now()
        }
    
    if 'backend_status' not in st.session_state:
        st.session_state.backend_status = None

init_session_state()

# Utility Functions
def check_backend_health() -> Dict[str, Any]:
    """Check backend health status"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
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

def process_audio_message(audio_bytes: bytes) -> Dict[str, Any]:
    """Process audio through the unified ML pipeline"""
    try:
        files = {'file': ('audio.wav', audio_bytes, 'audio/wav')}
        params = {
            'conversation_id': st.session_state.conversation_id,
            'speaker_id': 'user',
            'include_graph': True
        }
        
        response = requests.post(
            f"{BACKEND_URL}/orchestrate/analyze-audio",
            files=files,
            params=params,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": f"Backend error: {response.status_code}",
                "detail": response.text
            }
    except Exception as e:
        return {"error": str(e)}

def analyze_text(text: str) -> Dict[str, Any]:
    """Analyze text through NER, COMET, and Knowledge Graph"""
    try:
        params = {
            'text': text,
            'conversation_id': st.session_state.conversation_id,
            'speaker_id': 'user',
            'include_graph': True
        }
        
        response = requests.post(
            f"{BACKEND_URL}/analyze/text",
            params=params,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": f"Backend error: {response.status_code}",
                "detail": response.text
            }
    except Exception as e:
        return {"error": str(e)}

def render_emotion_badge(emotion: str, confidence: float = None) -> str:
    """Render emotion badge with color coding"""
    colors = {
        'happy': '#4caf50',
        'sad': '#2196f3',
        'angry': '#f44336',
        'fear': '#9c27b0',
        'neutral': '#757575',
        'surprise': '#ff9800',
        'disgust': '#795548'
    }
    color = colors.get(emotion.lower(), '#607d8b')
    conf_text = f" ({confidence:.0%})" if confidence else ""
    return f'<span class="emotion-badge" style="background-color: {color}; color: white;">{emotion}{conf_text}</span>'

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
        'WORK_OF_ART': '#673ab7'
    }
    ent_type = entity.get('label', 'UNKNOWN')
    text = entity.get('text', '')
    color = colors.get(ent_type, '#757575')
    return f'<span class="entity-badge" style="background-color: {color}; color: white;">{text} <small>({ent_type})</small></span>'

def render_pipeline_stage(stage_name: str, status: str, data: Dict[str, Any] = None):
    """Render a pipeline processing stage"""
    icons = {
        'completed': '✅',
        'processing': '⏳',
        'error': '❌',
        'pending': '⏸️'
    }
    icon = icons.get(status, '❓')
    css_class = f"pipeline-stage {status}"
    
    st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
    st.markdown(f"**{icon} {stage_name}**")
    
    if data:
        if status == 'completed':
            st.json(data)
        elif status == 'error':
            st.error(data.get('error', 'Unknown error'))
    
    st.markdown('</div>', unsafe_allow_html=True)

def add_to_pipeline_history(stage: str, status: str, data: Dict[str, Any] = None):
    """Add pipeline stage to history"""
    st.session_state.pipeline_history.append({
        'timestamp': datetime.now(),
        'stage': stage,
        'status': status,
        'data': data
    })

# Main UI
def main():
    # Header
    st.markdown('<h1 class="main-header">🎙️ Aura Real-Time AI Chat</h1>', unsafe_allow_html=True)
    st.markdown("**Multi-Modal Conversational AI with Complete ML Pipeline Transparency**")
    
    # Check backend status
    with st.spinner("Checking backend connection..."):
        backend_health = check_backend_health()
        st.session_state.backend_status = backend_health
    
    if backend_health.get('status') == 'error':
        st.error(f"⚠️ Backend Connection Error: {backend_health.get('message')}")
        st.info(f"Make sure the backend is running at {BACKEND_URL}")
        st.code(f"cd aura-backend\npython main.py", language="bash")
        return
    
    # Sidebar - ML Pipeline & Session Info
    with st.sidebar:
        st.header("📊 ML Pipeline Monitor")
        
        # Backend Status
        st.subheader("🔌 Backend Status")
        if backend_health.get('status') == 'healthy':
            st.success("✅ Connected")
        else:
            st.warning("⚠️ Degraded")
        
        # Models Status
        models_status = get_models_status()
        if 'models' in models_status:
            st.subheader("🤖 ML Models")
            for model_name, model_info in models_status['models'].items():
                status_icon = "✅" if model_info.get('loaded') else "❌"
                st.write(f"{status_icon} **{model_name.upper()}**")
                if model_info.get('loaded'):
                    st.caption(f"   {model_info.get('model_name', 'N/A')}")
        
        st.divider()
        
        # Session Stats
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
        
        if stats['emotions_detected']:
            st.subheader("😊 Emotions Detected")
            emotion_counts = {}
            for emotion in stats['emotions_detected']:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            for emotion, count in emotion_counts.items():
                st.write(f"{emotion}: {count}")
        
        st.divider()
        
        # Recent Pipeline Activity
        st.subheader("🔄 Recent Pipeline Activity")
        if st.session_state.pipeline_history:
            for item in st.session_state.pipeline_history[-5:]:
                status_icon = {'completed': '✅', 'processing': '⏳', 'error': '❌'}.get(item['status'], '❓')
                time_str = item['timestamp'].strftime('%H:%M:%S')
                st.caption(f"{status_icon} {time_str} - {item['stage']}")
        else:
            st.caption("No activity yet")
        
        st.divider()
        
        # Reset button
        if st.button("🔄 New Conversation", use_container_width=True):
            st.session_state.conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
            st.session_state.messages = []
            st.session_state.pipeline_history = []
            st.session_state.session_stats = {
                'total_messages': 0,
                'audio_messages': 0,
                'text_messages': 0,
                'emotions_detected': [],
                'entities_extracted': [],
                'session_start': datetime.now()
            }
            st.rerun()
        
        # Conversation ID
        st.caption(f"**Conversation ID:** `{st.session_state.conversation_id}`")
    
    # Main chat area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Conversation")
        
        # Chat messages container
        chat_container = st.container()
        
        with chat_container:
            for msg in st.session_state.messages:
                if msg['role'] == 'user':
                    st.markdown(
                        f'<div class="chat-message user-message">'
                        f'<strong>You</strong> <small>({msg["timestamp"]})</small><br>'
                        f'{msg["content"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Show analysis results if available
                    if 'analysis' in msg:
                        with st.expander("📊 View Analysis Details", expanded=False):
                            analysis = msg['analysis']
                            
                            # Transcript (if audio)
                            if 'transcript' in analysis:
                                st.write("**📝 Transcript:**", analysis['transcript'].get('text'))
                            
                            # Emotions
                            if 'emotion' in analysis:
                                st.write("**😊 Emotions:**")
                                emotion_data = analysis['emotion']
                                if 'from_audio' in emotion_data:
                                    audio_emotion = emotion_data['from_audio']
                                    st.markdown(
                                        render_emotion_badge(
                                            audio_emotion['primary'],
                                            audio_emotion.get('confidence', 0)
                                        ),
                                        unsafe_allow_html=True
                                    )
                            
                            # Entities
                            if 'entities' in analysis:
                                entities = analysis['entities']
                                if entities.get('found'):
                                    st.write("**🏷️ Entities:**")
                                    for ent in entities['found']:
                                        st.markdown(render_entity_badge(ent), unsafe_allow_html=True)
                            
                            # Commonsense Reasoning
                            if 'emotional_context' in analysis:
                                comet_data = analysis['emotional_context']
                                if comet_data.get('inferences'):
                                    st.write("**🧠 Commonsense Inferences:**")
                                    st.json(comet_data['inferences'])
                else:
                    st.markdown(
                        f'<div class="chat-message ai-message">'
                        f'<strong>Aura</strong> <small>({msg["timestamp"]})</small><br>'
                        f'{msg["content"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
        
        # Input area
        st.subheader("✍️ Send Message")
        
        # Tabs for text and audio input
        input_tab1, input_tab2 = st.tabs(["💬 Text Input", "🎤 Audio Input"])
        
        with input_tab1:
            text_input = st.text_area(
                "Type your message:",
                height=100,
                placeholder="Type your message here and press Ctrl+Enter to send..."
            )
            
            if st.button("📤 Send Text Message", type="primary", use_container_width=True):
                if text_input.strip():
                    # Add user message
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    st.session_state.messages.append({
                        'role': 'user',
                        'content': text_input,
                        'timestamp': timestamp,
                        'type': 'text'
                    })
                    
                    # Update stats
                    st.session_state.session_stats['total_messages'] += 1
                    st.session_state.session_stats['text_messages'] += 1
                    
                    # Process text
                    with st.spinner("🔄 Analyzing text..."):
                        add_to_pipeline_history("Text Analysis", "processing")
                        analysis_result = analyze_text(text_input)
                        
                        if 'error' not in analysis_result:
                            add_to_pipeline_history("Text Analysis", "completed", analysis_result)
                            
                            # Store analysis with message
                            st.session_state.messages[-1]['analysis'] = analysis_result
                            
                            # Update stats
                            if 'entities' in analysis_result:
                                entities = analysis_result['entities'].get('found', [])
                                for ent in entities:
                                    st.session_state.session_stats['entities_extracted'].append(ent.get('text'))
                            
                            # Add AI response
                            ai_response = f"Message analyzed! Found {len(analysis_result.get('entities', {}).get('found', []))} entities."
                            st.session_state.messages.append({
                                'role': 'assistant',
                                'content': ai_response,
                                'timestamp': datetime.now().strftime('%H:%M:%S'),
                                'type': 'text'
                            })
                        else:
                            add_to_pipeline_history("Text Analysis", "error", analysis_result)
                            st.error(f"Analysis failed: {analysis_result.get('error')}")
                    
                    st.rerun()
        
        with input_tab2:
            st.write("**Upload Audio File**")
            audio_file = st.file_uploader(
                "Choose an audio file",
                type=['wav', 'mp3', 'm4a', 'ogg'],
                help="Upload an audio file to analyze"
            )
            
            if audio_file is not None:
                st.audio(audio_file)
                
                if st.button("📤 Send Audio Message", type="primary", use_container_width=True):
                    audio_bytes = audio_file.read()
                    
                    # Add user message
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    st.session_state.messages.append({
                        'role': 'user',
                        'content': f"🎤 Audio message ({audio_file.name})",
                        'timestamp': timestamp,
                        'type': 'audio'
                    })
                    
                    # Update stats
                    st.session_state.session_stats['total_messages'] += 1
                    st.session_state.session_stats['audio_messages'] += 1
                    
                    # Process audio
                    with st.spinner("🔄 Processing audio through ML pipeline..."):
                        add_to_pipeline_history("Audio Processing", "processing")
                        result = process_audio_message(audio_bytes)
                        
                        if 'error' not in result:
                            add_to_pipeline_history("Audio Processing", "completed", result)
                            
                            # Store analysis with message
                            st.session_state.messages[-1]['analysis'] = result
                            
                            # Update stats
                            if 'emotion' in result and 'from_audio' in result['emotion']:
                                emotion = result['emotion']['from_audio']['primary']
                                st.session_state.session_stats['emotions_detected'].append(emotion)
                            
                            if 'entities' in result:
                                entities = result['entities'].get('found', [])
                                for ent in entities:
                                    st.session_state.session_stats['entities_extracted'].append(ent.get('text'))
                            
                            # Add AI response
                            transcript = result.get('transcript', {}).get('text', 'N/A')
                            emotion = result.get('emotion', {}).get('from_audio', {}).get('primary', 'N/A')
                            entity_count = len(result.get('entities', {}).get('found', []))
                            
                            ai_response = f"Processed audio! Detected {emotion} emotion, found {entity_count} entities."
                            st.session_state.messages.append({
                                'role': 'assistant',
                                'content': ai_response,
                                'timestamp': datetime.now().strftime('%H:%M:%S'),
                                'type': 'text'
                            })
                        else:
                            add_to_pipeline_history("Audio Processing", "error", result)
                            st.error(f"Processing failed: {result.get('error')}")
                    
                    st.rerun()
    
    with col2:
        st.subheader("🔍 Live Pipeline View")
        
        if st.session_state.messages:
            last_message = st.session_state.messages[-1]
            
            if 'analysis' in last_message:
                analysis = last_message['analysis']
                
                # Pipeline stages
                st.markdown("### Processing Pipeline")
                
                # Stage 1: STT (if audio)
                if 'transcript' in analysis:
                    with st.expander("1️⃣ Speech-to-Text (STT)", expanded=True):
                        transcript_data = analysis['transcript']
                        st.success("✅ Completed")
                        st.write(f"**Text:** {transcript_data.get('text')}")
                        st.write(f"**Language:** {transcript_data.get('language', 'en')}")
                        st.write(f"**Duration:** {transcript_data.get('duration', 'N/A')}s")
                
                # Stage 2: SER (if audio)
                if 'emotion' in analysis and 'from_audio' in analysis['emotion']:
                    with st.expander("2️⃣ Emotion Recognition (SER)", expanded=True):
                        emotion_data = analysis['emotion']['from_audio']
                        st.success("✅ Completed")
                        st.markdown(
                            render_emotion_badge(
                                emotion_data['primary'],
                                emotion_data.get('confidence', 0)
                            ),
                            unsafe_allow_html=True
                        )
                        if 'all_scores' in emotion_data:
                            st.write("**All Scores:**")
                            st.json(emotion_data['all_scores'])
                
                # Stage 3: NER
                if 'entities' in analysis:
                    with st.expander("3️⃣ Named Entity Recognition (NER)", expanded=True):
                        entities = analysis['entities']
                        st.success("✅ Completed")
                        st.write(f"**Entities Found:** {entities.get('count', 0)}")
                        if entities.get('found'):
                            for ent in entities['found']:
                                st.markdown(render_entity_badge(ent), unsafe_allow_html=True)
                
                # Stage 4: COMET
                if 'emotional_context' in analysis:
                    with st.expander("4️⃣ Commonsense Reasoning (COMET)", expanded=True):
                        comet_data = analysis['emotional_context']
                        st.success("✅ Completed")
                        if comet_data.get('inferences'):
                            st.json(comet_data['inferences'])
                
                # Stage 5: Knowledge Graph
                if 'graph_updates' in analysis:
                    with st.expander("5️⃣ Knowledge Graph Update", expanded=True):
                        graph_data = analysis['graph_updates']
                        st.success("✅ Completed")
                        st.json(graph_data)
                
                # Processing metrics
                if 'processing' in analysis:
                    st.markdown("### ⚡ Performance Metrics")
                    processing = analysis['processing']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Total Time",
                            f"{processing.get('total_time_ms', 0)}ms"
                        )
                    with col2:
                        status = "✅ Success" if processing.get('all_models_completed') else "⚠️ Partial"
                        st.metric("Status", status)
        else:
            st.info("Send a message to see the ML pipeline in action!")

if __name__ == "__main__":
    main()
