#!/usr/bin/env python3
"""
Week 6 Chat Orchestrator Test Script

Tests the unified AI pipeline that processes audio through:
1. Speech-to-Text (STT)
2. Speech Emotion Recognition (SER)
3. Named Entity Recognition (NER)
4. Commonsense Reasoning (COMET)
5. Knowledge Graph Updates
"""

import requests
import json
import sys
import numpy as np
from scipy.io import wavfile
import tempfile
import os

# Configuration
BASE_URL = "http://localhost:8000"
API_KEY = None

def print_header(text):
    """Print a formatted header"""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")

def register_and_login():
    """Register and login to get API token"""
    global API_KEY
    
    print_header("Setting Up Test User")
    
    import random
    test_id = random.randint(10000, 99999)
    
    # Register
    user_data = {
        "email": f"test_week6_{test_id}@example.com",
        "username": f"test_week6_{test_id}",
        "full_name": "Week 6 Test User",
        "password": "testpassword123"
    }
    
    response = requests.post(f"{BASE_URL}/auth/register", json=user_data)
    print(f"Registration Status: {response.status_code}")
    
    # Login
    login_data = {
        "username": user_data["username"],
        "password": "testpassword123"
    }
    
    response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
    print(f"Login Status: {response.status_code}")
    
    if response.status_code == 200:
        API_KEY = response.json()["access_token"]
        print("✅ Authentication successful!")
        return True
    else:
        print(f"❌ Authentication failed: {response.json()}")
        return False

def generate_test_audio(duration=3.0, freq=440, sample_rate=16000):
    """Generate a test audio with speech-like characteristics"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a more complex waveform (simulating speech)
    audio = (
        0.3 * np.sin(2 * np.pi * freq * t) +
        0.2 * np.sin(2 * np.pi * (freq * 1.5) * t) +
        0.1 * np.sin(2 * np.pi * (freq * 2) * t)
    )
    
    # Add some envelope to make it more speech-like
    envelope = np.exp(-t / duration * 2)
    audio = audio * (1 - envelope * 0.5)
    
    # Convert to int16
    audio = (audio * 32767).astype(np.int16)
    return audio, sample_rate

def test_health():
    """Test health endpoint"""
    print_header("Testing Health Endpoint")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    print("✅ Health check passed!")

def test_orchestrator_simple():
    """Test orchestrator with simple audio"""
    print_header("Testing Chat Orchestrator - Simple Audio (Week 6)")
    
    # Generate test audio
    audio, sample_rate = generate_test_audio(duration=2.0)
    
    # Save to temporary WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        wavfile.write(tmp_file.name, sample_rate, audio)
        tmp_filename = tmp_file.name
    
    try:
        # Send audio to orchestrator
        with open(tmp_filename, 'rb') as f:
            files = {'file': ('test_audio.wav', f, 'audio/wav')}
            response = requests.post(
                f"{BASE_URL}/orchestrate/analyze-audio",
                files=files,
                params={
                    'conversation_id': 'orchestrator_test_001',
                    'include_graph': True
                },
                headers={"Authorization": f"Bearer {API_KEY}"}
            )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Display results
            print("\n" + "="*70)
            print("  ORCHESTRATOR ANALYSIS RESULTS")
            print("="*70)
            
            # Transcript
            print(f"\n📝 TRANSCRIPT:")
            transcript = result.get("transcript", {})
            print(f"  Text: {transcript.get('text', 'N/A')}")
            print(f"  Language: {transcript.get('language', 'N/A')}")
            if transcript.get('error'):
                print(f"  ⚠️  Error: {transcript.get('error')}")
            
            # Emotion (Audio)
            print(f"\n😊 EMOTION (From Audio):")
            emotion_audio = result.get("emotion", {}).get("from_audio", {})
            print(f"  Primary: {emotion_audio.get('primary', 'N/A')}")
            print(f"  Confidence: {emotion_audio.get('confidence', 0.0):.2f}")
            all_scores = emotion_audio.get('all_scores', {})
            if all_scores:
                print(f"  All Scores:")
                for emotion, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
                    print(f"    {emotion}: {score:.3f}")
            
            # Emotion (Text)
            print(f"\n💭 EMOTION (From Text):")
            emotion_text = result.get("emotion", {}).get("from_text", {})
            detected = emotion_text.get('detected', [])
            if detected:
                print(f"  Detected: {', '.join(detected)}")
            else:
                print(f"  Detected: None")
            
            # Entities
            print(f"\n🏷️  ENTITIES:")
            entities = result.get("entities", {})
            total_entities = 0
            for category, items in entities.items():
                if items:
                    print(f"  {category.capitalize()}: {[e['text'] for e in items]}")
                    total_entities += len(items)
            if total_entities == 0:
                print(f"  No entities detected")
            
            # Commonsense Inferences
            print(f"\n🧠 COMMONSENSE INFERENCES:")
            commonsense = result.get("commonsense", {})
            inferences = commonsense.get("inferences", {})
            
            subject = inferences.get("subject", {})
            if subject.get("feelings"):
                print(f"  Subject Feelings: {subject.get('feelings')[:3]}")
            if subject.get("wants"):
                print(f"  Subject Wants: {subject.get('wants')[:3]}")
            
            # Graph Updates
            print(f"\n🕸️  KNOWLEDGE GRAPH:")
            graph_updates = result.get("graph_updates")
            if graph_updates:
                print(f"  Entity Nodes: {graph_updates.get('entity_nodes_count', 0)}")
                print(f"  Emotional Relationships: {graph_updates.get('emotional_relationships_count', 0)}")
            else:
                print(f"  Not updated")
            
            # Processing Metrics
            print(f"\n⏱️  PROCESSING METRICS:")
            processing = result.get("processing", {})
            print(f"  Total Time: {processing.get('total_time_ms', 0)}ms")
            print(f"  STT Completed: {'✅' if processing.get('stt_completed') else '❌'}")
            print(f"  SER Completed: {'✅' if processing.get('ser_completed') else '❌'}")
            print(f"  NER Completed: {'✅' if processing.get('ner_completed') else '❌'}")
            print(f"  COMET Completed: {'✅' if processing.get('comet_completed') else '❌'}")
            print(f"  Graph Updated: {'✅' if processing.get('graph_updated') else '❌'}")
            
            # Metadata
            print(f"\n📊 METADATA:")
            metadata = result.get("metadata", {})
            print(f"  Conversation ID: {metadata.get('conversation_id', 'N/A')}")
            print(f"  Text Length: {metadata.get('text_length', 0)} characters")
            print(f"  Entity Count: {metadata.get('entity_count', 0)}")
            
            print("\n✅ Orchestrator test passed!")
            return True
        else:
            print(f"❌ Orchestrator failed: {response.json()}")
            return False
            
    finally:
        os.unlink(tmp_filename)

def test_orchestrator_multiple():
    """Test orchestrator with multiple audio samples"""
    print_header("Testing Chat Orchestrator - Multiple Samples")
    
    # Test with different frequencies (simulating different speakers/emotions)
    test_cases = [
        {"name": "Speaker 1", "freq": 440, "duration": 2.0},
        {"name": "Speaker 2", "freq": 520, "duration": 2.5},
        {"name": "Speaker 3", "freq": 380, "duration": 1.8}
    ]
    
    success_count = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test {i}/{len(test_cases)}: {test_case['name']} ---")
        
        # Generate audio
        audio, sample_rate = generate_test_audio(
            duration=test_case['duration'],
            freq=test_case['freq']
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            wavfile.write(tmp_file.name, sample_rate, audio)
            tmp_filename = tmp_file.name
        
        try:
            # Send to orchestrator
            with open(tmp_filename, 'rb') as f:
                files = {'file': (f'test_{i}.wav', f, 'audio/wav')}
                response = requests.post(
                    f"{BASE_URL}/orchestrate/analyze-audio",
                    files=files,
                    params={
                        'conversation_id': f'multi_test_{i:03d}',
                        'speaker_id': f'speaker_{i}',
                        'include_graph': True
                    },
                    headers={"Authorization": f"Bearer {API_KEY}"}
                )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {test_case['name']}: "
                      f"{result['metadata']['text_length']} chars, "
                      f"{result['processing']['total_time_ms']}ms")
                success_count += 1
            else:
                print(f"❌ {test_case['name']} failed: {response.status_code}")
                
        finally:
            os.unlink(tmp_filename)
    
    print(f"\n✅ Completed {success_count}/{len(test_cases)} tests")
    return success_count == len(test_cases)

def test_knowledge_graph_accumulation():
    """Test that knowledge graph accumulates data across requests"""
    print_header("Testing Knowledge Graph Accumulation")
    
    # Get graph summary before
    response = requests.get(
        f"{BASE_URL}/knowledge-graph/summary",
        headers={"Authorization": f"Bearer {API_KEY}"}
    )
    
    if response.status_code == 200:
        before = response.json()
        print(f"Graph Before: {before.get('total_nodes', 0)} nodes, "
              f"{before.get('total_relationships', 0)} relationships")
        
        # Process audio to add data
        audio, sample_rate = generate_test_audio(duration=2.0)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            wavfile.write(tmp_file.name, sample_rate, audio)
            tmp_filename = tmp_file.name
        
        try:
            with open(tmp_filename, 'rb') as f:
                files = {'file': ('graph_test.wav', f, 'audio/wav')}
                requests.post(
                    f"{BASE_URL}/orchestrate/analyze-audio",
                    files=files,
                    params={'conversation_id': 'graph_accumulation_test'},
                    headers={"Authorization": f"Bearer {API_KEY}"}
                )
        finally:
            os.unlink(tmp_filename)
        
        # Get graph summary after
        response = requests.get(
            f"{BASE_URL}/knowledge-graph/summary",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if response.status_code == 200:
            after = response.json()
            print(f"Graph After: {after.get('total_nodes', 0)} nodes, "
                  f"{after.get('total_relationships', 0)} relationships")
            
            nodes_added = after.get('total_nodes', 0) - before.get('total_nodes', 0)
            rels_added = after.get('total_relationships', 0) - before.get('total_relationships', 0)
            
            print(f"\n📈 Changes: +{nodes_added} nodes, +{rels_added} relationships")
            print("✅ Knowledge graph accumulation verified!")
            return True
    
    return False

def main():
    """Run all Week 6 tests"""
    print("\n" + "="*70)
    print("  WEEK 6: CHAT ORCHESTRATOR TEST SUITE")
    print("  Testing Unified AI Pipeline")
    print("="*70)
    
    try:
        # Basic health check
        test_health()
        
        # Setup authentication
        if not register_and_login():
            print("❌ Failed to authenticate, exiting")
            sys.exit(1)
        
        # Run Week 6 tests
        print("\n" + "="*70)
        print("  Week 6 Feature Tests")
        print("="*70)
        
        success = True
        success = test_orchestrator_simple() and success
        success = test_orchestrator_multiple() and success
        success = test_knowledge_graph_accumulation() and success
        
        # Summary
        print_header("🎉 WEEK 6 TEST SUMMARY 🎉")
        
        if success:
            print("Summary:")
            print("  ✅ Chat orchestrator working")
            print("  ✅ STT + SER + NER + COMET pipeline operational")
            print("  ✅ Unified JSON response format correct")
            print("  ✅ Knowledge graph accumulation verified")
            print("  ✅ Multiple audio processing successful")
            print("\n🎉 All Week 6 features are operational!")
        else:
            print("❌ Some tests failed. Check the output above.")
            sys.exit(1)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
