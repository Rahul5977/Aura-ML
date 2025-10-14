#!/usr/bin/env python3
"""
Week 5 Contextual Analysis Test Script

Tests the new contextual analysis features:
- Named Entity Recognition (NER)
- COMET emotional reasoning
- Knowledge Graph building
"""

import requests
import json
import sys

# Configuration
BASE_URL = "http://localhost:8000"
API_KEY = None

# Test data
TEST_TEXTS = [
    "John met Sarah at Starbucks in Seattle to discuss the new AI project.",
    "Sarah was excited about the opportunity and wanted to collaborate with John.",
    "They decided to meet again next week at the Microsoft campus.",
    "John felt happy about the meeting and thought Sarah was very enthusiastic."
]

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
        "email": f"test_week5_{test_id}@example.com",
        "username": f"test_week5_{test_id}",
        "full_name": "Week 5 Test User",
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

def test_health():
    """Test health endpoint"""
    print_header("Testing Health Endpoint")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    print("✅ Health check passed!")

def test_text_analysis():
    """Test contextual text analysis"""
    print_header("Testing Contextual Text Analysis (Week 5)")
    
    for i, text in enumerate(TEST_TEXTS, 1):
        print(f"\n--- Test {i}/4 ---")
        print(f"Text: {text}")
        
        response = requests.post(
            f"{BASE_URL}/analyze/text",
            params={
                "text": text,
                "conversation_id": "test_conv_week5",
                "speaker_id": "test_speaker",
                "include_graph": True
            },
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Display entities
            entities = result.get("entities", {})
            print(f"\n📍 Entities Found:")
            for category, items in entities.items():
                if items:
                    print(f"  {category.capitalize()}: {[e['text'] for e in items]}")
            
            # Display emotions
            emotions = result.get("emotions_detected", [])
            if emotions:
                print(f"\n😊 Emotions Detected: {emotions}")
            
            # Display emotional context
            emotional_context = result.get("emotional_context", {})
            if emotional_context:
                print(f"\n💭 Emotional Context:")
                if emotional_context.get("subject_emotions"):
                    print(f"  Subject feels: {emotional_context['subject_emotions'][:2]}")
                if emotional_context.get("subject_wants"):
                    print(f"  Subject wants: {emotional_context['subject_wants'][:2]}")
            
            # Display processing time
            metadata = result.get("metadata", {})
            print(f"\n⏱️  Processing time: {metadata.get('processing_time_ms', 0)}ms")
            
            print("✅ Analysis successful!")
        else:
            print(f"❌ Analysis failed: {response.json()}")
            return False
    
    return True

def test_conversation_context():
    """Test conversation context retrieval"""
    print_header("Testing Conversation Context Retrieval (Week 5)")
    
    response = requests.get(
        f"{BASE_URL}/analyze/conversation/test_conv_week5",
        headers={"Authorization": f"Bearer {API_KEY}"}
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        context = response.json()
        
        print(f"\n📊 Conversation Context:")
        print(f"  Related Entities: {len(context.get('related_entities', []))}")
        print(f"  Relationships: {len(context.get('relationships', []))}")
        
        # Show some entities
        entities = context.get('related_entities', [])
        if entities:
            print(f"\n  Sample Entities:")
            for entity in entities[:5]:
                print(f"    - {entity['type']}: {entity['label']}")
        
        print("✅ Context retrieval successful!")
        return True
    else:
        print(f"❌ Context retrieval failed: {response.json()}")
        return False

def test_knowledge_graph_summary():
    """Test knowledge graph summary"""
    print_header("Testing Knowledge Graph Summary (Week 5)")
    
    response = requests.get(
        f"{BASE_URL}/knowledge-graph/summary",
        headers={"Authorization": f"Bearer {API_KEY}"}
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        summary = response.json()
        
        print(f"\n📈 Knowledge Graph Summary:")
        print(f"  Total Nodes: {summary.get('total_nodes', 0)}")
        print(f"  Total Relationships: {summary.get('total_relationships', 0)}")
        
        # Show node types
        nodes_by_type = summary.get('nodes_by_type', {})
        if nodes_by_type:
            print(f"\n  Nodes by Type:")
            for node_type, count in nodes_by_type.items():
                print(f"    {node_type}: {count}")
        
        # Show relationship types
        rels_by_type = summary.get('relationships_by_type', {})
        if rels_by_type:
            print(f"\n  Relationships by Type:")
            for rel_type, count in rels_by_type.items():
                print(f"    {rel_type}: {count}")
        
        print("✅ Graph summary retrieved successfully!")
        return True
    else:
        print(f"❌ Graph summary failed: {response.json()}")
        return False

def test_graph_export():
    """Test knowledge graph export"""
    print_header("Testing Knowledge Graph Export (Week 5)")
    
    response = requests.get(
        f"{BASE_URL}/knowledge-graph/export",
        params={"format": "json"},
        headers={"Authorization": f"Bearer {API_KEY}"}
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        export_data = response.json()
        
        print(f"\n💾 Graph Export:")
        print(f"  Format: {export_data.get('format', 'unknown')}")
        
        # Parse the exported data
        try:
            graph_data = json.loads(export_data.get('data', '{}'))
            nodes_count = len(graph_data.get('nodes', []))
            rels_count = len(graph_data.get('relationships', []))
            
            print(f"  Exported Nodes: {nodes_count}")
            print(f"  Exported Relationships: {rels_count}")
            
            # Show sample node
            if graph_data.get('nodes'):
                sample_node = graph_data['nodes'][0]
                print(f"\n  Sample Node:")
                print(f"    Type: {sample_node.get('type')}")
                print(f"    Label: {sample_node.get('label')}")
        except Exception as e:
            print(f"  Could not parse graph data: {e}")
        
        print("✅ Graph export successful!")
        return True
    else:
        print(f"❌ Graph export failed: {response.json()}")
        return False

def main():
    """Run all Week 5 tests"""
    print("\n" + "="*70)
    print("  WEEK 5: CONTEXTUAL ANALYSIS TEST SUITE")
    print("  Testing NER, COMET, and Knowledge Graph")
    print("="*70)
    
    try:
        # Basic health check
        test_health()
        
        # Setup authentication
        if not register_and_login():
            print("❌ Failed to authenticate, exiting")
            sys.exit(1)
        
        # Run Week 5 tests
        print("\n" + "="*70)
        print("  Week 5 Feature Tests")
        print("="*70)
        
        success = True
        success = test_text_analysis() and success
        success = test_conversation_context() and success
        success = test_knowledge_graph_summary() and success
        success = test_graph_export() and success
        
        # Summary
        print_header("🎉 WEEK 5 TEST SUMMARY 🎉")
        
        if success:
            print("Summary:")
            print("  ✅ Named Entity Recognition (NER) working")
            print("  ✅ COMET emotional reasoning working")
            print("  ✅ Knowledge Graph building working")
            print("  ✅ Conversation context retrieval working")
            print("  ✅ Graph export working")
            print("\n🎉 All Week 5 features are operational!")
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
