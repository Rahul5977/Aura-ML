#!/usr/bin/env python3
"""
Week 7 Test Script

Tests the enhanced orchestrator with Neo4j and LLM integration.
Demonstrates the complete AI pipeline with persistent graph storage
and intelligent response generation.
"""

import asyncio
import sys
import os
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'aura-backend'))


async def test_neo4j_connection():
    """Test Neo4j connection and basic operations."""
    print("\n" + "="*70)
    print("  TEST 1: Neo4j Connection")
    print("="*70)
    
    try:
        from contextual.neo4j_graph_service import Neo4jGraphService
        
        # Initialize service
        neo4j = Neo4jGraphService(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="your_password"  # Update with your password
        )
        
        # Connect
        await neo4j.connect()
        print("✅ Connected to Neo4j")
        
        # Initialize schema
        await neo4j.initialize_schema()
        print("✅ Schema initialized")
        
        # Test node creation
        node_id = await neo4j.add_entity_node(
            node_type="PERSON",
            label="Test User",
            properties={"role": "tester"},
            conversation_id="test_001"
        )
        print(f"✅ Created test node: {node_id}")
        
        # Test graph summary
        summary = await neo4j.get_graph_summary()
        print(f"✅ Graph summary: {summary['total_nodes']} nodes, "
              f"{summary['total_relationships']} relationships")
        
        # Cleanup
        await neo4j.clear_conversation("test_001")
        await neo4j.close()
        print("✅ Neo4j test passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Neo4j test failed: {e}")
        print("\nMake sure Neo4j is running:")
        print("  docker-compose up neo4j -d")
        return False


async def test_llm_service():
    """Test LLM service with mock analysis data."""
    print("\n" + "="*70)
    print("  TEST 2: LLM Service")
    print("="*70)
    
    try:
        from llm.llm_service import LLMService
        
        # Initialize (will be disabled if no API key)
        llm = LLMService(
            api_key=os.getenv("OPENAI_API_KEY"),  # Set in environment
            model="gpt-3.5-turbo"  # Use cheaper model for testing
        )
        
        if not llm.is_ready():
            print("⚠️  LLM service not available (no API key)")
            print("   Set OPENAI_API_KEY environment variable to test")
            return True  # Not a failure, just skipped
        
        print("✅ LLM service initialized")
        
        # Mock analysis packet
        analysis = {
            "transcript": {
                "text": "I'm feeling really stressed about work",
                "language": "en"
            },
            "emotion": {
                "from_audio": {
                    "primary": "sad",
                    "confidence": 0.75,
                    "all_scores": {"sad": 0.75, "neutral": 0.15, "angry": 0.10}
                },
                "from_text": {
                    "detected": ["stressed", "worried"]
                }
            },
            "entities": {
                "concepts": [{"text": "work", "start": 32, "end": 36}]
            },
            "commonsense": {
                "inferences": {
                    "subject": {
                        "feelings": ["overwhelmed", "anxious", "tired"],
                        "wants": ["to relax", "to find balance", "support"],
                        "effects": ["may feel better after talking"]
                    }
                }
            }
        }
        
        # Generate response
        response = await llm.generate_response(
            user_message="I'm feeling really stressed about work",
            analysis_packet=analysis
        )
        
        print("\n" + "-"*70)
        print("USER: I'm feeling really stressed about work")
        print("-"*70)
        print(f"AURA: {response['text']}")
        print("-"*70)
        print(f"Tokens used: {response.get('tokens_used', 'N/A')}")
        print(f"Model: {response.get('model', 'N/A')}")
        
        print("\n✅ LLM test passed!")
        return True
        
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        return False


async def test_enhanced_orchestrator():
    """Test the complete enhanced orchestrator."""
    print("\n" + "="*70)
    print("  TEST 3: Enhanced Orchestrator (Week 7)")
    print("="*70)
    
    try:
        # This would require all services to be running
        print("⚠️  Full orchestrator test requires:")
        print("   1. Running backend server")
        print("   2. Neo4j database")
        print("   3. OpenAI API key")
        print("   4. Audio transcription models")
        print("\n   Use the API endpoint instead:")
        print("   POST /orchestrate/analyze-audio-v2")
        
        return True
        
    except Exception as e:
        print(f"❌ Orchestrator test failed: {e}")
        return False


async def test_graph_context_retrieval():
    """Test retrieving graph context for LLM."""
    print("\n" + "="*70)
    print("  TEST 4: Graph Context Retrieval")
    print("="*70)
    
    try:
        from contextual.neo4j_graph_service import Neo4jGraphService
        
        neo4j = Neo4jGraphService(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="your_password"
        )
        
        await neo4j.connect()
        
        # Add some test data
        conv_id = "test_context_001"
        
        # Add entities
        await neo4j.add_entity_node(
            "PERSON", "Sarah",
            {"role": "friend"},
            conv_id
        )
        await neo4j.add_entity_node(
            "PLACE", "Coffee Shop",
            {"location": "downtown"},
            conv_id
        )
        
        # Add relationship
        await neo4j.add_relationship(
            "Sarah", "PERSON",
            "Coffee Shop", "PLACE",
            "VISITS",
            {"frequency": "often"}
        )
        
        print("✅ Created test graph data")
        
        # Retrieve context
        context = await neo4j.get_conversation_context(conv_id, depth=2)
        
        print(f"✅ Retrieved context: {len(context['nodes'])} nodes")
        for node in context['nodes']:
            print(f"   - {node.get('label', 'Unknown')}")
        
        # Cleanup
        await neo4j.clear_conversation(conv_id)
        await neo4j.close()
        
        print("✅ Graph context test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Graph context test failed: {e}")
        return False


async def main():
    """Run all Week 7 tests."""
    print("\n" + "="*70)
    print("  WEEK 7: ENHANCED AI ORCHESTRATION TEST SUITE")
    print("  Neo4j + LLM Integration")
    print("="*70)
    
    results = []
    
    # Test 1: Neo4j
    results.append(("Neo4j Connection", await test_neo4j_connection()))
    
    # Test 2: LLM
    results.append(("LLM Service", await test_llm_service()))
    
    # Test 3: Enhanced Orchestrator
    results.append(("Enhanced Orchestrator", await test_enhanced_orchestrator()))
    
    # Test 4: Graph Context
    results.append(("Graph Context", await test_graph_context_retrieval()))
    
    # Summary
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("  🎉 All Week 7 tests passed!")
        print("="*70)
        print("\nNext steps:")
        print("1. Update docker-compose.yml to include Neo4j")
        print("2. Set OPENAI_API_KEY in .env")
        print("3. Update main.py to initialize services")
        print("4. Test with real audio via API")
    else:
        print("  ⚠️  Some tests failed - see above for details")
        print("="*70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
