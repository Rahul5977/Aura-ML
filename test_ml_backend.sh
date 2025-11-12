#!/bin/bash

# Aura ML Backend - Complete Test Script
# Tests all ML endpoints without authentication

echo "======================================================================"
echo "  AURA ML BACKEND - COMPREHENSIVE TEST SUITE"
echo "======================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Base URL
BASE_URL="http://localhost:8000"

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Function to print test result
print_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ PASS${NC}: $2"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}❌ FAIL${NC}: $2"
        ((TESTS_FAILED++))
    fi
    echo ""
}

# Function to test endpoint
test_endpoint() {
    local method=$1
    local endpoint=$2
    local description=$3
    local extra_args="${@:4}"
    
    echo -e "${BLUE}Testing:${NC} $description"
    echo "Endpoint: $method $endpoint"
    
    response=$(curl -s -w "\n%{http_code}" -X $method "$BASE_URL$endpoint" $extra_args)
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)
    
    if [ "$http_code" -eq 200 ] || [ "$http_code" -eq 201 ]; then
        print_result 0 "$description"
        echo "Response preview: $(echo $body | jq -r '.' 2>/dev/null | head -n 5)"
    else
        print_result 1 "$description (HTTP $http_code)"
        echo "Error: $body"
    fi
}

echo "======================================================================"
echo " 1. HEALTH & INFO ENDPOINTS"
echo "======================================================================"
echo ""

# Test 1: Root endpoint
test_endpoint GET "/" "Root endpoint - API information"

# Test 2: Health check
test_endpoint GET "/health" "Health check endpoint"

# Test 3: Models status
test_endpoint GET "/models/status" "Models status endpoint"

echo "======================================================================"
echo " 2. AUDIO PROCESSING ENDPOINTS"
echo "======================================================================"
echo ""

# Create a test audio file if it doesn't exist
if [ ! -f "test_audio.wav" ]; then
    echo "Creating test audio file..."
    # Create a simple sine wave audio file using ffmpeg (if available)
    if command -v ffmpeg &> /dev/null; then
        ffmpeg -f lavfi -i "sine=frequency=440:duration=3" -ar 16000 test_audio.wav -y 2>/dev/null
        echo "✅ Test audio file created"
    else
        echo "⚠️  ffmpeg not found. Some audio tests will be skipped."
        echo "   Install ffmpeg: brew install ffmpeg"
    fi
fi

# Test 4: Transcribe audio
if [ -f "test_audio.wav" ]; then
    echo -e "${BLUE}Testing:${NC} Audio transcription"
    echo "Endpoint: POST /transcribe"
    
    response=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/transcribe" \
        -F "file=@test_audio.wav")
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)
    
    if [ "$http_code" -eq 200 ]; then
        print_result 0 "Audio transcription"
        echo "$body" | jq '.'
    else
        print_result 1 "Audio transcription (HTTP $http_code)"
        echo "Error: $body"
    fi
else
    echo "⚠️  Skipping transcription test (no audio file)"
    echo ""
fi

# Test 5: Emotion recognition
if [ -f "test_audio.wav" ]; then
    echo -e "${BLUE}Testing:${NC} Emotion recognition"
    echo "Endpoint: POST /recognize-emotion"
    
    response=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/recognize-emotion" \
        -F "file=@test_audio.wav")
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)
    
    if [ "$http_code" -eq 200 ]; then
        print_result 0 "Emotion recognition"
        echo "$body" | jq '.'
    else
        print_result 1 "Emotion recognition (HTTP $http_code)"
        echo "Error: $body"
    fi
else
    echo "⚠️  Skipping emotion recognition test (no audio file)"
    echo ""
fi

echo "======================================================================"
echo " 3. TEXT ANALYSIS ENDPOINTS"
echo "======================================================================"
echo ""

# Test 6: Text analysis
TEST_TEXT="I am meeting Sarah tomorrow in Mumbai to discuss the project"
echo -e "${BLUE}Testing:${NC} Text analysis with NER and COMET"
echo "Endpoint: POST /analyze/text"
echo "Test text: '$TEST_TEXT'"

response=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/analyze/text" \
    -G --data-urlencode "text=$TEST_TEXT" \
    --data-urlencode "conversation_id=test_conv_001" \
    --data-urlencode "speaker_id=test_speaker_001" \
    --data-urlencode "include_graph=true")
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n-1)

if [ "$http_code" -eq 200 ]; then
    print_result 0 "Text analysis"
    echo "$body" | jq '.'
else
    print_result 1 "Text analysis (HTTP $http_code)"
    echo "Error: $body"
fi

echo "======================================================================"
echo " 4. UNIFIED ML PIPELINE"
echo "======================================================================"
echo ""

# Test 7: Complete pipeline with audio
if [ -f "test_audio.wav" ]; then
    echo -e "${BLUE}Testing:${NC} Unified ML pipeline (orchestrator)"
    echo "Endpoint: POST /orchestrate/analyze-audio"
    echo "Processing: STT → SER → NER → COMET → Knowledge Graph"
    
    response=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/orchestrate/analyze-audio" \
        -F "file=@test_audio.wav" \
        -F "conversation_id=test_conv_pipeline" \
        -F "speaker_id=test_speaker" \
        -F "include_graph=true")
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)
    
    if [ "$http_code" -eq 200 ]; then
        print_result 0 "Unified ML pipeline"
        echo "$body" | jq '.' | head -n 30
        echo "... (truncated)"
    else
        print_result 1 "Unified ML pipeline (HTTP $http_code)"
        echo "Error: $body"
    fi
else
    echo "⚠️  Skipping pipeline test (no audio file)"
    echo ""
fi

echo "======================================================================"
echo " 5. KNOWLEDGE GRAPH ENDPOINTS"
echo "======================================================================"
echo ""

# Test 8: Get conversation context
test_endpoint GET "/analyze/conversation/test_conv_001" "Get conversation context from graph"

# Test 9: Knowledge graph summary
test_endpoint GET "/knowledge-graph/summary" "Knowledge graph summary statistics"

# Test 10: Export knowledge graph
test_endpoint GET "/knowledge-graph/export?format=json" "Export knowledge graph"

echo "======================================================================"
echo " 6. UTILITY ENDPOINTS"
echo "======================================================================"
echo ""

# Test 11: Echo test
echo -e "${BLUE}Testing:${NC} Echo test endpoint"
echo "Endpoint: POST /test/echo"

response=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/test/echo" \
    -H "Content-Type: application/json" \
    -d '{"test": "hello", "number": 42}')
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n-1)

if [ "$http_code" -eq 200 ]; then
    print_result 0 "Echo test endpoint"
    echo "$body" | jq '.'
else
    print_result 1 "Echo test endpoint (HTTP $http_code)"
    echo "Error: $body"
fi

echo "======================================================================"
echo " TEST SUMMARY"
echo "======================================================================"
echo ""
echo -e "${GREEN}✅ Tests Passed: $TESTS_PASSED${NC}"
echo -e "${RED}❌ Tests Failed: $TESTS_FAILED${NC}"
echo ""

TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED))
if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}⚠️  Some tests failed. Please check the errors above.${NC}"
    exit 1
fi
