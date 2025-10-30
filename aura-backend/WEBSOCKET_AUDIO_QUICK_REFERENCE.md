# WebSocket Audio Streaming - Quick Reference

**Updated:** October 31, 2025  
**Version:** 2.0 with Full Pipeline Support

---

## Connection

### Endpoint

```
ws://localhost:8000/ws/v1/audio
```

### Query Parameters

| Parameter         | Type    | Required | Default | Description                                           |
| ----------------- | ------- | -------- | ------- | ----------------------------------------------------- |
| `token`           | string  | Yes      | -       | JWT authentication token                              |
| `conversation_id` | string  | Yes      | -       | Conversation ID for context storage                   |
| `full_pipeline`   | boolean | No       | true    | Enable full AI pipeline (STT+SER+NER+COMET+Graph+LLM) |

### Example Connection URLs

**Full Pipeline (Recommended):**

```
ws://localhost:8000/ws/v1/audio?token=eyJhbGc...&conversation_id=clx123&full_pipeline=true
```

**Basic Mode (Fast):**

```
ws://localhost:8000/ws/v1/audio?token=eyJhbGc...&conversation_id=clx123&full_pipeline=false
```

---

## Client Implementation

### JavaScript/TypeScript

```javascript
// 1. Get JWT token from login
const loginResponse = await fetch("http://localhost:8000/auth/login", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ username: "user", password: "pass" }),
});
const { access_token } = await loginResponse.json();

// 2. Create or get conversation
const convResponse = await fetch("http://localhost:8000/conversations", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    Authorization: `Bearer ${access_token}`,
  },
  body: JSON.stringify({ title: "Voice Chat" }),
});
const { id: conversationId } = await convResponse.json();

// 3. Connect WebSocket
const ws = new WebSocket(
  `ws://localhost:8000/ws/v1/audio?token=${access_token}&conversation_id=${conversationId}&full_pipeline=true`
);

// 4. Handle connection
ws.onopen = () => {
  console.log("✅ Connected to Aura AI");
  startAudioCapture();
};

// 5. Handle messages
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  switch (data.type) {
    case "status":
      console.log("Status:", data.content);
      updateStatusUI(data.content);
      break;

    case "analysis":
      console.log("Analysis:", data.analysis_packet);
      displayTranscript(data.analysis_packet.transcript.text);
      displayEmotion(data.analysis_packet.emotion);
      displayEntities(data.analysis_packet.entities);
      break;

    case "response":
      console.log("AI Response:", data.ai_response.text);
      displayAIMessage(data.ai_response.text);
      break;

    case "error":
      console.error("Error:", data.content);
      showError(data.content);
      break;
  }
};

// 6. Send audio chunks
function sendAudioChunk(audioBuffer) {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(audioBuffer);
  }
}

// 7. Handle disconnection
ws.onclose = () => {
  console.log("❌ Disconnected from Aura AI");
  stopAudioCapture();
};

ws.onerror = (error) => {
  console.error("WebSocket error:", error);
};
```

---

## Audio Capture

### Web Audio API

```javascript
let mediaRecorder;
let audioContext;
let audioWorkletNode;

async function startAudioCapture() {
  try {
    // Get microphone access
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1, // Mono
        sampleRate: 16000, // 16kHz
        echoCancellation: true,
        noiseSuppression: true,
      },
    });

    // Create audio context
    audioContext = new AudioContext({ sampleRate: 16000 });
    const source = audioContext.createMediaStreamSource(stream);

    // Create script processor for audio chunks
    const processor = audioContext.createScriptProcessor(4096, 1, 1);

    processor.onaudioprocess = (event) => {
      const inputData = event.inputBuffer.getChannelData(0);

      // Convert Float32Array to Int16Array (PCM 16-bit)
      const int16Data = new Int16Array(inputData.length);
      for (let i = 0; i < inputData.length; i++) {
        const s = Math.max(-1, Math.min(1, inputData[i]));
        int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }

      // Send to WebSocket
      ws.send(int16Data.buffer);
    };

    source.connect(processor);
    processor.connect(audioContext.destination);

    console.log("🎤 Audio capture started");
  } catch (error) {
    console.error("Microphone access denied:", error);
  }
}

function stopAudioCapture() {
  if (audioContext) {
    audioContext.close();
    console.log("🛑 Audio capture stopped");
  }
}
```

---

## Message Formats

### Server → Client Messages

#### 1. Status Message

```json
{
  "type": "status",
  "content": "Connected to Aura AI - Full AI Pipeline",
  "user_id": "clx123",
  "username": "john_doe",
  "conversation_id": "conv456",
  "pipeline_mode": "Full AI Pipeline",
  "timestamp": "2025-10-31T10:30:00Z"
}
```

#### 2. Analysis Message (Full Pipeline)

```json
{
  "type": "analysis",
  "analysis_packet": {
    "transcript": {
      "text": "I'm meeting Sarah at the coffee shop tomorrow",
      "language": "en"
    },
    "emotion": {
      "from_audio": {
        "primary": "neutral",
        "confidence": 0.85,
        "all_scores": {
          "neutral": 0.85,
          "happy": 0.1,
          "sad": 0.05
        }
      },
      "from_text": {
        "detected": ["hopeful", "excited"],
        "context": {
          "subject_emotions": ["hopeful"],
          "other_emotions": []
        }
      }
    },
    "entities": {
      "people": [{ "text": "Sarah", "start": 13, "end": 18 }],
      "places": [{ "text": "coffee shop", "start": 26, "end": 37 }],
      "dates": [{ "text": "tomorrow", "start": 38, "end": 46 }]
    },
    "commonsense": {
      "inferences": {
        "subject": {
          "feelings": ["hopeful", "excited"],
          "wants": ["to meet friend", "to socialize"],
          "effects": ["strengthens friendship"]
        },
        "other": {
          "feelings": ["welcomed", "appreciated"],
          "wants": ["to enjoy company"],
          "effects": ["feels valued"]
        }
      }
    },
    "graph_updates": {
      "nodes_created": 3,
      "relationships_created": 5,
      "summary": "Added Sarah (person), coffee shop (place), tomorrow (date)"
    },
    "metadata": {
      "total_processing_time_ms": 420,
      "audio_duration_seconds": 2.5,
      "timestamp": "2025-10-31T10:30:01.234Z"
    }
  },
  "conversation_id": "conv456",
  "timestamp": "2025-10-31T10:30:01.234Z"
}
```

#### 3. AI Response Message

```json
{
  "type": "response",
  "ai_response": {
    "text": "That sounds wonderful! Meeting Sarah at a coffee shop is a great way to catch up. I hope you both have a lovely time tomorrow! Is there anything specific you're looking forward to discussing?",
    "model": "gpt-4",
    "tokens_used": 45,
    "finish_reason": "stop",
    "timestamp": "2025-10-31T10:30:03.456Z"
  },
  "timestamp": "2025-10-31T10:30:03.456Z"
}
```

#### 4. Analysis Message (Basic Mode)

```json
{
  "type": "analysis",
  "transcript": {
    "text": "Hello, how are you?",
    "language": "en"
  },
  "emotion": {
    "primary": "neutral",
    "confidence": 0.85,
    "all_scores": {
      "neutral": 0.85,
      "happy": 0.1,
      "sad": 0.05
    }
  },
  "audio": {
    "duration": 1.5,
    "sample_rate": 16000
  },
  "processing": {
    "total_time_ms": 200,
    "transcription_time_ms": 150,
    "emotion_time_ms": 100
  },
  "timestamp": "2025-10-31T10:30:00Z"
}
```

#### 5. Error Message

```json
{
  "type": "error",
  "content": "Pipeline processing failed",
  "error": "Transcription service unavailable",
  "timestamp": "2025-10-31T10:30:00Z"
}
```

---

## React Component Example

```tsx
import React, { useEffect, useState, useRef } from "react";

interface Message {
  type: string;
  content?: any;
}

export function VoiceChat() {
  const [connected, setConnected] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [emotion, setEmotion] = useState("");
  const [aiResponse, setAiResponse] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);

  const wsRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);

  useEffect(() => {
    // Connect WebSocket
    const token = localStorage.getItem("access_token");
    const conversationId = localStorage.getItem("conversation_id");

    const ws = new WebSocket(
      `ws://localhost:8000/ws/v1/audio?token=${token}&conversation_id=${conversationId}&full_pipeline=true`
    );

    ws.onopen = () => {
      setConnected(true);
      startRecording();
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setMessages((prev) => [...prev, data]);

      if (data.type === "analysis") {
        setTranscript(data.analysis_packet.transcript.text);
        setEmotion(data.analysis_packet.emotion.from_audio.primary);
      } else if (data.type === "response") {
        setAiResponse(data.ai_response.text);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      stopRecording();
    };

    wsRef.current = ws;

    return () => {
      ws.close();
    };
  }, []);

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, sampleRate: 16000 },
    });

    const mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (event) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(event.data);
      }
    };

    mediaRecorder.start(100); // Send chunks every 100ms
    mediaRecorderRef.current = mediaRecorder;
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
  };

  return (
    <div className="voice-chat">
      <div className="status">
        {connected ? "🟢 Connected" : "🔴 Disconnected"}
      </div>

      <div className="transcript">
        <h3>You said:</h3>
        <p>{transcript}</p>
        <span className="emotion">Emotion: {emotion}</span>
      </div>

      <div className="ai-response">
        <h3>Aura says:</h3>
        <p>{aiResponse}</p>
      </div>

      <div className="messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.type}`}>
            {JSON.stringify(msg, null, 2)}
          </div>
        ))}
      </div>
    </div>
  );
}
```

---

## Python Client Example

```python
import asyncio
import websockets
import json
import pyaudio

async def audio_stream_client():
    # Authentication
    token = "your_jwt_token"
    conversation_id = "your_conversation_id"

    # WebSocket URL
    url = f"ws://localhost:8000/ws/v1/audio?token={token}&conversation_id={conversation_id}&full_pipeline=true"

    async with websockets.connect(url) as ws:
        print("✅ Connected to Aura AI")

        # Audio setup
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000

        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )

        # Start audio capture task
        async def send_audio():
            while True:
                data = stream.read(CHUNK)
                await ws.send(data)
                await asyncio.sleep(0.01)

        # Start receive task
        async def receive_messages():
            async for message in ws:
                data = json.loads(message)
                print(f"\n[{data['type'].upper()}]")

                if data['type'] == 'analysis':
                    packet = data['analysis_packet']
                    print(f"Transcript: {packet['transcript']['text']}")
                    print(f"Emotion: {packet['emotion']['from_audio']['primary']}")
                    print(f"Entities: {packet['entities']}")

                elif data['type'] == 'response':
                    print(f"AI: {data['ai_response']['text']}")

        # Run both tasks
        await asyncio.gather(
            send_audio(),
            receive_messages()
        )

if __name__ == "__main__":
    asyncio.run(audio_stream_client())
```

---

## Testing

### Using wscat

```bash
# Install wscat
npm install -g wscat

# Connect (get token first from /auth/login)
wscat -c "ws://localhost:8000/ws/v1/audio?token=YOUR_TOKEN&conversation_id=CONV_ID&full_pipeline=true"

# Note: wscat doesn't support binary audio, use for testing connection only
```

### Using Postman

1. Open Postman
2. Create new WebSocket request
3. Enter URL: `ws://localhost:8000/ws/v1/audio?token=TOKEN&conversation_id=ID`
4. Click Connect
5. Switch to Binary mode for sending audio

---

## Troubleshooting

### Common Issues

**1. Authentication Failed**

```
Error: "Invalid authentication token"
Solution: Get fresh token from /auth/login
```

**2. Conversation Not Found**

```
Error: "Conversation not found or access denied"
Solution: Create conversation first via POST /conversations
```

**3. Pipeline Not Available**

```
Error: "Chat orchestrator service not available"
Solution: Ensure all models are loaded on backend startup
```

**4. Audio Format Issues**

```
Error: "Failed to process audio data"
Solution: Ensure audio is 16kHz, mono, 16-bit PCM
```

---

## Performance Tips

1. **Use Full Pipeline for Conversations**

   - Rich context understanding
   - Better AI responses
   - Worth the extra 200-300ms

2. **Use Basic Mode for Dictation**

   - Fast transcription only
   - No entity extraction
   - Lower latency

3. **Optimize Audio Chunks**

   - Send ~100ms chunks (1600 samples @ 16kHz)
   - Reduces network overhead
   - Maintains smooth streaming

4. **Handle Silence Detection**
   - Server detects silence after 1.5s
   - Triggers processing automatically
   - No need for client-side VAD

---

## Security

### Best Practices

1. **Token Storage**

   ```javascript
   // Store in httpOnly cookie (best)
   document.cookie = `token=${token}; HttpOnly; Secure; SameSite=Strict`;

   // Or secure storage
   sessionStorage.setItem("token", token);
   ```

2. **Token Refresh**

   ```javascript
   // Refresh before expiry (30 min default)
   setTimeout(refreshToken, 25 * 60 * 1000);
   ```

3. **Secure WebSocket**
   ```javascript
   // Use WSS in production
   const ws = new WebSocket("wss://your-domain.com/ws/v1/audio?...");
   ```

---

## Resources

- [System Design Documentation](./SYSTEM_DESIGN.md)
- [Week 8 Enhancement Summary](./WEEK8_ENHANCEMENT_SUMMARY.md)
- [API Reference](./API_REFERENCE.md)
- [Authentication Guide](./AUTH_SYSTEM.md)

---

**Last Updated:** October 31, 2025  
**Version:** 2.0
