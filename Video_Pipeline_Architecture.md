# Aura Video Analysis Pipeline - Complete Architecture Documentation

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Module 1: Keyframe Extraction](#module-1-keyframe-extraction)
4. [Module 2: Scene Analysis (LLaVA)](#module-2-scene-analysis-llava)
5. [Module 3: Face Analysis](#module-3-face-analysis)
6. [Module 4: Integrated Pipeline](#module-4-integrated-pipeline)
7. [Complete Code Walkthrough](#complete-code-walkthrough)
8. [Testing Guide](#testing-guide)

---

## Overview

The Aura Video Analysis Pipeline is a sophisticated multi-modal AI system that processes video content to extract:

- **Scene descriptions** using vision-language models
- **Face detection and tracking** across frames
- **Emotion recognition** from facial expressions
- **Identity persistence** throughout the video

### Key Components

```
video/
├── scene_captioner.py      (498 lines) - LLaVA-based scene analysis
├── face_analysis.py         (567 lines) - Multi-model face pipeline
├── integrated_analysis.py   (549 lines) - Combined orchestration
└── __init__.py
```

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                          VIDEO INPUT FILE                             │
│                     (MP4, AVI, MOV, MKV, etc.)                       │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    MODULE 1: KEYFRAME EXTRACTION                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Function: extract_keyframes()                                  │  │
│  │  Location: scene_captioner.py (lines 27-137)                   │  │
│  │                                                                 │  │
│  │  Process:                                                       │  │
│  │  1. Open video with cv2.VideoCapture()                        │  │
│  │  2. Read metadata: FPS, resolution, frame count               │  │
│  │  3. Calculate frame_interval = FPS × interval_sec             │  │
│  │  4. Loop through frames:                                      │  │
│  │     • if frame_count % frame_interval == 0:                   │  │
│  │       - Extract frame                                         │  │
│  │       - Convert BGR → RGB                                     │  │
│  │       - Create PIL Image                                      │  │
│  │       - Calculate timestamp = frame_count / FPS               │  │
│  │  5. Return List[Dict] with frame, timestamp, metadata        │  │
│  │                                                                 │  │
│  │  Key Variables:                                                 │  │
│  │  • fps: Frames per second from video metadata                 │  │
│  │  • frame_interval: Number of frames to skip                   │  │
│  │  • frame_count: Current position in video                     │  │
│  │  • timestamp: Time position in seconds                        │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
                List[Dict] containing:
                [
                  {
                    'frame': PIL.Image,
                    'timestamp': 1.5,
                    'frame_number': 45,
                    'formatted_time': '00:00:01',
                    'fps': 30.0
                  },
                  ...
                ]
                               │
               ┌───────────────┴────────────────┐
               │                                 │
               ▼                                 ▼
┌──────────────────────────┐      ┌──────────────────────────────┐
│   MODULE 2: SCENE        │      │   MODULE 3: FACE             │
│   ANALYSIS (LLaVA)       │      │   ANALYSIS (Multi-Model)     │
└──────────────────────────┘      └──────────────────────────────┘
               │                                 │
               ▼                                 ▼

[DETAILED MODULE SECTIONS BELOW]
```

---

## Module 1: Keyframe Extraction

### File: `scene_captioner.py` (Lines 27-137)

#### Function: `extract_keyframes()`

**Purpose:** Extract frames from video at specified time intervals

**Input Parameters:**

- `video_path` (str): Path to video file
- `interval_sec` (float): Time between keyframes (default: 1.0s)
- `max_frames` (Optional[int]): Maximum frames to extract

**Return Value:**

```python
List[Dict[str, Any]] = [
    {
        'frame': PIL.Image (RGB format),
        'timestamp': float (seconds),
        'frame_number': int (frame index),
        'formatted_time': str ('HH:MM:SS'),
        'fps': float (frames per second)
    },
    ...
]
```

### Line-by-Line Breakdown

```python
# LINE 31-34: Input validation
video_path = Path(video_path)
if not video_path.exists():
    raise FileNotFoundError(f\"Video file not found: {video_path}\")
```

**Explanation:** Convert string to Path object and verify file exists before proceeding.

```python
# LINE 41: Open video file
cap = cv2.VideoCapture(str(video_path))
```

**Explanation:** Create VideoCapture object. This opens the video file and prepares it for reading.

```python
# LINE 43-44: Validate video opened successfully
if not cap.isOpened():
    raise ValueError(f\"Failed to open video file: {video_path}\")
```

**Explanation:** Check if video file was opened correctly. Returns False if file is corrupted or format unsupported.

```python
# LINE 47-51: Read video metadata
fps = cap.get(cv2.CAP_PROP_FPS)                    # Frames per second
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames
duration = total_frames / fps if fps > 0 else 0    # Duration in seconds
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))     # Width in pixels
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))   # Height in pixels
```

**Explanation:**

- `CAP_PROP_FPS`: Property ID for frames per second (typically 24, 30, or 60)
- `CAP_PROP_FRAME_COUNT`: Total number of frames in video
- Duration calculated as: total_frames ÷ FPS = seconds
- Example: 900 frames ÷ 30 FPS = 30 seconds

```python
# LINE 68: Calculate frame interval
frame_interval = int(fps * interval_sec)
```

**Explanation:**

- Formula: FPS × interval_seconds = frames_to_skip
- Example: 30 FPS × 2.0s = 60 (extract every 60th frame)
- This determines how many frames to skip between keyframes

```python
# LINE 69-70: Ensure minimum interval
if frame_interval < 1:
    frame_interval = 1
```

**Explanation:** Prevent division by zero or negative intervals. Minimum is 1 (extract every frame).

```python
# LINE 80-94: Main extraction loop
while cap.isOpened():
    ret, frame = cap.read()  # Read next frame

    if not ret:
        break  # End of video reached
```

**Explanation:**

- `cap.read()` returns tuple: (success_flag, frame_data)
- `ret` is True if frame was read successfully
- `frame` is NumPy array with shape (height, width, 3) in BGR format
- Loop continues until `ret` is False (end of video)

```python
# LINE 91: Check if frame should be extracted
if frame_count % frame_interval == 0:
```

**Explanation:**

- Modulo operator (%) checks if frame_count is divisible by interval
- Example: frame_count=0,60,120,180... with interval=60
- Only extracts frames at these positions

```python
# LINE 93: Convert color space
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

**Explanation:**

- OpenCV reads images in BGR format (Blue-Green-Red)
- Standard format is RGB (Red-Green-Blue)
- Conversion necessary for PIL, matplotlib, and neural networks
- Mathematical operation: channels are reordered [2,1,0] → [0,1,2]

```python
# LINE 96: Create PIL Image
pil_image = Image.fromarray(frame_rgb)
```

**Explanation:**

- Converts NumPy array to PIL Image object
- PIL Images are standard in Python image processing
- Required format for many ML models

```python
# LINE 99: Calculate timestamp
timestamp = frame_count / fps if fps > 0 else 0
```

**Explanation:**

- Formula: frame_index ÷ FPS = time_in_seconds
- Example: frame 90 ÷ 30 FPS = 3.0 seconds
- Tells us when this frame appears in the video

```python
# LINE 102-103: Format timestamp
time_delta = timedelta(seconds=timestamp)
formatted_time = str(time_delta).split('.')[0]
```

**Explanation:**

- `timedelta` converts seconds to HH:MM:SS format
- Example: 125.5 seconds → "0:02:05.500000"
- `.split('.')[0]` removes fractional seconds → "0:02:05"

```python
# LINE 106-112: Store keyframe data
keyframe_data = {
    'frame': pil_image,
    'timestamp': timestamp,
    'frame_number': frame_count,
    'formatted_time': formatted_time
}
keyframes.append(keyframe_data)
```

**Explanation:** Create dictionary with all frame metadata and append to results list.

```python
# LINE 133: Release video capture
finally:
    cap.release()
```

**Explanation:**

- Always release video file handle, even if error occurs
- Prevents file lock and memory leaks
- `finally` block ensures this runs even with exceptions

---

## Module 2: Scene Analysis (LLaVA)

### File: `scene_captioner.py` (Lines 140-498)

### Class: `SceneCaptioner`

**Purpose:** Use LLaVA (Large Language and Vision Assistant) to generate natural language descriptions of video frames.

#### What is LLaVA?

LLaVA is a vision-language model that combines:

1. **Vision Encoder** (CLIP ViT): Processes images into embeddings
2. **Language Decoder** (LLaMA/Vicuna): Generates text descriptions
3. **Projection Layer**: Connects vision and language spaces

**Architecture:**

```
Image (PIL) → Vision Encoder (ViT) → Vision Embeddings (768D)
                                           ↓
                              Projection Layer (768D → 4096D)
                                           ↓
Text Prompt → Tokenizer → Token Embeddings → Language Model (LLaMA)
                                                      ↓
                                              Generated Caption
```

### Line-by-Line Breakdown

```python
# LINE 146-156: Constructor
def __init__(
    self,
    model_name: str = \"llava-hf/llava-1.5-7b-hf\",
    device: Optional[str] = None,
    load_in_8bit: bool = False
):
```

**Parameters:**

- `model_name`: Hugging Face model identifier
  - `llava-1.5-7b-hf`: 7 billion parameters (standard)
  - `llava-1.5-13b-hf`: 13 billion parameters (higher quality)
- `device`: Computation device ('cuda' or 'cpu')
- `load_in_8bit`: Use quantization for memory efficiency (reduces 16GB → 8GB)

```python
# LINE 157-159: Device selection
self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
```

**Explanation:**

- Checks if NVIDIA GPU with CUDA is available
- Falls back to CPU if no GPU found
- GPU provides 10-100x speedup for inference

```python
# LINE 168: Load processor
self.processor = LlavaNextProcessor.from_pretrained(self.model_name)
```

**Explanation:**

- Processor handles two tasks:
  1. **Image preprocessing**: Resize, normalize, convert to tensor
  2. **Text tokenization**: Convert prompt text to token IDs
- Downloads from Hugging Face Hub (cached after first use)

```python
# LINE 173-179: Load model with 8-bit quantization
if self.load_in_8bit and self.device == 'cuda':
    self.model = LlavaNextForConditionalGeneration.from_pretrained(
        self.model_name,
        load_in_8bit=True,
        device_map=\"auto\",
        low_cpu_mem_usage=True
    )
```

**Explanation:**

- **8-bit quantization**: Reduces memory by converting FP32 weights to INT8
- **device_map=\"auto\"**: Automatically distributes model across GPUs
- **low_cpu_mem_usage**: Minimizes RAM usage during loading
- Memory impact: 7B model: ~14GB FP16 → ~7GB INT8

```python
# LINE 181-190: Standard model loading
else:
    torch_dtype = torch.float16 if self.device == 'cuda' else torch.float32
    self.model = LlavaNextForConditionalGeneration.from_pretrained(
        self.model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True
    )
    self.model = self.model.to(self.device)
```

**Explanation:**

- **float16 (GPU)**: Half precision, 2x memory reduction, minimal quality loss
- **float32 (CPU)**: Full precision required for CPU stability
- `.to(device)`: Moves model tensors to GPU or CPU

```python
# LINE 193: Set evaluation mode
self.model.eval()
```

**Explanation:**

- Disables dropout and batch normalization updates
- Enables inference-only mode (no gradient computation)
- Improves speed and ensures consistent outputs

### Function: `generate_caption()`

**Purpose:** Generate descriptive caption for a single image

```python
# LINE 206-213: Function signature
def generate_caption(
    self,
    image: Image.Image,
    prompt: Optional[str] = None,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    do_sample: bool = True
) -> str:
```

**Parameters:**

- `image`: PIL Image object (RGB format)
- `prompt`: Custom prompt template (uses default if None)
- `max_new_tokens`: Maximum caption length in tokens (~150 words)
- `temperature`: Creativity control (0.0=deterministic, 1.0=creative)
- `do_sample`: Enable sampling (vs greedy decoding)

```python
# LINE 216-217: Default prompt
if prompt is None:
    prompt = \"USER: <image>\\nWhat is happening in this scene? Provide a detailed description.\\nASSISTANT:\"
```

**Explanation:**

- LLaVA uses chat format with USER/ASSISTANT roles
- `<image>` token is special placeholder for image embedding
- Prompt engineering affects output quality significantly

```python
# LINE 221-225: Prepare inputs
inputs = self.processor(
    text=prompt,
    images=image,
    return_tensors=\"pt\"
)
```

**Explanation:**

- Processor performs two operations simultaneously:
  1. **Image**: Resize to 336×336, normalize to [-1, 1], convert to tensor
  2. **Text**: Tokenize prompt, add special tokens ([BOS], [EOS])
- Returns dictionary:
  ```python
  {
      'input_ids': Tensor([batch, seq_len]),      # Text tokens
      'pixel_values': Tensor([batch, 3, 336, 336]) # Image tensor
  }
  ```

```python
# LINE 228: Move to device
inputs = {k: v.to(self.device) for k, v in inputs.items()}
```

**Explanation:**

- Transfers all tensors from CPU to GPU memory
- Required before model inference
- Dictionary comprehension iterates over all inputs

```python
# LINE 231-239: Generate caption
with torch.no_grad():
    output_ids = self.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=0.9,
        num_beams=1
    )
```

**Explanation:**

- `torch.no_grad()`: Disables gradient computation (saves memory, faster)
- `**inputs`: Unpacks dictionary as keyword arguments
- **Generation parameters:**
  - `max_new_tokens`: Stop after N tokens (prevents infinite generation)
  - `temperature`: Controls randomness
    - 0.0 = Always pick most likely token (deterministic)
    - 1.0 = Sample according to probabilities (creative)
  - `top_p=0.9`: Nucleus sampling (consider top 90% probability mass)
  - `num_beams=1`: No beam search (faster, less memory)

**Generation Process:**

1. Model encodes image through vision encoder
2. Combines image embedding with text prompt
3. Autoregressively generates tokens:
   - At each step, predict next token
   - Sample from probability distribution
   - Append to sequence
   - Repeat until [EOS] or max length

```python
# LINE 242-245: Decode output
generated_text = self.processor.decode(
    output_ids[0],
    skip_special_tokens=True
)
```

**Explanation:**

- Converts token IDs back to text string
- `output_ids[0]`: Take first (and only) sequence from batch
- `skip_special_tokens`: Remove [BOS], [EOS], [PAD] tokens
- Example: [1, 450, 5156, 338, ...] → \"A person is sitting at a desk...\"

```python
# LINE 248-252: Extract assistant response
if \"ASSISTANT:\" in generated_text:
    caption = generated_text.split(\"ASSISTANT:\")[-1].strip()
else:
    caption = generated_text.strip()
```

**Explanation:**

- LLaVA outputs include the full conversation
- We only want the assistant's response
- `.split(\"ASSISTANT:\")[-1]`: Take text after last \"ASSISTANT:\"
- `.strip()`: Remove leading/trailing whitespace

---

## Module 3: Face Analysis

### File: `face_analysis.py` (Lines 1-567)

This module uses three separate AI models in sequence:

1. **MTCNN** - Face Detection
2. **InceptionResnetV1** - Identity Embeddings
3. **ViT** - Emotion Recognition

### Function: `load_face_models()`

**Purpose:** Initialize all three face analysis models

```python
# LINE 24-48: Load MTCNN
from facenet_pytorch import MTCNN

mtcnn = MTCNN(
    image_size=160,          # Output face size
    margin=20,               # Pixels around face
    min_face_size=20,        # Minimum detectable face
    thresholds=[0.6, 0.7, 0.7],  # P-Net, R-Net, O-Net thresholds
    factor=0.709,            # Scale factor for pyramid
    post_process=True,       # Apply post-processing
    device=device,
    keep_all=True,           # Return all faces
    select_largest=False
)
```

**MTCNN Architecture:**

```
Input Image (any size)
       ↓
┌──────────────────┐
│   P-Net (Stage 1)│  Proposal Network
│   Quick scan     │  • Create image pyramid (scaled versions)
│   12×12 windows  │  • Scan with 12×12 conv net
└────────┬─────────┘  • Generate candidate windows
         ↓            • Threshold: 0.6
    Candidates
         ↓
┌──────────────────┐
│   R-Net (Stage 2)│  Refinement Network
│   Refine boxes   │  • Resize candidates to 24×24
│   24×24 windows  │  • More complex CNN
└────────┬─────────┘  • Refine bounding boxes
         ↓            • Threshold: 0.7
    Better boxes
         ↓
┌──────────────────┐
│   O-Net (Stage 3)│  Output Network
│   Final detection│  • Resize to 48×48
│   48×48 + landmarks│• Full CNN
└────────┬─────────┘  • Facial landmarks (eyes, nose, mouth)
         ↓            • Final bounding boxes
         ↓            • Threshold: 0.7
  Detected Faces
  [x1, y1, x2, y2]
  + 5 landmarks
```

**Key Parameters Explained:**

- **image_size=160**: Output size for detected faces

  - All detected faces are resized to 160×160 pixels
  - This standardization is required for InceptionResnetV1

- **margin=20**: Padding around face

  - Adds 20 pixels on each side of detected box
  - Ensures full head is captured, not just facial features

- **min_face_size=20**: Minimum face size in pixels

  - Faces smaller than 20×20 pixels are ignored
  - Prevents false positives from small objects

- **thresholds=[0.6, 0.7, 0.7]**: Confidence thresholds

  - Higher = fewer false positives, more missed faces
  - Lower = more detections, more false positives
  - [P-Net, R-Net, O-Net] stages

- **factor=0.709**: Image pyramid scale factor
  - Each scale is 70.9% of previous
  - More scales = better detection at multiple sizes
  - √0.5 ≈ 0.709 (halves area each iteration)

```python
# LINE 80-91: Load InceptionResnetV1
from facenet_pytorch import InceptionResnetV1

identity_model = InceptionResnetV1(
    pretrained='vggface2',
    device=device
).eval()
```

**InceptionResnetV1 Architecture:**

```
Input: Face image (160×160×3)
         ↓
┌─────────────────────┐
│  Stem (Initial Conv)│
│  • Conv 3×3, stride 2│  Output: 80×80×64
│  • Conv 3×3          │
│  • MaxPool           │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Inception-ResNet-A  │  • 5× blocks
│ (35×35 feature maps)│  • Multi-scale features
└──────────┬──────────┘  • Residual connections
           ↓
┌─────────────────────┐
│ Reduction-A         │  Downsample: 35×35 → 17×17
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Inception-ResNet-B  │  • 10× blocks
│ (17×17 feature maps)│  • Deeper features
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Reduction-B         │  Downsample: 17×17 → 8×8
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Inception-ResNet-C  │  • 5× blocks
│ (8×8 feature maps)  │  • Highest-level features
└──────────┬──────────┘
           ↓
    Global Average Pool
           ↓
    Dropout (0.6)
           ↓
    Linear (1792 → 512)
           ↓
    L2 Normalization
           ↓
Output: 512-D Embedding
```

**What are embeddings?**

- Embeddings are dense vector representations
- Each face is encoded as 512 numbers
- Similar faces have similar embeddings (measured by cosine similarity)
- Example: Two photos of same person have similarity > 0.9

**Pre-trained on VGGFace2:**

- Dataset: 3.3 million face images
- 9,131 unique identities
- Model learned to distinguish between different people

```python
# LINE 106-130: Load Emotion Model
from transformers import AutoImageProcessor, AutoModelForImageClassification

emotion_model_name = \"trpakov/vit-face-expression\"

emotion_processor = AutoImageProcessor.from_pretrained(emotion_model_name)
emotion_model = AutoModelForImageClassification.from_pretrained(emotion_model_name)
emotion_model = emotion_model.to(device)
emotion_model.eval()

emotion_labels = list(emotion_model.config.id2label.values())
```

**ViT (Vision Transformer) for Emotion:**

```
Input: Face crop (224×224×3)
         ↓
┌─────────────────────┐
│ Patch Embedding     │
│ • Split into 16×16  │  Output: 196 patches
│   patches           │
│ • Linear projection │  Each patch → 768-D vector
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Position Embedding  │  Add learnable position info
└──────────┬──────────┘  (patches lose spatial info)
           ↓
┌─────────────────────┐
│ Transformer Encoder │
│ • 12 layers         │  • Self-attention mechanism
│ • Multi-head attn   │  • Learns relationships between patches
│ • Feed-forward      │  • Residual connections
└──────────┬──────────┘
           ↓
    Classification Head
           ↓
    Softmax
           ↓
Emotion Probabilities
[angry, disgust, fear, happy, neutral, sad, surprise]
```

**Emotion Labels:**

- angry, disgust, fear, happy, neutral, sad, surprise
- Model outputs probability for each (sums to 1.0)
- Example: [0.02, 0.01, 0.05, 0.87, 0.03, 0.01, 0.01] → \"happy\"

---

## Module 4: Integrated Pipeline

### File: `integrated_analysis.py`

**Purpose:** Orchestrate both scene and face analysis pipelines

### Function: `analyze_video_complete()`

**High-Level Flow:**

```
1. Extract keyframes from video (Module 1)
2. For each keyframe (parallel processing):
   a. Scene Analysis: Generate caption (Module 2)
   b. Face Analysis: Detect faces, emotions, identities (Module 3)
3. Combine results per frame
4. Track identities across frames
5. Generate statistics and summaries
6. Return structured results
```

**Complete Process Diagram:**

```
Video File
    ↓
extract_keyframes()
    ↓
[Frame 0, Frame 1, Frame 2, ...]
    ↓
    ├─────────────────────┬─────────────────────┐
    ↓                     ↓                     ↓
Scene Analysis      Face Detection      Identity Tracking
    ↓                     ↓                     ↓
\"Person at desk\"   [{box:[x,y,w,h],      Person 0: Frames 0-5
                      emotion:\"happy\",    Person 1: Frames 2-8
                      embedding:[...]}]
    ↓                     ↓                     ↓
    └─────────────────────┴─────────────────────┘
                          ↓
                  Combined Results
                          ↓
              [{
                timestamp: 0.0,
                scene: \"Person at desk\",
                faces: [...],
                emotions: [\"happy\"],
                num_faces: 1
              }, ...]
```

### Detailed Code Walkthrough

```python
# LINE 23-32: Function signature
def analyze_video_complete(
    video_path: str,
    keyframe_interval: float = 2.0,
    scene_model: str = 'llava-hf/llava-1.5-7b-hf',
    face_confidence: float = 0.9,
    identity_threshold: float = 0.6,
    analyze_scenes: bool = True,
    analyze_faces: bool = True
) -> List[Dict[str, Any]]:
```

**Parameters Explained:**

- `video_path`: Path to input video
- `keyframe_interval`: Seconds between frames (2.0 = analyze every 2 seconds)
- `scene_model`: LLaVA model variant
- `face_confidence`: Minimum confidence for face detection (0.9 = 90%)
- `identity_threshold`: Similarity threshold for same person (0.6 = 60%)
- `analyze_scenes`: Enable/disable scene captions
- `analyze_faces`: Enable/disable face analysis

```python
# LINE 52-62: Step 1 - Extract keyframes
keyframes = extract_keyframes(video_path, interval_sec=keyframe_interval)

# Extract components
frames = [kf['frame'] for kf in keyframes]
timestamps = [kf['timestamp'] for kf in keyframes]
frame_numbers = [kf['frame_number'] for kf in keyframes]
```

**Explanation:**

- Call keyframe extractor from Module 1
- Separate frames from metadata for processing
- Lists maintain parallel indexing: frames[i] has timestamp[i]

```python
# LINE 72-94: Step 2 - Scene Analysis (if enabled)
if analyze_scenes:
    scene_captioner = SceneCaptioner(model_name=scene_model)

    for i, (frame, ts) in enumerate(zip(frames, timestamps), 1):
        caption = scene_captioner.caption_frame(frame)
        scene_captions.append(caption)
```

**Explanation:**

- Initialize LLaVA model once
- Process each frame sequentially
- `zip()` pairs frames with timestamps
- `enumerate(..., 1)` starts counting from 1
- Captions stored in order matching frames

```python
# LINE 109-154: Step 3 - Face Analysis (if enabled)
if analyze_faces:
    # Load models
    face_models = load_face_models()

    # Analyze all frames
    all_faces = analyze_faces_in_video_frames(frames, face_models)

    # Track identities
    all_faces = track_identities_across_frames(all_faces, similarity_threshold=identity_threshold)
```

**Explanation:**

- Load MTCNN, InceptionResnetV1, ViT once
- Process all frames (can be parallelized)
- Identity tracking assigns consistent IDs across frames

**Identity Tracking Algorithm:**

```python
def track_identities_across_frames(all_faces, similarity_threshold):
    known_identities = []  # Store representative embeddings

    for frame_faces in all_faces:
        for face in frame_faces:
            embedding = face['identity_embedding']

            # Compare with known identities
            best_match = None
            best_similarity = 0.0

            for id, known_emb in enumerate(known_identities):
                sim = cosine_similarity(embedding, known_emb)
                if sim > best_similarity and sim >= threshold:
                    best_match = id
                    best_similarity = sim

            # Assign ID
            if best_match is not None:
                face['identity_id'] = best_match
            else:
                # New person
                face['identity_id'] = len(known_identities)
                known_identities.append(embedding)
```

**Cosine Similarity:**

```
similarity = (A · B) / (||A|| × ||B||)

Where:
• A, B = embedding vectors
• A · B = dot product
• ||A|| = L2 norm (magnitude)

Range: [-1, 1]
• 1.0 = identical
• 0.0 = orthogonal (unrelated)
• -1.0 = opposite

In practice:
• > 0.6 = likely same person
• 0.4-0.6 = uncertain
• < 0.4 = different people
```

```python
# LINE 164-183: Step 4 - Combine Results
for i in range(len(keyframes)):
    timestamp = timestamps[i]
    frame_number = frame_numbers[i]
    faces_in_frame = all_faces[i]

    emotions = [face['emotion'] for face in faces_in_frame]

    unique_people = len(set(face['identity_id'] for face in faces_in_frame))

    result = {
        'timestamp': timestamp,
        'frame_number': frame_number,
        'scene_caption': scene_captions[i],
        'faces': faces_in_frame,
        'num_faces': len(faces_in_frame),
        'emotions': emotions,
        'unique_people': unique_people
    }

    combined_results.append(result)
```

**Explanation:**

- Iterate through all frames
- For each frame, combine:
  - Temporal metadata (timestamp, frame number)
  - Scene caption from LLaVA
  - All detected faces with emotions and IDs
  - Aggregate statistics (num_faces, unique_people)
- Results list maintains chronological order

---

## Complete Code Walkthrough

### Critical Functions Deep Dive

#### 1. Color Space Conversion

```python
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

**Why necessary?**

- OpenCV uses BGR (Blue-Green-Red) channel order
- Standard libraries (PIL, matplotlib, PyTorch) use RGB
- Neural networks trained on RGB data

**Mathematical operation:**

```
BGR: [B, G, R] = [pixel[0], pixel[1], pixel[2]]
RGB: [R, G, B] = [pixel[2], pixel[1], pixel[0]]

Example pixel:
BGR: [255, 100, 50]  (blue=255, green=100, red=50)
RGB: [50, 100, 255]  (red=50, green=100, blue=255)
```

#### 2. Face Embedding Computation

```python
# Normalize face image to [-1, 1]
face_tensor = (face_tensor - 127.5) / 128.0

# Get embedding
with torch.no_grad():
    identity_embedding = identity_model(face_tensor)
```

**Normalization formula:**

```
normalized_pixel = (pixel_value - mean) / std

For [0, 255] range to [-1, 1]:
normalized = (pixel - 127.5) / 128.0

Example:
pixel = 0   → (0 - 127.5) / 128.0 = -0.996  ≈ -1
pixel = 127 → (127 - 127.5) / 128.0 = -0.004 ≈ 0
pixel = 255 → (255 - 127.5) / 128.0 = 0.996  ≈ 1
```

**Why normalize?**

- Neural networks converge faster with inputs near 0
- Prevents gradient saturation
- Matches training data preprocessing

#### 3. Emotion Classification

```python
# Get emotion logits
outputs = emotion_model(**inputs)
logits = outputs.logits

# Convert to probabilities
probs = torch.nn.functional.softmax(logits, dim=-1)

# Get predicted emotion
predicted_idx = torch.argmax(probs, dim=-1).item()
emotion = emotion_labels[predicted_idx]
```

**Softmax formula:**

```
softmax(x_i) = exp(x_i) / Σ exp(x_j)

Example:
logits = [2.0, 1.0, 0.1, -1.0]

exp(2.0) = 7.39
exp(1.0) = 2.72
exp(0.1) = 1.11
exp(-1.0) = 0.37
sum = 11.59

probs = [7.39/11.59, 2.72/11.59, 1.11/11.59, 0.37/11.59]
      = [0.637, 0.235, 0.096, 0.032]

argmax = 0 (highest probability)
```

**Why softmax?**

- Converts arbitrary numbers (logits) to probabilities
- Output sums to 1.0
- Preserves relative ordering
- Differentiable (for training)

---

## Testing Guide

### Prerequisites

```bash
# Install dependencies
pip install torch torchvision torchaudio
pip install transformers accelerate
pip install opencv-python pillow
pip install facenet-pytorch
pip install matplotlib seaborn tqdm
```

### Test 1: Keyframe Extraction Only

```python
from video.scene_captioner import extract_keyframes

keyframes = extract_keyframes(
    video_path=\"sample.mp4\",
    interval_sec=1.0,
    max_frames=10
)

print(f\"Extracted {len(keyframes)} frames\")
print(f\"First frame timestamp: {keyframes[0]['timestamp']}s\")
```

**Expected output:**

```
📹 EXTRACTING KEYFRAMES FROM VIDEO
================================================================================
📁 File: sample.mp4

📊 Video Properties:
   Resolution:    1920 × 1080 pixels
   FPS:           30.00 frames/second
   Total Frames:  900
   Duration:      30.00 seconds (0:00:30)

🎯 Extraction Settings:
   Interval:         1.0s per keyframe
   Frame Interval:   Every 30 frames
   Expected Output:  ~30 keyframes

⏳ Extracting keyframes...
Progress: 100%|██████████| 30/30 [00:05<00:00,  5.50frames/s]

✅ Successfully extracted 30 keyframes
   Time range: 0:00:00 to 0:00:29
================================================================================

Extracted 30 frames
First frame timestamp: 0.0s
```

### Test 2: Scene Analysis Only

```python
from video.scene_captioner import SceneCaptioner, extract_keyframes
from PIL import Image

# Extract one frame
keyframes = extract_keyframes(\"sample.mp4\", max_frames=1)
frame = keyframes[0]['frame']

# Initialize captioner
captioner = SceneCaptioner(model_name=\"llava-hf/llava-1.5-7b-hf\")

# Generate caption
caption = captioner.generate_caption(frame)

print(f\"Caption: {caption}\")
```

**Expected output:**

```
🚀 Initializing SceneCaptioner
   Model: llava-hf/llava-1.5-7b-hf
   Device: cuda
   8-bit Loading: False

📥 Loading LLaVA model from Hugging Face...
   This may take a few minutes on first run...
   Loading processor...
   Loading model...
   ✅ Model loaded in torch.float16 precision

✅ SceneCaptioner ready for inference on cuda

Caption: A person is sitting at a desk working on a laptop computer.
The desk has a lamp, some books, and a coffee mug on it. The room
appears to be a home office with bookshelves in the background.
```

### Test 3: Face Analysis Only

```python
from video.face_analysis import load_face_models, analyze_faces_in_frame
from video.scene_captioner import extract_keyframes

# Load models
models = load_face_models()

# Extract one frame
keyframes = extract_keyframes(\"sample.mp4\", max_frames=1)
frame = keyframes[0]['frame']

# Analyze faces
faces = analyze_faces_in_frame(
    frame=frame,
    mtcnn=models['mtcnn'],
    identity_model=models['identity_model'],
    emotion_model=models['emotion_model'],
    emotion_processor=models['emotion_processor'],
    emotion_labels=models['emotion_labels'],
    device=models['device']
)

print(f\"Found {len(faces)} face(s)\")
for i, face in enumerate(faces):
    print(f\"\\nFace {i+1}:\")
    print(f\"  Box: {face['box']}\")
    print(f\"  Emotion: {face['emotion']} ({face['emotion_confidence']:.2f})\")
    print(f\"  Embedding dimension: {len(face['identity_embedding'])}\")
```

**Expected output:**

```
================================================================================
🔧 LOADING FACE ANALYSIS MODELS
================================================================================
Device: cuda

📸 Loading MTCNN (Face Detection)...
✅ MTCNN loaded successfully
   - Min face size: 20px
   - Multi-face detection: Enabled

🔍 Loading InceptionResnetV1 (Identity Embeddings)...
✅ InceptionResnetV1 loaded successfully
   - Pre-trained on: VGGFace2
   - Embedding size: 512 dimensions
   - Mode: Evaluation (no gradients)

😊 Loading Facial Emotion Recognition Model...
   Model: trpakov/vit-face-expression
✅ Emotion model loaded successfully
   - Model: Vision Transformer (ViT)
   - Emotions: angry, disgust, fear, happy, neutral, sad, surprise
   - Total classes: 7

================================================================================
✅ ALL FACE ANALYSIS MODELS LOADED
================================================================================
Models ready on device: cuda
Total models: 3
================================================================================

Found 1 face(s)

Face 1:
  Box: [450, 120, 680, 420]
  Emotion: happy (0.87)
  Embedding dimension: 512
```

### Test 4: Complete Pipeline

```python
from video.integrated_analysis import analyze_video_complete

results = analyze_video_complete(
    video_path=\"sample.mp4\",
    keyframe_interval=2.0,
    analyze_scenes=True,
    analyze_faces=True
)

# Print summary
print(f\"\\nAnalyzed {len(results)} frames\")
print(f\"\\nFirst frame:\")
print(f\"  Timestamp: {results[0]['timestamp']}s\")
print(f\"  Caption: {results[0]['scene_caption']}\")
print(f\"  Faces: {results[0]['num_faces']}\")
print(f\"  Emotions: {results[0]['emotions']}\")
```

**Expected output:**

```
================================================================================
🎬 COMPLETE VIDEO ANALYSIS
================================================================================
📁 Video: sample.mp4
⏱️  Keyframe interval: 2.0s
🎨 Scene analysis: ENABLED
🎭 Face analysis: ENABLED
================================================================================

📹 STEP 1: Extracting keyframes from video...
────────────────────────────────────────────────────────────────────────────────
✅ Extracted 15 keyframes
   Duration: 28.50s
   FPS: 30.0

🎨 STEP 2: Analyzing scene content...
────────────────────────────────────────────────────────────────────────────────
   Frame 1/15 (0.0s)
   → A person is sitting at a desk with a laptop computer.
   Frame 2/15 (2.0s)
   → The person is typing on the keyboard and looking at the screen.
   ...

✅ Scene analysis complete

🎭 STEP 3: Analyzing faces and emotions...
────────────────────────────────────────────────────────────────────────────────
   Loading models...
   Processing frames...
Processed 15/15 frames...
✅ Analyzed 15 frames, found 12 total faces

   Tracking identities...
Identified 1 unique person(s) across all frames

✅ Face analysis complete
   Total detections: 12
   Frames with faces: 12/15
   Unique people: 1

🔗 STEP 4: Combining results...
────────────────────────────────────────────────────────────────────────────────
✅ Combined 15 frame results

================================================================================
✅ VIDEO ANALYSIS COMPLETE
================================================================================
📊 Summary:
   Frames analyzed: 15
   Total duration: 28.50s
   Total faces detected: 12
   Most common emotion: happy (8 occurrences)
================================================================================

Analyzed 15 frames

First frame:
  Timestamp: 0.0s
  Caption: A person is sitting at a desk with a laptop computer.
  Faces: 1
  Emotions: ['happy']
```

---

## Performance Metrics

### Memory Requirements

| Component            | GPU Memory | CPU RAM    |
| -------------------- | ---------- | ---------- |
| LLaVA-1.5-7B (FP16)  | ~14 GB     | ~28 GB     |
| LLaVA-1.5-7B (8-bit) | ~7 GB      | ~14 GB     |
| MTCNN                | ~100 MB    | ~200 MB    |
| InceptionResnetV1    | ~200 MB    | ~400 MB    |
| ViT Emotion          | ~300 MB    | ~600 MB    |
| **Total (Standard)** | **~15 GB** | **~29 GB** |
| **Total (8-bit)**    | **~8 GB**  | **~15 GB** |

### Processing Speed

**GPU (NVIDIA RTX 3090):**

- Keyframe extraction: ~100 FPS (0.01s per frame)
- Scene captioning: ~2 FPS (0.5s per frame)
- Face detection: ~50 FPS (0.02s per frame)
- Face analysis: ~30 FPS (0.03s per frame)
- **Total: ~1.5 FPS** (0.65s per frame)

**CPU (Intel i9-12900K):**

- Keyframe extraction: ~60 FPS (0.017s per frame)
- Scene captioning: ~0.1 FPS (10s per frame)
- Face detection: ~5 FPS (0.2s per frame)
- Face analysis: ~3 FPS (0.33s per frame)
- **Total: ~0.09 FPS** (11s per frame)

**30-second video (15 frames at 2s intervals):**

- GPU: ~10 seconds total
- CPU: ~3 minutes total

---

## Common Issues & Solutions

### Issue 1: Out of Memory

**Symptom:**

```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**

1. Enable 8-bit quantization:

```python
captioner = SceneCaptioner(load_in_8bit=True)
```

2. Reduce keyframe extraction:

```python
keyframes = extract_keyframes(interval_sec=5.0)  # Instead of 1.0
```

3. Process in batches:

```python
for i in range(0, len(frames), 5):
    batch = frames[i:i+5]
    # Process batch
```

### Issue 2: Slow Processing

**Symptom:** Processing takes hours for short videos

**Solutions:**

1. Use GPU:

```python
# Check GPU availability
print(torch.cuda.is_available())  # Should be True
```

2. Reduce frame analysis:

```python
results = analyze_video_complete(
    keyframe_interval=5.0,  # Analyze every 5 seconds
    analyze_scenes=False     # Skip scene analysis
)
```

3. Use smaller models:

```python
# Use 7B instead of 13B model
captioner = SceneCaptioner(model_name=\"llava-hf/llava-1.5-7b-hf\")
```

### Issue 3: No Faces Detected

**Symptom:** `num_faces: 0` for all frames

**Solutions:**

1. Lower confidence threshold:

```python
results = analyze_video_complete(face_confidence=0.7)  # Instead of 0.9
```

2. Check image quality:

```python
frame = keyframes[0]['frame']
plt.imshow(frame)
plt.show()  # Verify frame is not corrupted
```

3. Adjust MTCNN parameters:

```python
mtcnn = MTCNN(
    min_face_size=15,  # Lower minimum
    thresholds=[0.5, 0.6, 0.6]  # Lower thresholds
)
```

---

## Optimization Tips

### 1. Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor

def process_frame_parallel(frames, captioner):
    with ThreadPoolExecutor(max_workers=4) as executor:
        captions = list(executor.map(captioner.generate_caption, frames))
    return captions
```

### 2. Caching Results

```python
import pickle

# Save results
with open('results.pkl', 'wb') as f:
    pickle.dump(results, f)

# Load results
with open('results.pkl', 'rb') as f:
    results = pickle.load(f)
```

### 3. Progressive Loading

```python
# Process video in chunks
for chunk_start in range(0, video_duration, 30):
    chunk_keyframes = extract_keyframes(
        video_path,
        start_time=chunk_start,
        duration=30
    )
    # Process chunk
```

---

## API Integration Example

```python
from fastapi import FastAPI, File, UploadFile
from video.integrated_analysis import analyze_video_complete

app = FastAPI()

@app.post(\"/analyze-video/\")
async def analyze_video(file: UploadFile = File(...)):
    # Save uploaded file
    video_path = f\"temp/{file.filename}\"
    with open(video_path, \"wb\") as f:
        f.write(await file.read())

    # Analyze
    results = analyze_video_complete(
        video_path=video_path,
        keyframe_interval=2.0
    )

    # Return results
    return {
        \"total_frames\": len(results),
        \"duration\": results[-1]['timestamp'],
        \"frames\": results
    }
```

---

## Conclusion

This architecture documentation provides:

- ✅ Complete line-by-line code explanation
- ✅ Detailed model architecture diagrams
- ✅ Mathematical formulas and algorithms
- ✅ Testing procedures with expected outputs
- ✅ Performance metrics and optimization tips
- ✅ Troubleshooting guide

For questions or issues, refer to:

- LLaVA paper: https://arxiv.org/abs/2304.08485
- FaceNet paper: https://arxiv.org/abs/1503.03832
- MTCNN paper: https://arxiv.org/abs/1604.02878

---

**Last Updated:** November 19, 2025  
**Version:** 1.0.0  
**Author:** Aura ML Team
