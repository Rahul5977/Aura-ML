# Hypercontextual Dataset Generation for LLM Fine-tuning

## 📋 Overview

The **Hypercontextual Dataset** is a rich instruction-tuning dataset created by enriching ESConv conversations with multi-modal analysis outputs from the complete Aura-ML pipeline. This dataset is specifically designed for fine-tuning Large Language Models (LLMs) to provide empathetic emotional support.

### What Makes It "Hypercontextual"?

Each training sample includes **6 sources of contextual information**:

1. **Emotion Labels** - From ESConv annotations (7 emotions)
2. **Extracted Causes** - From trained ECE model inference
3. **Named Entities** - From spaCy NER (people, places, dates, etc.)
4. **Problem Types** - From heuristic classification (relationship, work, health, etc.)
5. **Conversation History** - Sliding window of previous turns
6. **Support Strategies** - From ESConv annotations (8 strategies)

This multi-modal context teaches the LLM to:
- Understand emotional states
- Identify root causes of distress
- Track conversation context
- Apply appropriate support strategies
- Provide personalized, empathetic responses

---

## 📊 Dataset Statistics

**Target Size**: ~31,247 instruction-completion pairs

**Dataset Split**:
- Training: 90% (~28,122 samples)
- Validation: 10% (~3,125 samples)

**Coverage**:
- Conversations: ~1,053 ESConv dialogues
- Emotions: 7 categories (sad, fear, happy, angry, neutral, disgust, surprise)
- Problem Types: 7 categories (emotional distress, relationship, work, academic, financial, health, general)
- Support Strategies: 8 types (Question, Reflection, Affirmation, Suggestions, etc.)

---

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      ESConv Dataset                                  │
│              (train.json, valid.json, test.json)                    │
│                   ~1,053 conversations                              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
     ┌──────────▼──────────┐       ┌─────────▼──────────┐
     │   ECE Model         │       │  spaCy NER         │
     │   Inference         │       │  en_core_web_sm    │
     │                     │       │                    │
     │ RoBERTaForECE       │       │ Extracts:          │
     │ Extracts emotion    │       │ - PERSON           │
     │ cause spans         │       │ - ORG              │
     │ using BIO tags      │       │ - GPE              │
     │                     │       │ - DATE             │
     │ Input: "I'm sad     │       │ - MONEY            │
     │         because I   │       │ - EVENT            │
     │         lost job"   │       │                    │
     │ Output: "I lost     │       │                    │
     │          job"       │       │                    │
     └──────────┬──────────┘       └─────────┬──────────┘
                │                            │
                └────────────┬───────────────┘
                             │
              ┌──────────────▼─────────────────┐
              │  Problem Type Classifier       │
              │  (Heuristic Keyword Matching)  │
              │                                │
              │  Keywords:                     │
              │  - relationship: friend,       │
              │    partner, family, etc.       │
              │  - work: job, boss,            │
              │    career, etc.                │
              │  - health: sick, doctor,       │
              │    hospital, etc.              │
              │  - academic: school, exam,     │
              │    study, etc.                 │
              │  - financial: money, debt,     │
              │    bills, etc.                 │
              │  - emotional_distress:         │
              │    (from emotion)              │
              └──────────────┬─────────────────┘
                             │
              ┌──────────────▼─────────────────┐
              │  Context Manager                │
              │  (Sliding Window = 3 turns)    │
              │                                │
              │  Maintains conversation        │
              │  history:                      │
              │  Turn -3: User said...         │
              │  Turn -2: Aura said...         │
              │  Turn -1: User said...         │
              │  Turn  0: Current turn         │
              └──────────────┬─────────────────┘
                             │
              ┌──────────────▼─────────────────┐
              │  Instruction Generator          │
              │                                │
              │  Creates training samples:     │
              │  {                             │
              │    instruction: "You are       │
              │      Aura...",                 │
              │    input: {                    │
              │      user_message,             │
              │      emotion,                  │
              │      cause,                    │
              │      entities,                 │
              │      history,                  │
              │      problem_type              │
              │    },                          │
              │    output: "supporter          │
              │             response",         │
              │    metadata: {...}             │
              │  }                             │
              └──────────────┬─────────────────┘
                             │
              ┌──────────────▼─────────────────┐
              │  Hypercontextual Dataset       │
              │                                │
              │  llm_training_data.json        │
              │  llm_train.json (90%)          │
              │  llm_val.json (10%)            │
              │  dataset_statistics.json       │
              └────────────────────────────────┘
```

---

## 📝 Dataset Format

### Sample Structure

```json
{
  "instruction": "You are Aura, an empathetic AI assistant specialized in emotional support. Your user is feeling sad. They are saying: 'I've been really down lately because I lost my job last week'. The main reason they feel this way is: 'I lost my job last week'. Respond by asking open-ended questions to gather more information.",
  
  "input": {
    "user_message": "I've been really down lately because I lost my job last week",
    "emotion": "sad",
    "cause": "I lost my job last week",
    "entities": "last week (DATE)",
    "history": "User: Things have been rough | Aura: I'm here to listen. What's been going on? | User: I've been really down lately because I lost my job last week",
    "problem_type": "work"
  },
  
  "output": "I'm so sorry to hear that. Losing a job can be really difficult. How are you coping with this situation? What has been the hardest part for you?",
  
  "metadata": {
    "conversation_id": "conv_123",
    "strategy_used": "Question",
    "turn_index": 5
  }
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `instruction` | String | Complete task description with context for the LLM. Includes emotion state, user message, extracted cause, and target strategy. |
| `input.user_message` | String | The exact text the user/seeker said in the conversation. |
| `input.emotion` | String | Detected emotion (sad, fear, happy, angry, neutral, disgust, surprise). |
| `input.cause` | String | Extracted cause of emotion from ECE model. Falls back to full text if no specific cause detected. |
| `input.entities` | String | Named entities extracted by spaCy (PERSON, ORG, GPE, DATE, MONEY, EVENT). "None" if no entities found. |
| `input.history` | String | Formatted conversation history (sliding window of 3 turns). Format: "Speaker: text \| Speaker: text". |
| `input.problem_type` | String | Classified problem category (relationship, work, health, academic, financial, emotional_distress, general). |
| `output` | String | Ground truth supporter response - what the supporter actually said in ESConv. |
| `metadata.conversation_id` | String | ESConv conversation identifier for tracking. |
| `metadata.strategy_used` | String | Support strategy used (Question, Restatement or Paraphrasing, Reflection of feelings, Self-disclosure, Affirmation and Reassurance, Providing Suggestions, Information, Others). |
| `metadata.turn_index` | Int | Position of this turn in the conversation (0-indexed). |

---

## 🔧 Key Components

### 1. ECE Model Inference

**Purpose**: Extract emotion causes from user messages

**Model**: RoBERTaForECE (trained ECE classifier)

**Process**:
1. Tokenize input text with RoBERTa tokenizer
2. Run forward pass through ECE model
3. Get clause-level prediction (binary: has cause / no cause)
4. Get token-level BIO predictions (B-CAUSE, I-CAUSE, O)
5. Decode cause tokens to text
6. Return extracted cause (or full text if no cause detected)

**Example**:
```python
Input:  "I feel anxious because I have a big presentation tomorrow"
Output: "I have a big presentation tomorrow"
```

### 2. Named Entity Recognition (NER)

**Tool**: spaCy `en_core_web_sm` model

**Extracted Entity Types**:
- **PERSON**: Names of people
- **ORG**: Organizations, companies
- **GPE**: Geopolitical entities (countries, cities)
- **DATE**: Dates and times
- **MONEY**: Monetary values
- **EVENT**: Named events

**Example**:
```python
Text: "I'm meeting with Dr. Smith at Stanford Hospital on Monday about the surgery"

Entities: "Dr. Smith (PERSON), Stanford Hospital (ORG), Monday (DATE)"
```

### 3. Problem Type Classification

**Method**: Heuristic keyword matching

**Categories & Keywords**:

```python
relationship: ['friend', 'relationship', 'partner', 'family', 'girlfriend', 
               'boyfriend', 'spouse', 'husband', 'wife', 'parent', 'sibling',
               'breakup', 'divorce', 'argument', 'fight']

work: ['job', 'work', 'career', 'boss', 'colleague', 'coworker', 'office',
       'manager', 'workplace', 'fired', 'quit', 'promotion', 'interview']

health: ['health', 'sick', 'illness', 'hospital', 'doctor', 'disease',
         'pain', 'medical', 'appointment', 'diagnosis', 'treatment']

academic: ['school', 'exam', 'test', 'study', 'university', 'college',
           'grade', 'homework', 'assignment', 'professor', 'class', 'course']

financial: ['money', 'debt', 'financial', 'rent', 'bills', 'loan',
            'payment', 'salary', 'income', 'expense', 'budget', 'afford']

emotional_distress: (inferred from sad, fear, angry, disgust emotions)

general: (default if no keywords match)
```

**Example**:
```python
Text: "I'm stressed about my final exams next week"
Classification: "academic"
```

### 4. Conversation History

**Method**: Sliding window of last N turns (default: 3)

**Format**: `"Speaker: text | Speaker: text | Speaker: text"`

**Example**:
```
User: I've been feeling really overwhelmed lately
Aura: I hear you. What's been overwhelming you?
User: Work has been really demanding and I can't keep up
```

Formatted as:
```
"User: I've been feeling really overwhelmed lately | Aura: I hear you. What's been overwhelming you? | User: Work has been really demanding and I can't keep up"
```

### 5. Support Strategies

**Source**: ESConv annotations

**8 Strategy Types**:

| Strategy | Description | Example |
|----------|-------------|---------|
| **Question** | Asking open-ended questions | "What's been bothering you most about this situation?" |
| **Restatement or Paraphrasing** | Restating user's words | "So you're feeling stressed about your workload, is that right?" |
| **Reflection of feelings** | Validating emotions | "It sounds like you're feeling really overwhelmed and frustrated." |
| **Self-disclosure** | Sharing personal experiences | "I've been in a similar situation, and I know how hard it can be." |
| **Affirmation and Reassurance** | Providing encouragement | "You're doing the best you can, and that's what matters." |
| **Providing Suggestions** | Offering practical advice | "Have you considered talking to your manager about redistributing tasks?" |
| **Information** | Providing educational content | "Stress management techniques like deep breathing can help in these situations." |
| **Others** | General emotional support | "I'm here for you. You're not alone in this." |

---

## 🚀 Usage Guide

### Installation Requirements

```bash
# Install required packages
pip install torch transformers spacy

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Generate Dataset

#### Option 1: Using the Example Script

```bash
python examples/generate_hypercontextual_dataset.py
```

**Configuration** (edit the script):
```python
ESCONV_PATH = "esconv_dataset-20251120T185554Z-1-001/esconv_dataset"
ECE_MODEL_PATH = "data/models/ece/ece_roberta_model"
OUTPUT_DIR = "data/processed/hypercontextual"
HISTORY_WINDOW = 3
TRAIN_SPLIT = 0.9
```

#### Option 2: Using Python API

```python
from aura_ml.data import HypercontextualDatasetGenerator

# Initialize generator
generator = HypercontextualDatasetGenerator(
    ece_model_path="data/models/ece/ece_roberta_model",
    history_window=3,
    device='cuda'  # or 'cpu'
)

# Generate dataset
stats = generator.generate_dataset(
    esconv_path="esconv_dataset-20251120T185554Z-1-001/esconv_dataset",
    output_dir="data/processed/hypercontextual",
    train_split=0.9
)

print(f"Generated {stats['total_samples']} samples")
```

### Output Files

After generation, you'll have:

```
data/processed/hypercontextual/
├── llm_training_data.json     # All samples (100%)
├── llm_train.json              # Training split (90%)
├── llm_val.json                # Validation split (10%)
└── dataset_statistics.json     # Statistics and distributions
```

---

## 📈 Expected Statistics

Based on the report's mention of **31,247 instruction-completion pairs**, here's what to expect:

### Sample Counts
- **Total**: ~31,247 samples
- **Training**: ~28,122 samples (90%)
- **Validation**: ~3,125 samples (10%)

### Emotion Distribution (Estimated)
```
sad:       ~7,800 (25%)
fear:      ~6,900 (22%)
neutral:   ~4,700 (15%)
angry:     ~4,700 (15%)
happy:     ~3,750 (12%)
disgust:   ~1,560 (5%)
surprise:  ~1,875 (6%)
```

### Problem Type Distribution (Estimated)
```
emotional_distress: ~10,936 (35%)
relationship:       ~6,249 (20%)
work:               ~5,624 (18%)
academic:           ~4,687 (15%)
financial:          ~2,187 (7%)
health:             ~937 (3%)
general:            ~625 (2%)
```

### Strategy Distribution (Estimated)
```
Question:                      ~7,800 (25%)
Reflection of feelings:        ~6,250 (20%)
Affirmation and Reassurance:   ~5,600 (18%)
Providing Suggestions:         ~4,700 (15%)
Restatement or Paraphrasing:   ~3,125 (10%)
Information:                   ~1,875 (6%)
Self-disclosure:               ~1,250 (4%)
Others:                        ~625 (2%)
```

---

## 🎯 Use Cases

### 1. LLM Fine-tuning

Use this dataset to fine-tune instruction-following LLMs:

**Supported Models**:
- LLaMA 2/3 (7B, 13B, 70B)
- Mistral (7B)
- Phi-2/3
- GPT-2 (medium, large)
- Any causal language model

**Fine-tuning Approach**:
```python
# Format for training
for sample in dataset:
    prompt = f"{sample['instruction']}\n\nInput: {json.dumps(sample['input'])}"
    completion = sample['output']
    
    # Add to training data
    train_sample = {
        "prompt": prompt,
        "completion": completion
    }
```

### 2. Multi-task Learning

Train models to:
- Detect emotions
- Extract causes
- Classify problem types
- Select appropriate support strategies
- Generate empathetic responses

### 3. Context-aware Dialogue Systems

Use the rich context to build dialogue systems that:
- Track conversation history
- Understand user problems holistically
- Adapt responses based on problem type
- Apply appropriate support strategies

---

## 🔬 Quality Assurance

### Data Quality Checks

The generator includes:

1. **Text Validation**: Non-empty messages only
2. **Response Matching**: Only samples with supporter responses
3. **Cause Extraction**: Fallback to full text if ECE fails
4. **Entity Extraction**: Graceful handling of no entities
5. **Strategy Labeling**: All samples have valid strategy labels

### Statistics Tracking

Generated statistics include:
- Sample counts per split
- Emotion distribution
- Problem type distribution
- Support strategy distribution

### Sample Inspection

Always inspect generated samples:

```python
import json

with open('data/processed/hypercontextual/llm_train.json', 'r') as f:
    samples = json.load(f)

# Check first sample
print(json.dumps(samples[0], indent=2))
```

---

## 📚 Integration with Aura-ML Pipeline

### Complete Training Pipeline

```
Step 1: ECE Dataset Generation
  ↓ (esconv → ece_train.json, ece_val.json, ece_test.json)
  
Step 2: ECE Model Training
  ↓ (train RoBERTaForECE on ECE dataset)
  
Step 3: Hypercontextual Dataset Generation  ← YOU ARE HERE
  ↓ (esconv + ECE model → llm_train.json, llm_val.json)
  
Step 4: LLM Fine-tuning
  ↓ (fine-tune LLaMA/Mistral/Phi on hypercontextual dataset)
  
Step 5: Deployment
  ↓ (deploy fine-tuned LLM in Aura chatbot)
```

### Required Models

Before generating the hypercontextual dataset, ensure you have:

1. ✅ **Trained ECE Model**: `data/models/ece/ece_roberta_model/`
   - Train using ECE dataset generation pipeline
   - See `docs/ECE_COMPLETE_PIPELINE.md`

2. ✅ **spaCy NER Model**: `en_core_web_sm`
   - Auto-downloaded if not present
   - Or: `python -m spacy download en_core_web_sm`

3. ✅ **ESConv Dataset**: Raw conversation data
   - Should have `train.json`, `valid.json`, `test.json`

---

## 🐛 Troubleshooting

### Issue: "ECE model not found"

**Solution**: Train the ECE model first
```bash
python examples/generate_ece_dataset.py
# Then train the model using the generated ECE dataset
```

### Issue: "spaCy model not found"

**Solution**: Download spaCy model
```bash
python -m spacy download en_core_web_sm
```

### Issue: "CUDA out of memory"

**Solution**: Use CPU instead
```python
generator = HypercontextualDatasetGenerator(
    ece_model_path="...",
    device='cpu'  # Force CPU usage
)
```

### Issue: "No samples generated"

**Solution**: Check ESConv format
- Ensure ESConv has correct structure
- Check for `dialog`, `speaker`, `text` fields
- Verify emotion and strategy annotations exist

---

## 📖 References

### Papers & Datasets

1. **ESConv**: Emotional Support Conversation Dataset
   - Paper: https://arxiv.org/abs/2010.01441
   - GitHub: https://github.com/thu-coai/ESConv

2. **Emotion-Cause Extraction**: 
   - Multiple approaches for extracting emotion causes from text
   - BIO tagging for span extraction

### Libraries

- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face transformers library
- **spaCy**: Industrial-strength NLP
- **RoBERTa**: Robustly Optimized BERT

---

## 🤝 Contributing

To extend the hypercontextual dataset generator:

1. **Add New Context Sources**: Modify `process_conversation()` to extract additional features
2. **Improve Problem Classification**: Enhance keyword lists in `PROBLEM_KEYWORDS`
3. **Custom Instruction Templates**: Modify `create_instruction()` for different prompting styles
4. **Additional Entity Types**: Update `extract_entities()` to include more spaCy entity types
5. **Advanced History Formatting**: Enhance `format_history()` for better context representation

---

## ⚖️ License

Part of the Aura-ML project. See main repository for license details.

---

## 📝 Citation

If you use this dataset in your research, please cite:

```bibtex
@misc{aura-ml-hypercontextual,
  title={Hypercontextual Dataset for Empathetic AI Training},
  author={Aura-ML Team},
  year={2025},
  howpublished={\url{https://github.com/Rahul5977/Aura-ML}}
}
```

---

**Version**: 1.0  
**Last Updated**: November 21, 2025  
**Maintainer**: Aura-ML Development Team
