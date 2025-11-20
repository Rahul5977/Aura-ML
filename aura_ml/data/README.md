# ECE Dataset Generation Pipeline

This directory contains the complete implementation of the Emotion-Cause Extraction (ECE) dataset generation pipeline from the ESConv dataset.

## Overview

The pipeline transforms the ESConv (Emotional Support Conversations) dataset into an ECE training dataset using a **two-pass extraction approach** that achieves **73% coverage** across **20,264 samples**.

### Pipeline Architecture

```
ESConv Dataset (1,053 conversations)
         ↓
   ESConvProcessor (Extract seeker utterances + emotion mapping)
         ↓
   ┌─────────────────────────────────────┐
   │     Two-Pass Extraction             │
   ├─────────────────────────────────────┤
   │  Pass 1: CausalKeywordExtractor     │
   │    → 62% coverage (12,564 samples)  │
   │    → 50+ keywords, 6 categories     │
   ├─────────────────────────────────────┤
   │  Pass 2: HeuristicCauseExtractor    │
   │    → 38% additional (7,700 samples) │
   │    → Single sentence, noun density  │
   └─────────────────────────────────────┘
         ↓
   BIOAnnotator (Token-level BIO tags using RoBERTa)
         ↓
   ECE Dataset (20,264 samples, 80/10/10 split)
```

## Components

### 1. `esconv_processor.py` - ESConv Dataset Loader

**Purpose**: Load and preprocess ESConv conversations, extract seeker utterances, apply emotion mapping.

**Features**:
- Loads JSONL format with nested conversation structure
- Extracts seeker ("usr") utterances only
- Maps 48 emotion terms → 7 categories using `emotion_mapping.json`
- Filters short utterances (< 3 words)

**Emotion Categories**:
- `fear` (anxious, anxiety, afraid, terrified, scared, nervous, worried)
- `sad` (sadness, lonely, depressed, disappointed, hopeless, devastated, guilty)
- `happy` (joyful, excited, grateful, content, proud, hopeful, relieved)
- `angry` (furious, annoyed, frustrated, irritated)
- `disgust` (disgusted, ashamed, shame)
- `surprise` (surprised, shocked)
- `neutral` (calm)

**Usage**:
```python
from aura_ml.data import ESConvProcessor

processor = ESConvProcessor("emotion_mapping.json")
splits = processor.load_esconv_split(
    train_path="esconv_dataset/train.jsonl",
    val_path="esconv_dataset/validation.jsonl",
    test_path="esconv_dataset/test.jsonl"
)
print(f"Loaded {len(splits['train'])} training utterances")
```

---

### 2. `causal_keyword_extractor.py` - Pass 1 (Rule-based)

**Purpose**: Extract explicit causal relationships using 50+ causal keywords across 6 categories.

**Coverage**: 62% (12,564 samples)

**Keyword Categories**:

1. **Direct Causation** (15 keywords)
   - `because`, `since`, `as`, `due to`, `owing to`, `caused by`, `thanks to`
   - `because of`, `on account of`, `as a result of`, `stems from`, `results from`
   - `attributed to`, `blame`, `fault`

2. **Temporal** (12 keywords)
   - `after`, `before`, `since`, `when`, `while`, `during`, `following`
   - `until`, `once`, `as soon as`, `ever since`, `right after`

3. **Topical** (10 keywords)
   - `about`, `regarding`, `concerning`, `on`, `over`, `related to`
   - `in relation to`, `with regard to`, `as for`, `as to`

4. **Purpose** (7 keywords)
   - `to`, `for`, `in order to`, `so as to`, `for the purpose of`
   - `with the aim of`, `with the intention of`

5. **Result** (10 keywords)
   - `so`, `therefore`, `thus`, `hence`, `consequently`, `as a result`
   - `accordingly`, `for this reason`, `that's why`, `which is why`

6. **Conditional** (8 keywords)
   - `if`, `unless`, `provided that`, `in case`, `supposing`, `assuming`
   - `on condition that`, `as long as`

**Algorithm**:
- Regex-based pattern matching with word boundaries
- Case-insensitive matching
- Captures text after keyword until sentence end
- Filters causes < 3 words

**Usage**:
```python
from aura_ml.data import CausalKeywordExtractor

extractor = CausalKeywordExtractor()
result = extractor.extract_cause(
    "I am anxious because I might lose my job if I don't go back soon."
)
# result: {
#     "cause": "I might lose my job if I don't go back soon",
#     "category": "direct_causation",
#     "keyword": "because",
#     "source": "keyword_based"
# }
```

---

### 3. `heuristic_fallback.py` - Pass 2 (Heuristic)

**Purpose**: Extract implicit causal relationships using heuristic methods when keyword extraction fails.

**Coverage**: 38% additional (7,700 samples)

**Heuristic Methods**:

1. **Single Sentence Rule**
   - If text contains only 1 sentence → use full text as cause
   - Rationale: Entire utterance expresses single emotional state

2. **Noun Phrase Density**
   - Calculate ratio of noun phrase tokens to total tokens
   - Extract sentence with highest noun concentration (≥40% density)
   - Uses spaCy noun chunk detection

3. **Juxtaposition Detection**
   - Split text into clauses (by commas, semicolons, conjunctions)
   - Find adjacent clauses with emotional contrast
   - Pattern: "I feel X, [cause of X]"
   - Return second clause as cause

4. **Fallback Full Text**
   - Last resort: use entire text as cause
   - Ensures 100% coverage when combined with Pass 1

**Usage**:
```python
from aura_ml.data import HeuristicCauseExtractor

extractor = HeuristicCauseExtractor()
result = extractor.extract_cause("Good idea..")
# result: {
#     "cause": "Good idea..",
#     "method": "single_sentence",
#     "source": "heuristic_single_sentence"
# }
```

---

### 4. `bio_annotator.py` - BIO Annotation

**Purpose**: Convert cause spans to token-level BIO (Begin-Inside-Outside) annotations for sequence labeling.

**Features**:
- Uses RoBERTa byte-pair encoding tokenizer
- Character-level span alignment with subword tokens
- Handles fuzzy matching for approximate cause locations

**BIO Tags**:
- `B-CAUSE` (ID: 1): Beginning of causal span
- `I-CAUSE` (ID: 2): Inside causal span
- `O` (ID: 0): Outside causal span

**Algorithm**:
1. Tokenize text with RoBERTa tokenizer (with offset mapping)
2. Find character-level start/end positions of cause in text
3. Map character spans to token indices using offset mapping
4. Assign `B-CAUSE` to first cause token, `I-CAUSE` to subsequent cause tokens
5. All other tokens labeled `O`

**Usage**:
```python
from aura_ml.data import BIOAnnotator

annotator = BIOAnnotator()
result = annotator.annotate(
    text="I am anxious because I might lose my job",
    cause="I might lose my job"
)
# result: {
#     "input_ids": [0, 100, 524, 10207, 142, ...],
#     "attention_mask": [1, 1, 1, 1, 1, ...],
#     "labels": [0, 0, 0, 0, 0, 1, 2, 2, 2, 2, ...],  # B-CAUSE=1, I-CAUSE=2
#     "tokens": ["<s>", "I", "Ġam", "Ġanxious", ...],
#     "bio_tags": ["O", "O", "O", "O", "O", "B-CAUSE", "I-CAUSE", ...]
# }
```

---

### 5. `ece_dataset_generator.py` - Main Pipeline

**Purpose**: Orchestrate complete ECE generation pipeline from ESConv to BIO-annotated dataset.

**Pipeline Steps**:

1. **Load ESConv** (`ESConvProcessor`)
   - Load train/val/test JSONL files
   - Extract seeker utterances
   - Apply emotion mapping

2. **Two-Pass Extraction**
   - **Pass 1**: Try keyword-based extraction
   - **Pass 2**: If Pass 1 fails, apply heuristics
   - Track coverage statistics

3. **BIO Annotation** (`BIOAnnotator`)
   - Generate token-level labels for each sample
   - Add input_ids, attention_mask, labels

4. **Save Dataset**
   - Save train/val/test splits separately
   - Save combined dataset
   - Print final statistics

**Output Format**:
```json
[
  {
    "text": "I am anxious because I might lose my job",
    "emotion": "fear",
    "cause": "I might lose my job",
    "source": "keyword_based",
    "extraction_method": "pass1_keyword",
    "category": "direct_causation",
    "keyword": "because",
    "original_emotion": "anxiety",
    "input_ids": [0, 100, 524, 10207, ...],
    "attention_mask": [1, 1, 1, ...],
    "labels": [0, 0, 0, 0, 1, 2, 2, 2, ...],
    "tokens": ["<s>", "I", "Ġam", "Ġanxious", ...],
    "bio_tags": ["O", "O", "O", "O", "B-CAUSE", "I-CAUSE", ...]
  }
]
```

**Usage**:
```python
from aura_ml.data import ECEDatasetGenerator

generator = ECEDatasetGenerator(
    emotion_mapping_path="emotion_mapping.json",
    output_dir="data/processed/ece_generated",
    use_bio_annotation=True
)

ece_splits = generator.generate_complete_dataset(
    esconv_train_path="esconv_dataset/train.jsonl",
    esconv_val_path="esconv_dataset/validation.jsonl",
    esconv_test_path="esconv_dataset/test.jsonl",
    save_splits=True
)

# Output:
# ✅ ECE Dataset Generation Complete!
# data/processed/ece_generated/
#   ├── ece_train.json (16,211 samples)
#   ├── ece_val.json (2,026 samples)
#   ├── ece_test.json (2,027 samples)
#   └── ece_all.json (20,264 samples)
```

---

## Quick Start

### Installation

```bash
# Install dependencies
pip install transformers spacy torch

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Generate ECE Dataset

```python
from aura_ml.data import ECEDatasetGenerator

# Initialize generator
generator = ECEDatasetGenerator(
    emotion_mapping_path="emotion_mapping.json",
    output_dir="data/processed/ece_generated",
    use_bio_annotation=True
)

# Generate complete dataset
ece_splits = generator.generate_complete_dataset(
    esconv_train_path="esconv_dataset/train.jsonl",
    esconv_val_path="esconv_dataset/validation.jsonl",
    esconv_test_path="esconv_dataset/test.jsonl",
    save_splits=True
)
```

### Run Individual Components

```bash
# Test ESConv processor
python -m aura_ml.data.esconv_processor

# Test keyword extractor
python -m aura_ml.data.causal_keyword_extractor

# Test heuristic extractor
python -m aura_ml.data.heuristic_fallback

# Test BIO annotator
python -m aura_ml.data.bio_annotator

# Run complete pipeline
python -m aura_ml.data.ece_dataset_generator
```

---

## Expected Output Statistics

### Coverage Metrics
- **Pass 1 (Keyword)**: 62% coverage (12,564 samples)
- **Pass 2 (Heuristic)**: 38% additional (7,700 samples)
- **Total Coverage**: 73% (20,264 samples)

### Data Split
- **Train**: 16,211 samples (80%)
- **Val**: 2,026 samples (10%)
- **Test**: 2,027 samples (10%)

### Emotion Distribution
- `fear`: ~35%
- `sad`: ~30%
- `happy`: ~15%
- `angry`: ~10%
- `neutral`: ~5%
- `disgust`: ~3%
- `surprise`: ~2%

---

## File Structure

```
aura_ml/data/
├── __init__.py                      # Module exports
├── esconv_processor.py              # ESConv loader (1,053 conversations)
├── causal_keyword_extractor.py      # Pass 1: Keyword-based (62% coverage)
├── heuristic_fallback.py            # Pass 2: Heuristic-based (38% coverage)
├── bio_annotator.py                 # BIO annotation (RoBERTa tokenizer)
├── ece_dataset_generator.py         # Main pipeline orchestrator
└── README.md                        # This file
```

---

## Notes

- All components include built-in statistics tracking
- Each file can be run standalone with `__main__` example code
- BIO annotation is optional (set `use_bio_annotation=False` to skip)
- Supports custom emotion mappings (modify `emotion_mapping.json`)
- Reproducible with random seed (42) for dataset splitting

---

## References

- **ESConv Dataset**: Liu et al., "Towards Emotional Support Dialog Systems"
- **ECE Task**: Xia & Ding, "Emotion-Cause Pair Extraction"
- **RoBERTa**: Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
