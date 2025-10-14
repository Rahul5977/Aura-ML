# Aura Notebook Quick Reference

## 🎯 Notebook Cells Guide

### Setup & Import (Cells 1-5)

| Cell | Type     | Purpose                       | Run Once?       |
| ---- | -------- | ----------------------------- | --------------- |
| 1    | Markdown | Introduction and overview     | -               |
| 2    | Markdown | Environment setup explanation | -               |
| 3    | Python   | Install packages              | ✅ Yes          |
| 4    | Markdown | Import explanation            | -               |
| 5    | Python   | Import libraries              | ✅ Each session |

### Audio Processing (Cells 6-11)

| Cell | Type     | Purpose                   | Execution Time |
| ---- | -------- | ------------------------- | -------------- |
| 6    | Markdown | Audio section intro       | -              |
| 7    | Python   | Create/load audio file    | < 1s           |
| 8    | Markdown | Load explanation          | -              |
| 9    | Python   | Load with librosa         | < 1s           |
| 10   | Markdown | Visualization explanation | -              |
| 11   | Python   | Plot waveform/spectrogram | 2-5s           |

### STT - Whisper (Cells 12-16)

| Cell | Type     | Purpose            | Execution Time     |
| ---- | -------- | ------------------ | ------------------ |
| 12   | Markdown | STT overview       | -                  |
| 13   | Markdown | Model loading      | -                  |
| 14   | Python   | Load Whisper model | 5-30s (first time) |
| 15   | Markdown | Transcription      | -                  |
| 16   | Python   | Transcribe audio   | 2-10s              |

### SER - Emotion (Cells 17-21)

| Cell | Type     | Purpose              | Execution Time     |
| ---- | -------- | -------------------- | ------------------ |
| 17   | Markdown | SER overview         | -                  |
| 18   | Markdown | Model loading        | -                  |
| 19   | Python   | Load emotion model   | 5-20s (first time) |
| 20   | Markdown | Analysis             | -                  |
| 21   | Python   | Classify + visualize | 1-5s               |

### NER - Entities (Cells 22-27)

| Cell | Type     | Purpose          | Execution Time |
| ---- | -------- | ---------------- | -------------- |
| 22   | Markdown | NER overview     | -              |
| 23   | Markdown | Model loading    | -              |
| 24   | Python   | Load spaCy model | < 1s           |
| 25   | Markdown | Extraction       | -              |
| 26   | Python   | Extract entities | < 1s           |
| 27   | Markdown | Visualization    | -              |
| 28   | Python   | Displacy render  | < 1s           |

### COMET - Reasoning (Cells 29-32)

| Cell | Type     | Purpose             | Execution Time      |
| ---- | -------- | ------------------- | ------------------- |
| 29   | Markdown | COMET overview      | -                   |
| 30   | Markdown | Model loading       | -                   |
| 31   | Python   | Load COMET model    | 10-60s (first time) |
| 32   | Markdown | Inference           | -                   |
| 33   | Python   | Generate inferences | 5-20s               |

### Orchestration (Cells 34-37)

| Cell | Type     | Purpose             | Execution Time |
| ---- | -------- | ------------------- | -------------- |
| 34   | Markdown | Orchestrator intro  | -              |
| 35   | Markdown | Function definition | -              |
| 36   | Python   | Define orchestrator | < 1s           |
| 37   | Markdown | Execution           | -              |
| 38   | Python   | Run full pipeline   | 10-40s         |

### Neo4j Persistence (Cells 39-43)

| Cell | Type     | Purpose                  | Execution Time |
| ---- | -------- | ------------------------ | -------------- |
| 39   | Markdown | Neo4j overview           | -              |
| 40   | Markdown | Query generation         | -              |
| 41   | Python   | Define query generator   | < 1s           |
| 42   | Markdown | Display queries          | -              |
| 43   | Python   | Generate & print queries | < 1s           |
| 44   | Markdown | Graph visualization      | -              |

### Conclusion (Cells 45-48)

| Cell | Type     | Purpose        | Notes     |
| ---- | -------- | -------------- | --------- |
| 45   | Markdown | Summary        | Read-only |
| 46   | Markdown | Future work    | Read-only |
| 47   | Markdown | Final thoughts | Read-only |

---

## 🔑 Key Functions

### `run_full_analysis_pipeline(audio_data, sampling_rate)`

**Purpose**: Orchestrate all AI models in optimal order

**Input**:

- `audio_data`: numpy array of audio samples
- `sampling_rate`: sample rate in Hz (typically 16000)

**Output**: Dictionary with structure:

```python
{
  'metadata': {...},
  'stt': {...},
  'ser': {...},
  'ner': {...},
  'comet': {...}
}
```

**Execution Order**:

1. Phase 1 (Parallel): STT + SER
2. Phase 2 (Sequential): NER → COMET
3. Aggregate results

---

### `generate_neo4j_queries(analysis_packet)`

**Purpose**: Generate Cypher queries for Neo4j

**Input**: Analysis packet (from orchestrator)

**Output**: List of Cypher query strings

**Query Types**:

1. CREATE Utterance node
2. MERGE Emotion node + relationship
3. MERGE Entity nodes + relationships (per entity)
4. CREATE Inference nodes + relationships
5. MERGE Conversation link

---

### `generate_comet_inference(text, relation_type)`

**Purpose**: Generate commonsense inferences

**Input**:

- `text`: Input text for reasoning
- `relation_type`: One of ['xReact', 'oReact', 'xWant', 'oWant', 'xEffect', 'oEffect']

**Output**: List of inference strings

**Fallback**: If COMET unavailable, returns rule-based inferences

---

## 📊 Variables Reference

### Global Variables (Available After Execution)

| Variable             | Type      | Cell | Description             |
| -------------------- | --------- | ---- | ----------------------- |
| `audio`              | ndarray   | 9    | Loaded audio samples    |
| `sr`                 | int       | 9    | Sample rate (16000 Hz)  |
| `audio_file_path`    | str       | 7    | Path to audio file      |
| `whisper_model`      | Model     | 14   | Loaded Whisper model    |
| `emotion_classifier` | Pipeline  | 19   | Emotion classifier      |
| `nlp`                | Language  | 24   | spaCy NLP pipeline      |
| `comet_model`        | Model     | 31   | COMET model (if loaded) |
| `comet_tokenizer`    | Tokenizer | 31   | COMET tokenizer         |
| `comet_available`    | bool      | 31   | COMET availability flag |

### Result Variables

| Variable          | Type | Cell | Description       |
| ----------------- | ---- | ---- | ----------------- |
| `transcription`   | str  | 16   | Transcribed text  |
| `stt_result`      | dict | 16   | STT results       |
| `ser_result`      | dict | 21   | SER results       |
| `ner_result`      | dict | 26   | NER results       |
| `comet_result`    | dict | 33   | COMET results     |
| `analysis_packet` | dict | 38   | Complete analysis |
| `queries`         | list | 43   | Neo4j queries     |

---

## 🎨 Visualization Outputs

### Audio Visualizations (Cell 11)

- **Waveform**: Amplitude over time
- **Spectrogram**: Frequency content (Hz)
- **Mel Spectrogram**: Perceptually-weighted frequencies

### Emotion Visualization (Cell 21)

- **Horizontal Bar Chart**: Confidence scores for all emotions
- **Color-coded**: Viridis colormap
- **Percentage Labels**: Displayed on bars

### Entity Visualization (Cell 28)

- **Displacy Render**: Inline HTML visualization
- **Color-coded Entities**: By type (PERSON, GPE, ORG, etc.)
- **Interactive**: Hover to see entity labels

---

## 🚀 Execution Workflows

### Full Notebook Run (First Time)

```
Total Time: ~5-10 minutes

1. Cell 3 (Install): 2-3 min
2. Cell 5 (Imports): 5-10 sec
3. Cell 7-11 (Audio): 5-10 sec
4. Cell 14 (Whisper): 10-30 sec (download)
5. Cell 16 (Transcribe): 2-10 sec
6. Cell 19 (SER Model): 5-20 sec (download)
7. Cell 21 (Emotion): 1-5 sec
8. Cell 24 (spaCy): < 1 sec
9. Cell 26 (NER): < 1 sec
10. Cell 31 (COMET): 10-60 sec (download)
11. Cell 33 (Inference): 5-20 sec
12. Cell 38 (Orchestrator): 10-40 sec
13. Cell 43 (Neo4j): < 1 sec
```

### Restart & Run All (Subsequent Times)

```
Total Time: ~2-4 minutes

1. Cell 5 (Imports): 5-10 sec
2. Cell 7-11 (Audio): 5-10 sec
3. Cell 14 (Whisper): < 5 sec (cached)
4. Cell 16 (Transcribe): 2-10 sec
5. Cell 19 (SER): < 2 sec (cached)
6. Cell 21 (Emotion): 1-5 sec
7. Cell 24 (spaCy): < 1 sec
8. Cell 26 (NER): < 1 sec
9. Cell 31 (COMET): < 5 sec (cached)
10. Cell 33 (Inference): 5-20 sec
11. Cell 38 (Orchestrator): 10-40 sec
12. Cell 43 (Neo4j): < 1 sec
```

### Test Single Component

Run only relevant cells:

**STT Only**: Cells 5, 7, 9, 14, 16  
**SER Only**: Cells 5, 7, 9, 19, 21  
**NER Only**: Cells 5, 14, 16, 24, 26  
**COMET Only**: Cells 5, 14, 16, 31, 33  
**Orchestrator**: Cells 5, 7, 9, 14, 19, 24, 31, 36, 38

---

## 🔧 Customization Cheat Sheet

### Change Audio Source

```python
# Cell 7 - Replace with:
audio_file_path = "path/to/your/audio.wav"
```

### Use Smaller Models (Faster)

```python
# Cell 14 - Whisper
whisper_model = whisper.load_model("tiny")  # or "base", "small"

# Cell 19 - Emotion
emotion_classifier = pipeline(
    "audio-classification",
    model="superb/wav2vec2-base-superb-er"
)
```

### Skip COMET (Faster Execution)

```python
# Cell 31 - Force fallback
comet_available = False
```

### Adjust Output Verbosity

```python
# In orchestrator (Cell 36)
def run_full_analysis_pipeline(audio_data, sampling_rate, verbose=False):
    if verbose:
        print(...)  # Add/remove print statements
```

### Save Intermediate Results

```python
# After any cell with results
import pickle
with open('stt_result.pkl', 'wb') as f:
    pickle.dump(stt_result, f)
```

---

## 🐛 Common Issues & Fixes

### Issue: "No module named 'whisper'"

**Fix**: Run Cell 3 (install packages)

### Issue: "Model not found"

**Fix**: Delete `~/.cache/huggingface` and re-run

### Issue: "CUDA out of memory"

**Fix**: Add `device=-1` to force CPU

### Issue: Notebook freezes on COMET

**Fix**: Restart kernel, skip Cell 31

### Issue: "No entities found"

**Fix**: Audio likely silent - use real speech audio

---

## 📈 Performance Optimization Tips

1. **Use GPU**: Speeds up all models by 2-10x
2. **Smaller Models**: Trade accuracy for speed
3. **Skip COMET**: Saves 5-20 seconds
4. **Batch Processing**: Process multiple audios together
5. **Cache Models**: Keep kernel running between runs

---

**Last Updated**: October 13, 2025  
**Version**: 1.0.0
