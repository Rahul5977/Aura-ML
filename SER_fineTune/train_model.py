import os
import numpy as np
import pandas as pd
import torch
import evaluate
from datasets import Dataset, Audio, ClassLabel
from transformers import (
    AutoFeatureExtractor, 
    AutoModelForAudioClassification, 
    TrainingArguments, 
    Trainer
)

# --- CONFIGURATION ---
MODEL_ID = "superb/wav2vec2-base-superb-er"
DATA_PATH = "Audio_Speech_Actors_01-24" 
OUTPUT_DIR = "emotion_model_finetuned"

# RAVDESS Emotion codes
emotion_map = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

def main():
    # 1. LOAD DATA
    print(f"🔍 Scanning {DATA_PATH}...")
    file_paths = []
    labels = []

    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith(".wav"):
                # Filename format: 03-01-06-01-02-01-12.wav
                parts = file.split("-")
                if len(parts) > 2:
                    emotion_code = parts[2]
                    if emotion_code in emotion_map:
                        file_paths.append(os.path.join(root, file))
                        labels.append(emotion_map[emotion_code])

    print(f"✅ Found {len(file_paths)} audio files.")
    
    # Create Dataset
    df = pd.DataFrame({"file": file_paths, "label": labels})
    dataset = Dataset.from_pandas(df)
    
    # Split Data (80% Train, 20% Test)
    dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
    
    # Setup Labels
    label_list = sorted(list(emotion_map.values()))
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    # Convert label text to numbers
    dataset = dataset.class_encode_column("label")

    # 2. PREPROCESS AUDIO
    print("⚙️  Loading Feature Extractor...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    
    # IMPORTANT: Resample audio to 16kHz (required by model)
    dataset = dataset.cast_column("file", Audio(sampling_rate=16000))

    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples["file"]]
        inputs = feature_extractor(
            audio_arrays, 
            sampling_rate=16000, 
            max_length=16000 * 3, # Cut audio to 3 seconds to save memory
            truncation=True,
            padding=True
        )
        return inputs

    print("⏳ Processing audio files (this might take a moment)...")
    encoded_dataset = dataset.map(preprocess_function, remove_columns=["file"], batched=True)

    # 3. LOAD MODEL
    print("🤖 Loading Model...")
    model = AutoModelForAudioClassification.from_pretrained(
        MODEL_ID, 
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True
    )

    # 4. SETUP TRAINING
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions[0], axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=4, # Small batch size for laptops
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        fp16=True  ,
        gradient_checkpointing=True,
        num_train_epochs=3,             # We will train for 3 loops
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        dataloader_num_workers=0        # Avoid Windows/Linux multiprocessing issues
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
    )

    # 5. START TRAINING
    print("🚀 Starting Training!")
    trainer.train()
    
    print("🎉 Training Complete! Saving model...")
    trainer.save_model(OUTPUT_DIR + "/final")
    print(f"Model saved to {OUTPUT_DIR}/final")

if __name__ == "__main__":
    main()