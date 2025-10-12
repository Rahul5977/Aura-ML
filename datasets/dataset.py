from datasets import load_dataset
import os

# Define the dataset name and the output directory
dataset_name = "thu-coai/esconv"
output_dir = "esconv_dataset"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the dataset from Hugging Face
print("Loading the dataset...")
dataset = load_dataset(dataset_name)
print("Dataset loaded successfully.")

# Save each split as a separate JSON file
for split in dataset.keys():
    file_path = os.path.join(output_dir, f"{split}.json")
    print(f"Saving the '{split}' split to {file_path}...")
    dataset[split].to_json(file_path)
    print(f"'{split}' split saved successfully.")

print("\nAll dataset splits have been downloaded and saved as JSON files.")