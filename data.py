from datasets import load_dataset

# Load the DailyDialog dataset
dataset = load_dataset("daily_dialog", trust_remote_code=True)

train = dataset["train"]
test = dataset["test"]
validation = dataset["validation"]

print(dataset["test"][0])  # Example: Print first entry from training set