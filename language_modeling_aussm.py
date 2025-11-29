import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import sys
import os
# Add path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'extension-cpp'))  # Add this line
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ai-modules'))
from wavesAI.model.aussm import SSMSeq2Seq
# Add path for imports
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ai-modules'))
# from wavesAI.model.aussm import SSMSeq2Seq

# Configuration
BATCH_SIZE = 8
SEQ_LENGTH = 512
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load WikiText-2 dataset
print("Loading WikiText dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

vocab_size = len(tokenizer)
print("vocab_size: ", vocab_size) 
# exit()

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=SEQ_LENGTH + 1,  # +1 for shift
        padding="max_length",
        return_tensors="pt"
    )

print("Tokenizing dataset...")
tokenized_train = dataset["train"].map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)
tokenized_val = dataset["validation"].map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["validation"].column_names
)

# Create dataset class
class WikiTextDataset(Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_ids = torch.tensor(self.data[idx]["input_ids"], dtype=torch.long)
        # For language modeling: input is tokens[:-1], target is tokens[1:]
        return input_ids[:-1], input_ids[1:]

# Create data loaders
train_dataset = WikiTextDataset(tokenized_train)
val_dataset = WikiTextDataset(tokenized_val)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)
# Initialize model
print("Initializing model...")
model = SSMSeq2Seq(
    d_model=512,
    vocab_size=vocab_size,
    output_vocab_size=vocab_size,
    layers="m|m|a",
    d_state=16,
    verbose=True
).to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
# Training loop
print("Starting training...")
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        
        # Forward pass
        logits = model(inputs)  # (batch, seq_len, vocab_size)
        
        # Reshape for loss: (batch * seq_len, vocab_size) and (batch * seq_len,)
        loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
    
    # Validation
    model.eval()
    val_loss = 0
    val_batches = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
            
            val_loss += loss.item()
            val_batches += 1
    
    avg_val_loss = val_loss / val_batches
    print(f"Validation Loss: {avg_val_loss:.4f}\n")

print("Training completed!")