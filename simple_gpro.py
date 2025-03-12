#!/usr/bin/env python
"""
MPS-accelerated GRPO training for Apple Silicon Macs.
"""

import os
import re
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

# ---- CHECK FOR MPS AVAILABILITY ----
mps_available = hasattr(torch, 'mps') and torch.backends.mps.is_available()
print(f"MPS (Metal Performance Shaders) available: {mps_available}")

# Use MPS if available, otherwise fall back to CPU
device = torch.device("mps" if mps_available else "cpu")
print(f"Using device: {device}")

# ---- CONFIGURATION ----
MODEL_NAME = "distilgpt2"  # Smallest model (82M parameters)
OUTPUT_DIR = "tiny_model_mps"
MAX_STEPS = 30  # Very short training for demo

# ---- SYSTEM PROMPT ----
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# ---- MINIMAL DATASET ----
# Define a tiny dataset with just a few examples
TINY_DATASET = [
    {
        "question": "If a shirt costs $20 and is on sale for 25% off, how much does it cost?",
        "answer": "The shirt costs $20. With a 25% discount, I need to calculate 25% of $20, which is 0.25 × $20 = $5. Then I subtract this from the original price: $20 - $5 = $15. #### 15"
    },
    {
        "question": "John has 5 apples. He gives 2 apples to Mary and buys 3 more. How many apples does John have now?",
        "answer": "John starts with 5 apples. He gives 2 apples to Mary, so he has 5 - 2 = 3 apples left. Then he buys 3 more apples, so he now has 3 + 3 = 6 apples. #### 6"
    },
    {
        "question": "If 8 workers can build a house in 10 days, how many days would it take 4 workers to build the same house?",
        "answer": "With 8 workers, it takes 10 days to build a house. The total work is 8 workers × 10 days = 80 worker-days. If we have 4 workers instead, the time would be 80 worker-days ÷ 4 workers = 20 days. #### 20"
    },
    {
        "question": "A car travels at 60 miles per hour. How far will it travel in 2.5 hours?",
        "answer": "The car travels at 60 miles per hour for 2.5 hours. The distance is speed × time = 60 miles/hour × 2.5 hours = 150 miles. #### 150"
    }
]

# ---- UTILITY FUNCTIONS ----
def extract_answer(text):
    """Extract the answer from text with #### format"""
    if "####" not in text:
        return ""
    return text.split("####")[1].strip()

def extract_xml_answer(text):
    """Extract the answer from XML format"""
    if "<answer>" not in text or "</answer>" not in text:
        return ""
    answer = text.split("<answer>")[-1].split("</answer>")[0].strip()
    return answer

def prepare_dataset():
    """Prepare a tiny dataset for GRPO training"""
    data = [
        {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["question"]}
            ],
            "answer": extract_answer(item["answer"])
        }
        for item in TINY_DATASET
    ]
    return Dataset.from_list(data)

# ---- REWARD FUNCTIONS ----
def correctness_reward(prompts, completions, answer, **kwargs):
    """Reward function for correct answers"""
    responses = [completion[0]["content"] for completion in completions]
    extracted = [extract_xml_answer(r) for r in responses]
    return [1.0 if r.strip() == a.strip() else 0.0 for r, a in zip(extracted, answer)]

def format_reward(completions, **kwargs):
    """Reward function for proper format"""
    pattern = r"<reasoning>[\s\S]*?</reasoning>\s*<answer>[\s\S]*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [bool(re.search(pattern, r)) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

# ---- MANUAL TRAINING FUNCTION ----
def train_custom():
    """
    A simple custom training loop as an alternative to GRPOTrainer.
    This is a very simplified implementation for demonstration purposes.
    """
    print(f"Starting custom training loop with {MODEL_NAME}...")
    
    # Prepare dataset
    dataset = prepare_dataset()
    print(f"Dataset ready: {len(dataset)} examples")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Move model to MPS device if available
    model = model.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Simple training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        total_loss = 0
        
        for i, example in enumerate(dataset):
            # Create input prompt
            prompt = example["prompt"][1]["content"]  # Get the question
            full_prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {prompt}\n\nAnswer:"
            
            # Tokenize
            inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
            
            # Forward pass
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
            print(f"Epoch {epoch+1}, Example {i+1}/{len(dataset)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
    
    # Save the model
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Training complete! Model saved to {OUTPUT_DIR}")

# ---- STANDARD GRPO TRAINER ----
def train_with_grpo():
    """Attempt to train with GRPOTrainer using MPS-compatible settings"""
    print(f"Starting GRPO training with {MODEL_NAME}...")
    
    # Set env var to tell PyTorch about MPS
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    # Prepare dataset
    dataset = prepare_dataset()
    print(f"Dataset ready: {len(dataset)} examples")
    
    # Configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure training arguments
    training_args = GRPOConfig(
        # Basic settings
        output_dir=OUTPUT_DIR,
        
        # CRITICAL: Disable mixed precision for MPS compatibility
        fp16=False,
        bf16=False,
        
        # Training parameters
        per_device_train_batch_size=2,  # Must be divisible by num_generations
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        max_steps=MAX_STEPS,
        
        # GRPO specific parameters
        num_generations=2,  # Must evenly divide batch_size
        max_prompt_length=96,
        max_completion_length=96,
        
        # Control device placement
        no_cuda=True,  # Tell accelerator not to look for CUDA
        
        # Save & logging
        save_strategy="steps",
        save_steps=10,
        logging_steps=5,
        report_to="none",
    )
    
    # Initialize trainer
    print("Initializing trainer...")
    try:
        trainer = GRPOTrainer(
            model=MODEL_NAME,
            processing_class=tokenizer,
            reward_funcs=[correctness_reward, format_reward],
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )
        
        # Start training
        print("Training started...")
        trainer.train()
        
        # Save the model
        trainer.save_model(OUTPUT_DIR)
        print(f"Training complete! Model saved to {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"\nError in GRPOTrainer: {e}")
        print("\nFalling back to custom training...")
        train_custom()

# ---- TEST FUNCTION ----
def test_model(question):
    """Test the trained model with a question"""
    try:
        print(f"Testing model with question: {question}")
        
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR)
        tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
        
        # Move to MPS if available
        model = model.to(device)
        
        # Format prompt
        prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {question}\n\nAnswer:"
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
        )
        
        # Display result
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = response[len(prompt):]
        print("\nModel response:")
        print(result)
        return result
    except Exception as e:
        print(f"Error testing the model: {e}")
        return None

# ---- MAIN EXECUTION ----
if __name__ == "__main__":
    # First attempt with GRPOTrainer, with fallback to custom training
    train_with_grpo()
    
    # Test with a new question
    test_question = "If a rectangle has a length of 8 cm and a width of 5 cm, what is its area?"
    test_model(test_question)