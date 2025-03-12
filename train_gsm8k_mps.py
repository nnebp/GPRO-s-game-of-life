#!/usr/bin/env python
"""
MPS-accelerated GRPO training for Apple Silicon using the full GSM8K dataset.
This version is designed for more serious training with dataset options,
checkpointing, and better memory management.
"""

import os
import re
import argparse
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    get_linear_schedule_with_warmup,
    set_seed
)
from trl import GRPOConfig, GRPOTrainer

# ---- CHECK FOR MPS AVAILABILITY ----
mps_available = hasattr(torch, 'mps') and torch.backends.mps.is_available()
print(f"MPS (Metal Performance Shaders) available: {mps_available}")

# Use MPS if available, otherwise fall back to CPU
device = torch.device("mps" if mps_available else "cpu")
print(f"Using device: {device}")

# ---- SYSTEM PROMPT AND CONSTANTS ----
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# ---- UTILITY FUNCTIONS ----
def extract_hash_answer(text):
    """Extract the answer from text with #### format"""
    if "####" not in text:
        return ""
    return text.split("####")[1].strip()

def extract_xml_answer(text):
    """Extract the answer from XML format"""
    if "<answer>" not in text or "</answer>" not in text:
        return ""
    try:
        answer = text.split("<answer>")[-1].split("</answer>")[0].strip()
        return answer
    except:
        return ""

def load_gsm8k_dataset(args):
    """Load and prepare the GSM8K dataset"""
    print(f"Loading GSM8K dataset (split: {args.split})...")
    
    # Load the dataset
    data = load_dataset("openai/gsm8k", "main")[args.split]
    
    # Take only a subset if specified
    if args.max_samples and len(data) > args.max_samples:
        print(f"Taking subset of {args.max_samples} examples from {len(data)} total")
        # Use seed for reproducible sampling
        indices = np.random.RandomState(args.seed).choice(
            len(data), args.max_samples, replace=False
        )
        data = data.select(indices)
    else:
        print(f"Using all {len(data)} examples")
    
    # Format the data for GRPO training
    formatted_data = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )
    
    return formatted_data

# ---- REWARD FUNCTIONS ----
def correctness_reward(prompts, completions, answer, **kwargs):
    """Reward function for correct answers"""
    responses = [completion[0]["content"] for completion in completions]
    extracted = [extract_xml_answer(r) for r in responses]
    return [2.0 if r.strip() == a.strip() else 0.0 for r, a in zip(extracted, answer)]

def format_reward(completions, **kwargs):
    """Reward function for proper format"""
    pattern = r"<reasoning>[\s\S]*?</reasoning>\s*<answer>[\s\S]*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [bool(re.search(pattern, r)) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def numeric_answer_reward(completions, **kwargs):
    """Reward function that checks if the answer is numeric"""
    responses = [completion[0]["content"] for completion in completions]
    extracted = [extract_xml_answer(r) for r in responses]
    
    def is_numeric(text):
        # Remove commas, dollar signs, and whitespace
        text = text.replace(',', '').replace('$', '').strip()
        try:
            float(text)
            return True
        except:
            return False
            
    return [0.5 if is_numeric(r) else 0.0 for r in extracted]

# ---- STANDARD GRPO TRAINER ----
def train_with_grpo(args, dataset):
    """Train with GRPOTrainer using MPS-compatible settings"""
    print(f"Starting GRPO training with {args.model_name}...")
    
    # Set env var to tell PyTorch about MPS
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    # Configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure training arguments
    training_args = GRPOConfig(
        # Basic settings
        output_dir=args.output_dir,
        
        # CRITICAL: Disable mixed precision for MPS compatibility
        fp16=False,
        bf16=False,
        
        # Training parameters
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_ratio=0.1,
        
        # GRPO specific parameters
        num_generations=args.num_generations,  # Must evenly divide batch_size
        max_prompt_length=args.max_seq_length // 2,
        max_completion_length=args.max_seq_length // 2,
        beta=args.beta,  # Entropy coefficient
        
        # Control device placement
        no_cuda=True,  # Tell accelerator not to look for CUDA
        
        # Save & logging
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        report_to="none",
        seed=args.seed,
    )
    
    # Initialize trainer
    print("Initializing trainer...")
    try:
        trainer = GRPOTrainer(
            model=args.model_name,
            processing_class=tokenizer,
            reward_funcs=[correctness_reward, format_reward, numeric_answer_reward],
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )
        
        # Start training
        print(f"Starting training with {len(dataset)} examples...")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        
        # Save the model
        trainer.save_model(args.output_dir)
        print(f"Training complete! Model saved to {args.output_dir}")
        return True
        
    except Exception as e:
        print(f"\nError in GRPOTrainer: {e}")
        print("\nFalling back to custom training...")
        return False

# ---- CUSTOM TRAINING FUNCTION ----
def train_custom(args, dataset):
    """
    Custom training loop that explicitly uses MPS.
    This is a fallback if GRPOTrainer doesn't work.
    """
    print(f"Starting custom training loop with {args.model_name}...")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Move model to MPS device if available
    model = model.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create dataloaders
    from torch.utils.data import DataLoader, RandomSampler

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, tokenizer, max_length):
            self.dataset = dataset
            self.tokenizer = tokenizer
            self.max_length = max_length
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, idx):
            item = self.dataset[idx]
            question = item["prompt"][1]["content"]
            prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {question}\n\nAnswer:"
            
            encodings = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            return {
                "input_ids": encodings.input_ids[0],
                "attention_mask": encodings.attention_mask[0],
                "answer": item["answer"]
            }
    
    # Create dataset and dataloader
    train_dataset = TextDataset(dataset, tokenizer, args.max_seq_length)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size
    )
    
    # Setup learning rate scheduler
    total_steps = len(train_dataloader) * args.num_epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    # Training loop
    global_step = 0
    model.train()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(args.num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if k != "answer"}
            
            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["input_ids"]  # Use input as target for language modeling
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Update progress
            global_step += 1
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
            # Save checkpoint
            if global_step % args.save_steps == 0:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                print(f"Saved checkpoint to {checkpoint_dir}")
            
            # Check if we've reached max steps
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        
        # End of epoch
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Check if we've reached max steps
        if args.max_steps > 0 and global_step >= args.max_steps:
            break
    
    # Save the final model
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Training complete! Model saved to {args.output_dir}")

# ---- TEST FUNCTION ----
def test_model(model_path, questions):
    """Test the trained model with a list of questions"""
    try:
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Move to MPS if available
        model = model.to(device)
        
        print("\n===== MODEL EVALUATION =====")
        
        for i, question in enumerate(questions):
            print(f"\nTest Question {i+1}: {question}")
            
            # Format prompt
            prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {question}\n\nAnswer:"
            
            # Generate response
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
            )
            
            # Display result
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            result = response[len(prompt):]
            print("\nModel response:")
            print(result)
            
            # Try to extract and show the final answer
            answer = extract_xml_answer(result)
            if answer:
                print(f"\nExtracted answer: {answer}")
            
    except Exception as e:
        print(f"Error testing the model: {e}")
        return None

# ---- ARGUMENT PARSING ----
def parse_args():
    parser = argparse.ArgumentParser(description="Train a reasoning model on GSM8K with MPS")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="distilgpt2",
                        help="Model name or path (default: distilgpt2)")
    parser.add_argument("--output_dir", type=str, default="gsm8k_model",
                        help="Output directory for model and checkpoints")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Maximum sequence length")
    
    # Dataset parameters
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to use (train, test)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to use (None = all)")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of epochs (for custom training)")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Maximum number of training steps (0 = no limit)")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Ratio of warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Number of steps to accumulate gradients")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping")
    
    # GRPO parameters
    parser.add_argument("--num_generations", type=int, default=2,
                        help="Number of generations per prompt (must divide batch_size)")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="Entropy coefficient for GRPO")
    
    # Save and logging
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint every X steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every X steps")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Resume from checkpoint path")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--force_custom", action="store_true",
                        help="Force using custom training loop instead of GRPOTrainer")
    
    return parser.parse_args()

# ---- MAIN EXECUTION ----
if __name__ == "__main__":
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load dataset
    dataset = load_gsm8k_dataset(args)
    
    # Check if batch_size is divisible by num_generations
    if args.batch_size % args.num_generations != 0:
        print(f"Warning: batch_size ({args.batch_size}) must be divisible by "
              f"num_generations ({args.num_generations}). Adjusting batch_size.")
        args.batch_size = (args.batch_size // args.num_generations) * args.num_generations
        print(f"Adjusted batch_size to: {args.batch_size}")
    
    # Train the model
    success = False
    if not args.force_custom:
        # Try standard GRPO trainer first
        success = train_with_grpo(args, dataset)
    
    if not success or args.force_custom:
        # Fall back to custom training
        train_custom(args, dataset)
    
    # Test the model with some sample questions
    test_questions = [
        "If a rectangle has a length of 8 cm and a width of 5 cm, what is its area?",
        "Sarah has $120. She spends 1/3 of her money on a bag and 1/4 of the remainder on lunch. How much money does she have left?",
        "A train travels at 60 mph for 2 hours, then at 80 mph for 1.5 hours. How far did it travel in total?"
    ]
    
    test_model(args.output_dir, test_questions)