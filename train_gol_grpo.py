#!/usr/bin/env python
"""
GRPO training for Game of Life reasoning with both MPS and CUDA support.
This script fine-tunes a language model to reason about Conway's Game of Life
using GRPO and optionally LoRA for parameter-efficient fine-tuning.
"""

import os
import re
import torch
import numpy as np
from datasets import Dataset, load_dataset
from tqdm import tqdm
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from trl import GRPOConfig, GRPOTrainer
from game_of_life import GameOfLife  # Import the Game of Life implementation

# ---- CONFIGURATION ----
# Edit these parameters instead of using command-line arguments

# Model configuration
MODEL_NAME = "distilgpt2"  # Model to fine-tune
OUTPUT_DIR = "game_of_life_model"  # Where to save the model
MAX_SEQ_LENGTH = 512  # Maximum sequence length

# Dataset configuration
DATA_PATH = "data/game_of_life.parquet"  # Path to dataset
RECREATE_DATASET = True  # Force recreation of dataset even if it exists
NUM_SAMPLES = 10000  # Number of samples to generate
BOARD_SIZE = 5  # Size of Game of Life board (NxN)
VAL_SPLIT = 0.1  # Validation split ratio

# Training configuration
BATCH_SIZE = 4  # Batch size per device
LEARNING_RATE = 5e-5  # Initial learning rate
MAX_STEPS = 1000  # Maximum number of training steps (0 = no limit)
WARMUP_RATIO = 0.1  # Ratio of warmup steps
GRADIENT_ACCUMULATION_STEPS = 2  # Number of steps to accumulate gradients

# GRPO configuration
NUM_GENERATIONS = 2  # Number of generations per prompt (must divide batch_size)
BETA = 0.1  # Entropy coefficient for GRPO

# LoRA configuration
USE_LORA = False  # Disabled LoRA for now
LORA_R = 16
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]

# Save and logging configuration
SAVE_STEPS = 100  # Save checkpoint every X steps
LOGGING_STEPS = 10  # Log every X steps
EVAL_STEPS = 100  # Evaluate every X steps
RESUME_FROM_CHECKPOINT = None  # Path to resume from checkpoint
DISABLE_WANDB = True  # Disable Weights & Biases logging

# Misc configuration
SEED = 42  # Random seed

# ---- CHECK FOR AVAILABLE DEVICES ----
device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA available. Using device: {device}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"MPS (Metal Performance Shaders) available. Using device: {device}")
else:
    device = torch.device("cpu")
    print(f"No GPU available. Using CPU.")

# ---- SYSTEM PROMPT ----
SYSTEM_PROMPT = """
You are a reasoning assistant that specializes in Conway's Game of Life.
Given a Game of Life board state, you will explain the next state.

Respond in the following format:
<reasoning>
Analyze the board state and explain how each cell will change according to the rules:
1. Any live cell with fewer than two live neighbors dies (underpopulation)
2. Any live cell with two or three live neighbors lives
3. Any live cell with more than three live neighbors dies (overpopulation)
4. Any dead cell with exactly three live neighbors becomes alive (reproduction)
</reasoning>
<answer>
[The next board state]
</answer>
"""

# ---- UTILITY FUNCTIONS ----
def extract_xml_answer(text):
    """Extract the answer from XML format"""
    if "<answer>" not in text or "</answer>" not in text:
        return ""
    try:
        answer = text.split("<answer>")[-1].split("</answer>")[0].strip()
        return answer
    except:
        return ""

def create_or_load_dataset():
    """Create or load the Game of Life dataset"""
    if not os.path.exists(DATA_PATH) or RECREATE_DATASET:
        print(f"Creating new Game of Life dataset with {NUM_SAMPLES} samples...")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        
        questions = []
        answers = []
        explanations = []
        
        for i in tqdm(range(NUM_SAMPLES)):
            # Create a random problem
            size = (BOARD_SIZE, BOARD_SIZE)
            density = np.random.randint(2, 4) * 0.1
            game = GameOfLife(size=size)
            game.random_board(density=density)
            
            # Get the current state as the question
            game.next()  # Ensure it's valid
            problem = str(game)
            
            # Get the explanation and next state
            explanation = game.next_with_explanation()
            solution = str(game)
            
            questions.append(problem)
            answers.append(solution)
            explanations.append(explanation)
        
        # Create dataset
        data = {
            "question": questions,
            "answer": answers,
            "explanation": explanations
        }
        
        dataset = Dataset.from_dict(data)
        
        # Save dataset
        dataset.to_parquet(DATA_PATH)
        print(f"Dataset saved to {DATA_PATH}")
    else:
        print(f"Loading existing dataset from {DATA_PATH}...")
        dataset = load_dataset("parquet", data_files=DATA_PATH)["train"]
    
    # Format the data for GRPO training
    formatted_data = dataset.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Board state:\n{x['question']}\n\nWhat will be the next state?"},
            ],
            "answer": x["answer"],
            "explanation": x["explanation"]
        }
    )
    
    # Split into train and validation
    train_test_split = formatted_data.train_test_split(
        test_size=VAL_SPLIT, seed=SEED
    )
    
    return train_test_split["train"], train_test_split["test"]

# ---- REWARD FUNCTIONS ----
def correctness_reward(prompts, completions, answer, **kwargs):
    """Reward function for correct Game of Life next states"""
    responses = [completion[0]["content"] for completion in completions]
    extracted = [extract_xml_answer(r) for r in responses]
    
    # Compare the extracted answers with the ground truth
    rewards = []
    for resp, ans in zip(extracted, answer):
        # Remove whitespace and standardize the format
        resp_clean = re.sub(r'\s+', '', resp)
        ans_clean = re.sub(r'\s+', '', ans)
        
        if resp_clean == ans_clean:
            rewards.append(2.0)  # Perfect match
        else:
            # Check for partial correctness (percentage of matching cells)
            # This would need to be customized based on how Game of Life boards are represented
            # For now, just give 0
            rewards.append(0.0)
    
    return rewards

def format_reward(completions, **kwargs):
    """Reward function for proper format"""
    pattern = r"<reasoning>[\s\S]*?</reasoning>\s*<answer>[\s\S]*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [bool(re.search(pattern, r)) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def reasoning_quality_reward(prompts, completions, answer, explanation, **kwargs):
    """Reward function for quality of reasoning"""
    responses = [completion[0]["content"] for completion in completions]
    
    rewards = []
    for resp, exp in zip(responses, explanation):
        # Check if reasoning section exists
        if "<reasoning>" in resp and "</reasoning>" in resp:
            reasoning = resp.split("<reasoning>")[-1].split("</reasoning>")[0].strip()
            
            # Check for key phrases or concepts that should be in the reasoning
            key_concepts = [
                "live cell", "dead cell", "neighbor", "underpopulation", 
                "overpopulation", "reproduction", "survives", "dies", "born"
            ]
            
            # Count how many key concepts are mentioned
            concept_count = sum(1 for concept in key_concepts if concept.lower() in reasoning.lower())
            
            # Reward based on concept coverage
            rewards.append(min(0.5, concept_count / len(key_concepts)))
        else:
            rewards.append(0.0)
    
    return rewards

# ---- GRPO TRAINING FUNCTION ----
def train_with_grpo(train_dataset, val_dataset=None):
    """Train with GRPOTrainer"""
    print(f"Starting GRPO training with {MODEL_NAME}...")
    
    # Configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add a chat template to the tokenizer (required for GRPO)
    tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'system' %}\n{{ message['content'] }}\n{% elif message['role'] == 'user' %}\nUser: {{ message['content'] }}\n{% elif message['role'] == 'assistant' %}\nAssistant: {{ message['content'] }}\n{% endif %}\n{% endfor %}\n{% if add_generation_prompt %}\nAssistant: {% endif %}"
    
    # Setup LoRA if enabled
    peft_config = None
    if USE_LORA:
        print("Using LoRA for parameter-efficient fine-tuning")
        peft_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            task_type="CAUSAL_LM",
            lora_dropout=LORA_DROPOUT,
        )
    
    # Configure training arguments
    training_args = GRPOConfig(
        # Basic settings
        output_dir=OUTPUT_DIR,
        
        # Handle mixed precision based on device
        fp16=device.type == "cuda",
        bf16=False,
        
        # Training parameters
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        max_steps=MAX_STEPS,
        warmup_ratio=WARMUP_RATIO,
        
        # GRPO specific parameters
        num_generations=NUM_GENERATIONS,
        max_prompt_length=MAX_SEQ_LENGTH // 2,
        max_completion_length=MAX_SEQ_LENGTH // 2,
        beta=BETA,
        
        # Control device placement
        no_cuda=device.type != "cuda",
        
        # Save & logging
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=EVAL_STEPS if val_dataset else None,
        report_to="none" if DISABLE_WANDB else "wandb",
        seed=SEED,
    )
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = GRPOTrainer(
        model=MODEL_NAME,
        processing_class=tokenizer,
        reward_funcs=[correctness_reward, format_reward, reasoning_quality_reward],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
    )
    
    # Start training
    print(f"Starting training with {len(train_dataset)} examples...")
    trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)
    
    # Save the model
    trainer.save_model(OUTPUT_DIR)
    print(f"Training complete! Model saved to {OUTPUT_DIR}")

# ---- TEST FUNCTION ----
def test_model(model_path, test_problems):
    """Test the trained model with a list of Game of Life problems"""
    try:
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Move to appropriate device
        model = model.to(device)
        
        print("\n===== MODEL EVALUATION =====")
        
        for i, problem in enumerate(test_problems):
            print(f"\nTest Problem {i+1}:\n{problem}")
            
            # Format prompt
            prompt = f"{SYSTEM_PROMPT}\n\nBoard state:\n{problem}\n\nWhat will be the next state?"
            
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
                print(f"\nExtracted answer (next state):\n{answer}")
            
    except Exception as e:
        print(f"Error testing the model: {e}")
        return None

# ---- MAIN EXECUTION ----
if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(SEED)
    
    # Check if batch_size is divisible by num_generations
    if BATCH_SIZE % NUM_GENERATIONS != 0:
        print(f"Warning: BATCH_SIZE ({BATCH_SIZE}) must be divisible by "
              f"NUM_GENERATIONS ({NUM_GENERATIONS}). Adjusting batch_size.")
        BATCH_SIZE = (BATCH_SIZE // NUM_GENERATIONS) * NUM_GENERATIONS
        print(f"Adjusted BATCH_SIZE to: {BATCH_SIZE}")
    
    # Create or load dataset
    train_dataset, val_dataset = create_or_load_dataset()
    
    # Train the model
    train_with_grpo(train_dataset, val_dataset)
    
    # Generate some test problems
    test_problems = []
    for _ in range(3):
        size = (BOARD_SIZE, BOARD_SIZE)
        density = np.random.randint(2, 4) * 0.1
        game = GameOfLife(size=size)
        game.random_board(density=density)
        game.next()  # Ensure it's valid
        test_problems.append(str(game))
    
    # Test the model
    test_model(OUTPUT_DIR, test_problems)