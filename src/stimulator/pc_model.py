"""Persona-chat model using pre-trained LLMs."""

import json
from pathlib import Path

import torch
import typer
from datasets import Dataset as HFDataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from stimulator.utils import get_config_value


def load_pc_dataset(file_path: Path) -> tuple[HFDataset, list[str], dict[str, int]]:
    """Load the Persona-Chat dataset from a JSONL file.

    Args:
        file_path: Path to the JSONL file containing the dataset.

    Returns:
        Dataset object containing the samples, a list of unique personas,
        and a dictionary mapping each persona to a unique ID.
    """
    samples = []
    personas = set()
    with open(file_path) as fp:
        for line in fp:
            # Unpack the JSON object
            item = json.loads(line)
            history = item["history"]
            next_msg = item["next_message"]
            persona = item["persona"]
            # delta_t = item.get("delta_t", 0)

            # Format the history and next message
            input_text = "\n".join(f"<{p}>: {m}" for p, m in history)
            target_text = f"<{persona}>: {next_msg}"
            samples.append(
                {
                    "messages": [
                        {"role": "user", "content": input_text},
                        {"role": "assistant", "content": target_text},
                    ],
                    # "persona": persona,
                    # "persona_id": None,  # We'll populate this in a secondary pass through the dataset
                    # "delta_t": delta_t,
                }
            )
            personas.add(persona)
    personas = list(sorted(personas))  # Sort personas for consistency
    persona2id = {persona: idx for idx, persona in enumerate(personas)}

    # Convert persona strings to IDs
    # for sample in samples:
    #     sample["persona_id"] = persona2id[sample["persona"]]
    #     del sample["persona"]  # Remove original persona string

    return HFDataset.from_list(samples), personas, persona2id


app = typer.Typer(help="PC Model CLI")


@app.command()
def train(
    dataset_path: Path = typer.Argument(
        help="Path to the Persona-Chat dataset JSONL file."
    ),
    output_dir: Path = typer.Option(
        "./outputs",
        help="Directory to save the trained model and tokenizer.",
    ),
    model_name: str = typer.Option(
        "microsoft/Phi-3-mini-4k-instruct", help="Pre-trained model name or path."
    ),
    max_length: int = typer.Option(
        2048, help="Maximum sequence length for input text."
    ),
    num_train_epochs: int = typer.Option(10, help="Number of training epochs."),
    wandb: bool = typer.Option(False, help="Enable Weights & Biases logging."),
) -> None:
    """Train the model on the Persona-Chat dataset."""
    hf_api_token = get_config_value(
        "HF_API_TOKEN", ask_user=True, secret=True, allow_empty=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=hf_api_token, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token  # Important for batching

    # Load the dataset
    dataset, personas, _ = load_pc_dataset(dataset_path)

    # Load the pre-trained model
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float32,
        )
    else:
        bnb_config = None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        token=hf_api_token,
    )
    # Apply 4-bit quantization and low-rank adaptation
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            # the rank of the adapter, the lower the fewer parameters you'll need to train
            r=8,
            lora_alpha=16,  # multiplier, usually 2*r
            bias="none",
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            # Newer models, such as Phi-3 at time of writing, may require
            # manually setting target modules
            target_modules=["o_proj", "qkv_proj", "gate_up_proj", "down_proj"],
        ),
    )
    typer.echo(f"Training on {len(dataset)} samples with {len(personas)} personas.")
    typer.echo(f"Using device: {model.device}")

    sft_config = SFTConfig(
        # GROUP 1: Memory usage
        # These arguments will squeeze the most out of your GPU's RAM
        # Checkpointing
        gradient_checkpointing=True,  # this saves a LOT of memory
        # Set this to avoid exceptions in newer versions of PyTorch
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Gradient Accumulation / Batch size
        # Actual batch (for updating) is same (1x) as micro-batch size
        gradient_accumulation_steps=1,
        # The initial (micro) batch size to start off with
        per_device_train_batch_size=16,
        # If batch size would cause OOM, halves its size until it works
        auto_find_batch_size=True,
        # GROUP 2: Dataset-related
        max_seq_length=max_length,
        # Dataset
        # packing a dataset means no padding is needed
        packing=True,
        # GROUP 3: These are typical training parameters
        num_train_epochs=num_train_epochs,
        learning_rate=3e-4,
        # Optimizer
        # 8-bit Adam optimizer - doesn't help much if you're using LoRA!
        optim="paged_adamw_8bit",
        # GROUP 4: Logging parameters
        logging_steps=10,
        logging_dir=str((output_dir / "logs").resolve()),
        output_dir=str(output_dir.resolve()),
        report_to="wandb" if wandb else "none",
        run_name=f"{model_name}-{dataset_path.stem}-sft",
        # GROUP 5: Other parameters
        save_strategy="epoch",  # Save model at the end of each epoch
        save_total_limit=3,  # Keep only the last 3 checkpoints
    )
    trainer = SFTTrainer(
        model=model, args=sft_config, train_dataset=dataset, processing_class=tokenizer
    )
    trainer.train()

    # Emulate a conversation with the trained model using the first sample
    # Load first sample from dataset
    sample = dataset[0]
    input_ids = sample["input_ids"].unsqueeze(0).to(model.device)  # [1, seq_len]

    # Set model to eval mode
    model.eval()
    with torch.no_grad():
        # Generate next response
        generated_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=512,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
        )
        # Decode generated response
        generated_text = tokenizer.decode(
            generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True
        )

    # Print the emulated conversation
    print("=== Conversation History ===")
    print(tokenizer.decode(input_ids[0], skip_special_tokens=True))
    print("\n=== Model Response ===")
    print(f"{generated_text.strip()}")
