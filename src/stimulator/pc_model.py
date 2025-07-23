"""Persona-chat model using pre-trained LLMs."""

import json
from pathlib import Path
from typing import Optional

import torch
import typer
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from stimulator.utils import get_config_value


def load_pc_dataset(
    file_path: Path,
    tokenizer: Optional[AutoTokenizer] = None,
) -> tuple[Dataset, dict[str, int]]:
    """Load the Persona-Chat dataset from a JSONL file.

    Args:
        file_path: Path to the JSONL file containing the dataset.
        tokenizer: Optional tokenizer for processing text data.

    Returns:
        Dataset object containing the samples and a mapping of personas to IDs.
    """
    with open(file_path) as fp:
        num_lines = sum(1 for _ in fp)
        fp.seek(0)  # Reset file pointer to the beginning
        samples = [
            json.loads(line)
            for line in tqdm(fp, desc="Loading dataset", total=num_lines)
        ]

    assert samples, "Dataset is empty or not properly formatted."

    personas = sorted({s["next_message"]["persona"] for s in samples})
    persona2id = {persona: idx for idx, persona in enumerate(personas)}
    assert len(personas) > 0, "No unique personas found in the dataset."
    assert len(personas) == len(persona2id), "Persona to ID mapping is incorrect."

    examples = []
    for sample in tqdm(samples, desc="Processing dataset"):
        examples.append(
            {
                "text": "\n".join(
                    "<{}>: {} <|delay|> {}sec\n".format(
                        m["persona"],
                        m["message"],
                        round(float(m["delta_t"]), 2),
                    )
                    for m in sample["history"]
                ),
                "labels": f"<{sample['next_message']['persona']}>: {sample['next_message']['message']}",
                "target_delay": sample["next_message"]["delta_t"],
            }
        )

    ds = Dataset.from_list(examples)
    if tokenizer:
        ds = ds.map(
            lambda x: tokenizer(
                text=x["text"], text_target=x["labels"], truncation=True, padding=True
            ),  # type: ignore
            remove_columns=["text"],
            desc="Tokenizing dataset",
            batched=True,
        )

    return ds, persona2id


def load_pretrained_lm(
    model_name: str, auth_token: Optional[str] = None
) -> tuple[nn.Module, AutoTokenizer]:
    """Load a pre-trained language model and its tokenizer."""
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=auth_token, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token  # Important for batching

    # Load the pre-trained model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float32,
    )

    # Check if the flash-attn package is available
    try:
        import flash_attn  # noqa: F401

        attn_implementation = "flash_attn_2"
        typer.echo("Using flash attention for faster training.")
    except ImportError:
        attn_implementation = None
        typer.echo("Flash attention not available, using default attention.")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        token=auth_token,
        quantization_config=bnb_config,
        attn_implementation=attn_implementation,
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
            # target_modules=["o_proj", "qkv_proj", "gate_up_proj", "down_proj"],
        ),
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    typer.echo(f"Loaded model {model_name} with {num_params} trainable parameters.")
    return model, tokenizer


class PCDownstreamModel(nn.Module):
    """Downstream model for predicting metadata based on conversation history.

    Args:
        lm: Pre-trained language model to use as a base.
        input_dim: Dimension of the input features (e.g., token embeddings).
        hidden_dim: Dimension of the hidden layer in the delay predictor.
    """

    lm: nn.Module
    delay_predictor: nn.Module

    def __init__(self, lm: nn.Module, hidden_dim: int = 128) -> None:
        """Initialize the PCDownstreamModel."""
        super().__init__()
        self.lm = lm
        input_dim = getattr(
            self.lm, "lm_head"
        ).in_features  # Get hidden size from the language model
        self.delay_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Output a single value for delay prediction
            nn.Softplus(
                dim=-1
            ),  # Ensure output is non-negative (since delay is a time value)
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            input_ids: Tokenized input IDs for the conversation history.

        Returns:
            Predicted delay values for the conversation history.
        """
        with torch.no_grad():
            # Get the output from the pre-trained language model
            outputs = self.lm(input_ids=input_ids)
            # Use the last hidden state for delay prediction
            last_hidden_state = outputs.last_hidden_state

        # Extract the last token's hidden state
        h = last_hidden_state[:, -1, :]  # Shape: (batch_size, input_dim)
        # Apply the delay predictor to the last hidden state
        delay_predictions = self.delay_predictor(h).squeeze(-1)  # Shape: (batch_size,)
        return delay_predictions


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
        "distilbert/distilgpt2",
        help="Pre-trained model name or path.",
    ),
    num_train_epochs: int = typer.Option(10, help="Number of training epochs."),
    wandb: bool = typer.Option(False, help="Enable Weights & Biases logging."),
) -> None:
    """Train the model on the Persona-Chat dataset."""
    hf_api_token = get_config_value(
        "HF_API_TOKEN", ask_user=True, secret=True, allow_empty=True
    )
    model, tokenizer = load_pretrained_lm(
        model_name=model_name, auth_token=hf_api_token
    )

    # Load the dataset
    dataset, personas = load_pc_dataset(dataset_path, tokenizer=tokenizer)
    typer.echo(
        f"Loaded dataset with {len(dataset)} samples and {len(personas)} personas."
    )

    typer.echo(f"Using device: {model.device}")
    typer.echo(f"Training LLM ({model_name})...")
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
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=dataset,
    )
    trainer.train()

    # Emulate a conversation with the trained model using the first sample
    # Load first sample from dataset
    first_sample = dataset[0]["messages"]
    input_text = first_sample[0]["content"]
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    # Generate a response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_length=512,  # Limit response length
            num_return_sequences=1,
            do_sample=True,  # Enable sampling for more diverse responses
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    typer.echo(f"Generated response: {response}")
