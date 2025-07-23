"""Persona-chat model using pre-trained LLMs."""

import json
from pathlib import Path
from typing import Optional

import torch
import typer
from datasets import Dataset
from torch import nn
from tqdm import tqdm
from unsloth import (FastLanguageModel, UnslothTrainer,
                     UnslothTrainingArguments, is_bfloat16_supported)

from stimulator.utils import get_config_value


def load_pc_dataset(
    file_path: Path,
    tokenizer: Optional[nn.Module] = None,
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
) -> tuple[nn.Module, nn.Module]:
    """Load a pre-trained language model and its tokenizer."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        device_map="auto",  # Automatically map model to available devices
        token=auth_token,  # Use auth token for private models
        max_seq_length=2048,  # Set max sequence length for training
        load_in_4bit=True,  # Load model in 4-bit precision
        dtype=None,  # Use default dtype (usually float16)
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=128,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "embed_tokens",
            "lm_head",
        ],
        lora_alpha=32,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        use_rslora=True,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
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
        "unsloth/mistral-7b-v0.3-bnb-4bit",
        help="Pre-trained model name or path.",
    ),
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
    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=8,
        args=UnslothTrainingArguments(
            # === BATCHING & ACCUMULATION ===
            per_device_train_batch_size=32,  # Number of samples per device (GPU) in each forward/backward pass
            gradient_accumulation_steps=8,  # Accumulate gradients over this many steps before optimizer update
            auto_find_batch_size=True,  # Automatically reduce batch size on OOM error
            # === TRAINING LENGTH ===
            num_train_epochs=1,  # Total number of training epochs
            max_seq_length=2048,  # Maximum sequence length for tokenized input
            # === LEARNING RATE SCHEDULING ===
            learning_rate=5e-5,  # Base learning rate
            embedding_learning_rate=5e-6,  # Separate learning rate for the embedding layer
            warmup_steps=10,  # Number of steps to linearly increase LR from 0 to set value
            warmup_ratio=0.1,  # Alternatively, fraction of total steps used for warmup
            lr_scheduler_type="cosine_with_restarts",  # Use cosine annealing with restarts to adjust learning rate
            # === OPTIMIZATION ===
            weight_decay=0.05,  # Weight decay (L2 penalty)
            optim="paged_adamw_8bit",  # Optimizer type (paged memory-efficient 8-bit AdamW)
            # === PRECISION SETTINGS ===
            fp16=not is_bfloat16_supported(),  # Use 16-bit floating point (fp16) if bf16 not supported
            bf16=is_bfloat16_supported(),  # Use bfloat16 if supported (preferred on newer hardware)
            # === MEMORY & COMPUTATION SAVING ===
            packing=True,  # Pack multiple short sequences into one for better efficiency
            gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
            gradient_checkpointing_kwargs={
                "use_reentrant": False
            },  # Disable reentrant mode for compatibility
            # === LOGGING ===
            report_to="wandb",  # Use Weights & Biases for experiment tracking
            logging_steps=1,  # Log training metrics every N steps
            logging_dir=str((output_dir / "logs").resolve()),  # Directory to save logs
            # === CHECKPOINTING & OUTPUT ===
            save_strategy="epoch",  # Save model at the end of each epoch
            save_total_limit=3,  # Keep only the last 3 saved checkpoints
            output_dir=str(
                output_dir.resolve()
            ),  # Directory where model and checkpoints are saved
        ),
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
