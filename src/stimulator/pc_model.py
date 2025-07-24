"""Persona-chat model using pre-trained LLMs."""

import json
from pathlib import Path
from typing import Any, Optional

import typer
import unsloth  # noqa: F401  # unsloth must be imported first to apply its patches
from datasets import Dataset
from torch import nn
from tqdm import tqdm
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, get_chat_template, is_bfloat16_supported

from stimulator.utils import get_config_value


def load_pc_dataset(file_path: Path) -> tuple[Dataset, dict[str, int]]:
    """Load the Persona-Chat dataset from a JSONL file.

    Args:
        file_path: Path to the JSONL file containing the dataset.

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
                "conversations": [
                    {
                        "role": "system",
                        "content": f"Your persona is: {sample['next_message']['persona']}",
                    },
                    {
                        "role": "user",
                        "content": "\n".join(
                            "<{}>: {} <|delay|> {}sec\n".format(
                                m["persona"],
                                m["message"],
                                round(float(m["delta_t"]), 2),
                            )
                            for m in sample["history"]
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": f"<{sample['next_message']['persona']}>: {sample['next_message']['message']}",
                    },
                ]
            }
        )

    ds = Dataset.from_list(examples)
    return ds, persona2id


def load_pretrained_lm(
    model_name: str,
    auth_token: Optional[str] = None,
    max_seq_length: Optional[int] = None,
    seed: Optional[int] = None,
) -> tuple[nn.Module, nn.Module]:
    """Load a pre-trained language model and its tokenizer."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        token=auth_token,  # Use auth token for private models
        max_seq_length=max_seq_length,  # Set max sequence length for training
        load_in_4bit=True,  # Load model in 4-bit precision
        dtype=None,  # Use default dtype (usually float16)
        attn_implementation="flash_attention_2",  # Use Flash Attention 2 for efficiency
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=seed,  # Set random seed for reproducibility
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    typer.echo(f"Loaded model {model_name} with {num_params} trainable parameters.")
    typer.echo(f"Using device: {model.device}")
    return model, tokenizer


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
        "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        help="Pre-trained model name or path.",
    ),
    max_seq_length: int = typer.Option(
        2048,
        help="Maximum sequence length for training.",
    ),
    num_train_epochs: int = typer.Option(
        1,
        help="Number of training epochs. If `num_steps` is set, this will be ignored.",
    ),
    num_steps: Optional[int] = typer.Option(
        None,
        help="Total number of training steps. If set, `num_train_epochs` will be ignored.",
    ),
    num_proc: int = typer.Option(
        8,
        help="Number of processes to use for dataset processing.",
    ),
    seed: int = typer.Option(42, help="Random seed for reproducibility."),
) -> None:
    """Train the model on the Persona-Chat dataset."""
    hf_api_token = get_config_value(
        "HF_API_TOKEN", ask_user=True, secret=True, allow_empty=True
    )
    model, tokenizer = load_pretrained_lm(
        model_name=model_name, auth_token=hf_api_token, seed=seed
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id

    # Load the dataset
    dataset, personas = load_pc_dataset(dataset_path)
    typer.echo(
        f"Loaded dataset with {len(dataset)} samples and {len(personas)} personas."
    )
    typer.echo(dataset)
    typer.echo(f"Personas: {', '.join(personas.keys())}")

    # Dataset is in HuggingFace's generic conversational format
    # Now we map it to the chat template expected by the model
    def formatting_prompts_func(examples: dict[str, Any]) -> dict[str, Any]:
        """Format the dataset examples to match the model's chat template."""
        # Each example is a dictionary with a "conversations" key
        # containing a list of messages in the chat format
        assert "conversations" in examples, "Expected 'conversations' key in examples."
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {
            "text": texts,
        }

    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        desc="Formatting dataset for training",
        num_proc=num_proc,
    )
    typer.echo("Dataset has been formatted for training. Here's a sample:")
    typer.echo(dataset[0])

    typer.echo(f"Training LLM ({model_name})...")
    trainer_args = SFTConfig(
        # === DATASET ===
        dataset_text_field="text",
        dataset_num_proc=num_proc,
        max_length=max_seq_length,
        packing=False,  # Can make training 5x faster for short sequences
        # === BATCHING & ACCUMULATION ===
        per_device_train_batch_size=2,  # Number of samples per device (GPU) in each forward/backward pass
        gradient_accumulation_steps=4,  # Accumulate gradients over this many steps before optimizer update
        # Effective batch size = per_device_train_batch_size * gradient_accumulation_steps
        auto_find_batch_size=True,  # Automatically reduce batch size on OOM error
        # === TRAINING LENGTH ===
        num_train_epochs=num_train_epochs,  # Total number of training epochs
        max_steps=num_steps if num_steps else -1,  # Total number of training steps
        # === LEARNING RATE SCHEDULING ===
        learning_rate=2e-4,  # Base learning rate
        warmup_steps=10,  # Number of steps to linearly increase LR from 0 to set value
        warmup_ratio=0.1,  # Alternatively, fraction of total steps used for warmup
        lr_scheduler_type="linear",
        # === OPTIMIZATION ===
        weight_decay=0.01,  # Weight decay (L2 penalty)
        optim="adamw_8bit",  # Use 8-bit AdamW optimizer for memory efficiency
        # === PRECISION SETTINGS ===
        fp16=not is_bfloat16_supported(),  # Use 16-bit floating point (fp16) if bf16 not supported
        bf16=is_bfloat16_supported(),  # Use bfloat16 if supported (preferred on newer hardware)
        # === MEMORY & COMPUTATION SAVING ===
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
        # === MISC ===
        seed=seed,  # Set random seed for reproducibility
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,  # type: ignore
        train_dataset=dataset,
        args=trainer_args,
    )
    trainer.train()

    # Emulate a conversation with the trained model using the first sample
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
    messages = dataset[0]["conversations"]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,  # Must add for generation
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        input_ids=inputs, max_new_tokens=64, use_cache=True, temperature=1.5, min_p=0.1
    )
    response = tokenizer.batch_decode(outputs)
    typer.echo("Sample conversation with the trained model:")
    typer.echo(f"Input: {json.dumps(messages, indent=2)}")
    typer.echo(f"Response: {response[0]}")

    # Save the trained model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    typer.echo(f"Model and tokenizer saved to {output_dir.resolve()}")
