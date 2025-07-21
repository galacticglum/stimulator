"""Persona-chat model using pre-trained LLMs."""

import json
from collections import namedtuple
from pathlib import Path
from typing import Optional, Union

import torch
import typer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from stimulator.utils import get_config_value, get_device

PersonaChatExample = namedtuple(
    "PersonaChatExample",
    ["input_text", "target_text", "persona", "delta_t"],
)


class PersonaChatDataset(Dataset):
    """Dataset for the Persona-Chat model.

    Args:
        file_path: Path to the JSONL file containing the dataset. Each line
            should be a JSON object with keys: "history", "next_message", "persona", and "delta_t".
        tokenizer: Tokenizer to convert text to input IDs.
        max_length: Maximum length of the input sequences.

    Instance Attributes:
        samples: List of tuples containing input text, target text, and persona.
        personas: List of unique personas in the dataset.
        persona2id: Dictionary mapping each persona to a unique ID.
    """

    samples: list[PersonaChatExample]
    personas: list[str]
    persona2id: dict[str, int]

    # Private Instance Attributes
    _tokenizer: PreTrainedTokenizer
    _max_length: int

    def __init__(
        self, file_path: Path, tokenizer: PreTrainedTokenizer, max_length: int = 2048
    ) -> None:
        """Initialize the PersonaChatDataset."""
        self._tokenizer = tokenizer
        self._max_length = max_length

        self.samples = []
        with open(file_path) as fp:
            for line in fp:
                # Unpack the JSON object
                item = json.loads(line)
                history = item["history"]
                next_msg = item["next_message"]
                persona = item["persona"]
                delta_t = item.get("delta_t", 0)

                # Format the history and next message
                input_text = "\n".join(f"<{p}>: {m}" for p, m in history)
                target_text = f"<{persona}>: {next_msg}"
                self.samples.append(
                    PersonaChatExample(input_text, target_text, persona, float(delta_t))
                )

        self.personas = sorted({p for _, _, p, _ in self.samples})
        self.persona2id = {p: i for i, p in enumerate(self.personas)}

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, int]:
        """Return a sample from the dataset as a dictionary.

        Args:
            idx: Index of the sample to retrieve.

        Raises:
            IndexError: If the index is out of bounds.

        Returns:
            A dictionary containing:
                - "input_ids": Token IDs for the input text.
                - "labels": Token IDs for the target text.
                - "persona_id": ID of the persona associated with the sample.
                - "delta_t": Time delta for the response.
        """
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of size {len(self.samples)}."
            )

        # Retrieve the sample
        input_text, target_text, persona, delta_t = self.samples[idx]
        input_ids = self._tokenizer(
            input_text,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        ).input_ids.squeeze()

        labels = self._tokenizer(
            target_text,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        ).input_ids.squeeze()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "persona_id": self.persona2id[persona],
            "delta_t": delta_t,
        }


class PersonaDataCollator:
    """Data collator for the PersonaChatDataset."""

    padding_value: float

    def __init__(
        self, tokenizer: AutoTokenizer, padding_value: Optional[float] = None
    ) -> None:
        """Initialize the data collator."""
        self.tokenizer = tokenizer
        if padding_value is None:
            self.padding_value = tokenizer.pad_token_id  # type: ignore
        else:
            self.padding_value = padding_value

    def __call__(
        self, features: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """Collate a batch of features into padded tensors."""
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        persona_ids = torch.tensor([f["persona_id"] for f in features])
        delta_ts = torch.tensor([f["delta_t"] for f in features], dtype=torch.float32)
        input_ids_padded = pad_sequence(
            input_ids, batch_first=True, padding_value=self.padding_value
        )
        labels_padded = pad_sequence(
            labels, batch_first=True, padding_value=-100
        )  # For loss masking

        return {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "persona_id": persona_ids,
            "delta_t": delta_ts,
        }


class PersonaChatModel(nn.Module):
    """Generic causal LLM for persona-aware dialogue modeling.

    This model combines a pre-trained causal LM with a linear classifier
    to predict the persona ID based on the last hidden state of the LLM.

    Args:
        model_name: Name or path of the pre-trained model.
        num_personas: Number of unique personas in the dataset.
        auth_token: Optional Hugging Face authentication token for private models.

    Shape:
        - input_ids: Tensor of shape (batch_size, sequence_length) containing token IDs.
        - labels: Tensor of shape (batch_size, sequence_length) containing target token IDs.
        - persona_id: Tensor of shape (batch_size,) containing persona IDs.

    Returns:
        A dictionary containing the following keys:
            - "loss": Combined loss (only if all targets are provided).
            - "lm_loss": Language modeling loss (0.0 if labels not provided).
            - "persona_loss": Persona classification loss (0.0 if persona_id not provided).
            - "time_loss": Delta time regression loss (0.0 if delta_t not provided).
            - "persona_logits": Raw logits from persona classification head.
            - "delta_t_pred": Predicted delta times from regression head.

    Instance Attributes:
        lm: The pre-trained model.
        classifier: A linear layer that maps the last hidden state to persona IDs.
    """

    lm: nn.Module
    classifier: nn.Linear

    def __init__(
        self,
        model_name: str,
        num_personas: int,
        auth_token: Optional[str] = None,
    ) -> None:
        """Initialize the PersonaChatModel."""
        super().__init__()
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
            device_map=get_device(),
            quantization_config=bnb_config,
            token=auth_token,
        )
        hidden_size = model.config.hidden_size
        # Apply 4-bit quantization and low-rank adaptation
        model = prepare_model_for_kbit_training(model)
        self.lm = get_peft_model(
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

        self.persona_head = nn.Linear(hidden_size, num_personas)
        self.delta_time_head = nn.Linear(hidden_size, 1)
        self.config = (
            self.lm.base_model.config
        )  # Wrapper so that the trainer can access the config

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        persona_id: Optional[torch.Tensor] = None,
        delta_t: Optional[torch.Tensor] = None,
        alpha: float = 0.5,
        beta: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for training or inference."""
        # If labels are provided, this will compute the causal LM loss
        outputs = self.lm(
            input_ids=input_ids, labels=labels if labels is not None else None
        )
        hidden_state = outputs.last_hidden_state  # [B, T, H]
        cls_hidden = hidden_state[:, 0, :]  # Use [CLS]-like first token representation

        result = {}

        if labels is not None:
            lm_loss = outputs.loss
            result["lm_loss"] = lm_loss
        else:
            result["lm_loss"] = torch.tensor(0.0, device=input_ids.device)

        # Persona classification
        if persona_id is not None:
            persona_logits = self.persona_head(cls_hidden)
            persona_loss = nn.CrossEntropyLoss()(persona_logits, persona_id)
            result["persona_logits"] = persona_logits
            result["persona_loss"] = persona_loss
        else:
            result["persona_logits"] = self.persona_head(cls_hidden)
            result["persona_loss"] = torch.tensor(0.0, device=input_ids.device)

        # Delta time regression
        if delta_t is not None:
            time_pred = self.delta_time_head(cls_hidden).squeeze(1)
            time_loss = nn.MSELoss()(time_pred, delta_t)
            result["delta_t_pred"] = time_pred
            result["time_loss"] = time_loss
        else:
            result["delta_t_pred"] = self.delta_time_head(cls_hidden).squeeze(1)
            result["time_loss"] = torch.tensor(0.0, device=input_ids.device)

        # Combine loss if training
        if labels is not None and persona_id is not None and delta_t is not None:
            total_loss = (
                result["lm_loss"]
                + alpha * result["persona_loss"]  # noqa: W503
                + beta * result["time_loss"]  # noqa: W503
            )
        else:
            total_loss = torch.tensor(0.0, device=input_ids.device)

        result["loss"] = total_loss

        return result


class PersonaChatSFTTrainer(SFTTrainer):
    """Trainer for fine-tuning the PersonaChatModel.

    Acts as a wrapper around the SFTTrainer to handle the multi-task loss.
    """

    def compute_loss(
        self,
        model: PersonaChatModel,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        """Compute the loss for the model."""
        outputs = model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
            persona_id=inputs["persona_id"],
            delta_t=inputs["delta_t"],
        )
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss


app = typer.Typer(help="Discord Scraper CLI")


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
) -> None:
    """Train the PersonaChatModel on the Persona-Chat dataset."""
    hf_api_token = get_config_value(
        "HF_API_TOKEN", ask_user=True, secret=True, allow_empty=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=hf_api_token, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Important for batching

    dataset = PersonaChatDataset(dataset_path, tokenizer, max_length)
    model = PersonaChatModel(
        model_name, num_personas=len(dataset.persona2id), auth_token=hf_api_token
    )

    typer.echo(
        f"Training on {len(dataset)} samples with {len(dataset.personas)} personas."
    )
    typer.echo(f"Using device: {model.lm.device}")

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
        report_to="none",
    )

    trainer = PersonaChatSFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=PersonaDataCollator(tokenizer),
    )

    trainer.train()
    trainer.save_model(model_name + "-" + dataset_path.stem + "-trained")

    # Emulate a conversation with the trained model using the first sample
    # Load first sample from dataset
    sample = dataset[0]
    input_ids = sample["input_ids"].unsqueeze(0).to(model.lm.device)  # [1, seq_len]

    # Set model to eval mode
    model.eval()
    with torch.no_grad():
        # Generate next response
        generated_ids = model.lm.generate(
            input_ids=input_ids,
            max_new_tokens=512,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
        )
        # Decode generated response
        generated_text = tokenizer.decode(
            generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True
        )

        # Forward pass to get persona & delta_t predictions
        outputs = model(input_ids=input_ids)
        persona_logits = outputs["persona_logits"]
        predicted_persona_id = torch.argmax(persona_logits, dim=1).item()
        predicted_persona = dataset.personas[predicted_persona_id]
        predicted_delta_t = outputs["delta_t_pred"].item()

    # Print the emulated conversation
    print("=== Conversation History ===")
    print(tokenizer.decode(input_ids[0], skip_special_tokens=True))
    print("\n=== Model Response ===")
    print(f"<{predicted_persona}>: {generated_text.strip()}")
    print(f"(Predicted response delay: {predicted_delta_t:.2f} seconds)")
