import os
from dataclasses import dataclass, field, asdict
from typing import Optional
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import load_dataset, concatenate_datasets, DatasetDict
import transformers
import trl
from transformers import TrainingArguments

@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-1.5B-Instruct")
    block_size: int = field(default=32768)
    wandb_project: Optional[str] = field(default="s1-agent")
    wandb_entity: Optional[str] = field(default=None)
    use_wandb: bool = field(default=False)
    train_file_path: Optional[str] = field(default='simplescaling/s1K_tokenized')
    dagger: bool = field(default=False)
    custom_fsdp_config: dict = field(
        default_factory=lambda: {
            "transformer_layer_cls_to_wrap": ["Qwen2DecoderLayer"],
            "min_num_params": 0,
            "xla": False,
            "xla_fsdp_v2": False,
            "xla_fsdp_grad_ckpt": False,
            "activation_checkpointing": False,
            "limit_all_gathers": True,
        }
    )

    def __post_init__(self):
        if self.use_wandb:
            if not self.wandb_entity:
                # Try to get from environment variable
                self.wandb_entity = os.environ.get('WANDB_ENTITY')
            if self.wandb_project:
                os.environ['WANDB_PROJECT'] = self.wandb_project
            if self.wandb_entity:
                os.environ['WANDB_ENTITY'] = self.wandb_entity
        else:
            # Disable wandb
            os.environ['WANDB_DISABLED'] = 'true'

def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    
    # Disable gradient checkpointing since we're using FSDP activation checkpointing
    args.gradient_checkpointing = False
    
    # Update args.fsdp_config with our custom config
    if hasattr(args, 'fsdp_config'):
        args.fsdp_config.update(config.custom_fsdp_config)
    
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    # loading model
    kwargs = {}
    if "70B" in config.model_name:
        # Removed "low_cpu_mem_usage": True, for 70B, since by default we are in FSDP,
        # it's more efficient to do  "cpu_ram_efficient_loading": true, in fsdp_config.json
        kwargs = {"device_map": "auto", "torch_dtype": "auto",
                  "attn_implementation": "flash_attention_2", "use_cache": False}
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name)

    dataset = load_dataset(config.train_file_path)

    # setting up trainer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if "Llama" in config.model_name:
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif "Qwen" in config.model_name:
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|fim_pad|>"

    # Add padding configuration
    args.padding = True  # Enable padding
    args.pad_to_multiple_of = 8  # Ensure consistent sequence lengths
    
    # Ensure max_length is set and consistent
    args.max_length = config.block_size
    args.truncation = True  # Enable truncation
    
    # Update data collator settings
    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
        padding=True,
        max_length=config.block_size,
        pad_to_multiple_of=8
    )
    args.dataset_text_field = 'text'
    args.max_seq_length = config.block_size
    trainer = trl.SFTTrainer(
        model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'],
        args=args,
        data_collator=collator
    )

    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
