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

@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-1.5B-Instruct")
    block_size: int = field(default=2048)
    wandb_project: Optional[str] = field(default="S1-Qwen2.5-1.5B-Instruct")
    wandb_entity: Optional[str] = field(default="openmodels")
    use_wandb: bool = field(default=True)
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
                self.wandb_entity = os.environ.get('WANDB_ENTITY', 'openmodels')
            os.environ['WANDB_PROJECT'] = self.wandb_project
            os.environ['WANDB_ENTITY'] = self.wandb_entity
        else:
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

    # Configure tokenizer with strict length control
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_name,
        use_fast=True,
        model_max_length=config.block_size,
        padding_side="right",
        truncation_side="right",
    )
    
    if "Llama" in config.model_name:
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif "Qwen" in config.model_name:
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
        tokenizer.pad_token = "<|fim_pad|>"

    # Ensure tokenizer configuration
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Process dataset with fixed length
    def preprocess_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=config.block_size,
            padding="max_length",
            return_tensors=None,
        )

    # Process dataset
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing dataset",
    )

    # Update data collator settings
    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
    )

    # Set dataset configuration
    args.dataset_text_field = 'text'
    args.max_seq_length = config.block_size

    # Create trainer with tokenizer
    trainer = trl.SFTTrainer(
        model,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'] if 'test' in tokenized_dataset else tokenized_dataset['train'],
        args=args,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
