import os.path
from transformers import EncoderDecoderModel, BertTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import DatasetDict

from pesto import ROOT_DIR

# Initialize BERT tokenizer and model
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
MODEL = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased')

# Set decoder start token id to the BERT's [CLS] token id
MODEL.config.decoder_start_token_id = TOKENIZER.cls_token_id
MODEL.config.pad_token_id = TOKENIZER.pad_token_id
MODEL.config.vocab_size = MODEL.config.decoder.vocab_size
MODEL.config.bos_token_id = TOKENIZER.bos_token_id


# Tokenize data
def preprocess_function(examples):
    inputs = TOKENIZER(examples['query'], padding="max_length", truncation=True, max_length=128)
    targets = TOKENIZER(examples['response'], padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = targets["input_ids"]
    return inputs


def train(dataset):
    dataset = dataset.map(preprocess_function, batched=True)

    # Split into train and test sets (optional)
    train_test_split = dataset.train_test_split(test_size=0.2)
    dataset = DatasetDict({
        'train': train_test_split['train'],
        'test': train_test_split['test']
    })

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(*[ROOT_DIR, "data", "results"]),
        evaluation_strategy='epoch',
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=20,
        weight_decay=0.001,
    )

    # Create data collator
    data_collator = DataCollatorForSeq2Seq(TOKENIZER, model=MODEL)

    # Define Trainer
    trainer = Trainer(
        model=MODEL,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=TOKENIZER,
        data_collator=data_collator
    )

    # Train model
    trainer.train()

    # Save model and tokenizer
    model_save_path = os.path.join(*[ROOT_DIR, "data", "results", "saved_model"])
    MODEL.save_pretrained(model_save_path)
    TOKENIZER.save_pretrained(model_save_path)

