# %%
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, EncoderDecoderModel, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
import evaluate

# %% [markdown]
# ## Text Generation
# 

# %% [markdown]
# ### GPT2
# 

# %%
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# %%
prompt = "The future of AI is "
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# %%
gen_tokens = model.generate(
    input_ids,
    max_length=20,
    num_return_sequences=1,
    do_sample=True,
    temperature=0.9,
)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
print(gen_text)

# %% [markdown]
# ### BertGeneration model (BERT based model)
# 

# %%
sentence_fuser = EncoderDecoderModel.from_pretrained("google/roberta2roberta_L-24_discofuse")
tokenizer = AutoTokenizer.from_pretrained("google/roberta2roberta_L-24_discofuse")

# %%
input_ids = tokenizer(
    prompt, add_special_tokens=False, return_tensors="pt"
).input_ids
outputs = sentence_fuser.generate(input_ids)
print(tokenizer.decode(outputs[0]))

# %% [markdown]
# ## Visualize Attention Weights using `bertviz`
# 

# %% [markdown]
# ### GPT2
# 

# %%
model = AutoModel.from_pretrained("gpt2", output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# %%
# Run the model to get the attention weights (not using generate here)
inputs = tokenizer.encode(prompt, return_tensors='pt')
outputs = model(inputs)
attention = outputs[-1]

# %%
tokens = tokenizer.convert_ids_to_tokens(inputs[0])
head_view(attention, tokens)
model_view(attention, tokens)

# %% [markdown]
# ### BERT
# 

# %%
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True)

# %%
inputs = tokenizer.encode(prompt, return_tensors='pt')
outputs = model(inputs)
attention = outputs[-1]
tokens = tokenizer.convert_ids_to_tokens(inputs[0]) 

# %%
head_view(attention, tokens)
model_view(attention, tokens)

# %% [markdown]
# ## Fine-Tune Models

# %%
dataset = load_dataset("yelp_review_full")
dataset["train"][100]

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# %% [markdown]
# ### GPT2

# %%
tokenizer = AutoTokenizer.from_pretrained("gpt2")

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# %%
tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(10))

# %%
model = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=5, torch_dtype="auto")
training_args = TrainingArguments(output_dir="test_trainer")

# %%
metric = evaluate.load("accuracy")

# %%
training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch", num_train_epochs=2)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()

# %% [markdown]
# ### BERT

# %%
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

# %%
tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(10))

# %%
model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5, torch_dtype="auto")
training_args = TrainingArguments(output_dir="test_trainer")

# %%
metric = evaluate.load("accuracy")

# %%
training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch", num_train_epochs=2)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()

# %% [markdown]
# ## Conclusion

# %% [markdown]
# GPT-2 performed better at text completion compared to the BERT model.
# 
# GPT-2 and BERT are built for different tasks:
# 
# - GPT-2 is awesome at generating text. It predicts the next word in a sequence, so it’s great for writing or continuing prompts.
# - BERT (or the pre-trained model based on BERT) is better at understanding context. It’s perfect for tasks like classification and question answering, but not really for generating text.
# 
# Based on the training aspect it was not possible to fine-tune GPT2 with my computer. It was only possible to train BERT up to 10 eposch and 50 entries of data. BERT seemed to perform well up to 7 then it proceeded to degrade. Overall with the low testing samples and low epoch the highest level of accurary was 30.


