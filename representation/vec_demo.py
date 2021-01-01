# AutoTokenizer
from transformers import AutoTokenizer
error_log = "at org.junit.Assert.assertEquals(Assert.java:115), at org.junit.Assert.assertEquals(Assert.java:144)"
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
encoded_input = tokenizer(error_log)
print(encoded_input)
print(encoded_input['input_ids'])
print(len(encoded_input['input_ids']))

# The longer the string, the longer the vector
"""
{'input_ids': [101, 1120, 8916, 119, 179, 19782, 1204, 119, 1249, 6906, 1204, 119, 23163, 2036, 13284, 3447, 113, 1249, 6906, 1204, 119, 179, 15677, 131, 10520, 114, 117, 1120, 8916, 119, 179, 19782, 1204, 119, 1249, 6906, 1204, 119, 23163, 2036, 13284, 3447, 113, 1249, 6906, 1204, 119, 179, 15677, 131, 15373, 114, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
[101, 1120, 8916, 119, 179, 19782, 1204, 119, 1249, 6906, 1204, 119, 23163, 2036, 13284, 3447, 113, 1249, 6906, 1204, 119, 179, 15677, 131, 10520, 114, 117, 1120, 8916, 119, 179, 19782, 1204, 119, 1249, 6906, 1204, 119, 23163, 2036, 13284, 3447, 113, 1249, 6906, 1204, 119, 179, 15677, 131, 15373, 114, 102]
53
"""

# ALBERT
from transformers import AlbertTokenizer, AlbertForPreTraining
import torch
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = AlbertForPreTraining.from_pretrained('albert-base-v2')
input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
outputs = model(input_ids)
prediction_logits = outputs.prediction_logits
sop_logits = outputs.sop_logits



# BertGeneration
from transformers import BertGenerationTokenizer, BertGenerationDecoder, BertGenerationConfig
import torch
tokenizer = BertGenerationTokenizer.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder')
config = BertGenerationConfig.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")
config.is_decoder = True
model = BertGenerationDecoder.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder', config=config)

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

prediction_logits = outputs.logits



# Blenderbot
from transformers import BlenderbotSmallTokenizer, BlenderbotForConditionalGeneration
mname = 'facebook/blenderbot-90M'
model = BlenderbotForConditionalGeneration.from_pretrained(mname)
tokenizer = BlenderbotSmallTokenizer.from_pretrained(mname)
UTTERANCE = "My friends are cool but they eat too many carbs."
inputs = tokenizer([UTTERANCE], return_tensors='pt')
reply_ids = model.generate(**inputs)
print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in reply_ids])



# CTRL
from transformers import CTRLTokenizer, CTRLModel
import torch

tokenizer = CTRLTokenizer.from_pretrained('ctrl')
model = CTRLModel.from_pretrained('ctrl')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state



# DeBERTa
from transformers import DebertaTokenizer, DebertaModel
import torch

tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
model = DebertaModel.from_pretrained('microsoft/deberta-base')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state



# DistilBERT
from transformers import DistilBertTokenizer, DistilBertModel
import torch

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state


from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits



# DPR
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
model = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
input_ids = tokenizer("Hello, is my dog cute ?", return_tensors='pt')["input_ids"]
embeddings = model(input_ids).pooler_output



# ELECTRA
from transformers import ElectraTokenizer, ElectraForPreTraining
import torch

tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
model = ElectraForPreTraining.from_pretrained('google/electra-small-discriminator')

input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
logits = model(input_ids).logits



# FlauBERT
from transformers import FlaubertTokenizer, FlaubertModel
import torch

tokenizer = FlaubertTokenizer.from_pretrained('flaubert/flaubert_base_cased')
model = FlaubertModel.from_pretrained('flaubert/flaubert_base_cased')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state



# Funnel Transforme
from transformers import FunnelTokenizer, FunnelModel
import torch

tokenizer = FunnelTokenizer.from_pretrained('funnel-transformer/small')
model = FunnelModel.from_pretrained('funnel-transformer/small')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
