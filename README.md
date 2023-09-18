# Roberta Zinc Decoder

This model is a GPT2 decoder model designed to reconstruct SMILES strings from embeddings created by the 
[roberta_zinc_480m](https://huggingface.co/entropy/roberta_zinc_480m) model.

The decoder model conditions generation on mean pooled embeddings from the encoder model. Mean pooled 
embeddings are used to allow for integration with vector databases, which require fixed length embeddings.

Condition embeddings are passed to the decoder model using the `encoder_hidden_states` attribute. 
The standard `GPT2LMHeadModel` does not support generation with encoder hidden states, so this repo 
includes a custom `ConditionalGPT2LMHeadModel`. See example below for how to instantiate the model.

## How to use
To use, install the [transformers](https://github.com/huggingface/transformers) library:

```
pip install transformers
```

Then use the following

```python
import torch
from transformers import AutoModelForCausalLM, RobertaTokenizerFast, RobertaForMaskedLM, DataCollatorWithPadding

tokenizer = RobertaTokenizerFast.from_pretrained("entropy/roberta_zinc_480m", max_len=256)
collator = DataCollatorWithPadding(tokenizer, padding=True, return_tensors='pt')

encoder_model = RobertaForMaskedLM.from_pretrained('entropy/roberta_zinc_480m')
encoder_model.eval();

commit_hash = '0ba58478f467056fe33003d7d91644ecede695a7'
decoder_model = AutoModelForCausalLM.from_pretrained("entropy/roberta_zinc_decoder",
                                                     trust_remote_code=True, revision=commit_hash)
decoder_model.eval();


smiles = ['Brc1cc2c(NCc3ccccc3)ncnc2s1',
 'Brc1cc2c(NCc3ccccn3)ncnc2s1',
 'Brc1cc2c(NCc3cccs3)ncnc2s1',
 'Brc1cc2c(NCc3ccncc3)ncnc2s1',
 'Brc1cc2c(Nc3ccccc3)ncnc2s1']

inputs = collator(tokenizer(smiles))
outputs = encoder_model(**inputs, output_hidden_states=True)
full_embeddings = outputs[1][-1]
mask = inputs['attention_mask']
mean_embeddings = ((full_embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1))

decoder_inputs = torch.tensor([[tokenizer.bos_token_id] for i in range(len(smiles))])

hidden_states = mean_embeddings[:,None] # hidden states shape (bs, 1, -1)

gen = decoder_model.generate(
              decoder_inputs,
              encoder_hidden_states=hidden_states,
              do_sample=False, # greedy decoding is recommended
              max_length=100, 
              temperature=1.,
              early_stopping=True,
              pad_token_id=tokenizer.pad_token_id,
                         )

reconstructed_smiles = tokenizer.batch_decode(gen, skip_special_tokens=True)
```
