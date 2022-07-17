from parrot import Parrot
from transformers import pipeline, GPTNeoForCausalLM, GPT2Tokenizer
import torch
import warnings
from flask import Flask

app = Flask(__name__)

@app.route("/")
class Generator():
    def test_generate(self, input_str: str):
        tokenizerfile = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
        tokenizerfile.pad_token = tokenizerfile.eos_token
        gradient_ckpt = True
        model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M", pad_token_id=tokenizerfile.eos_token_id, gradient_checkpointing=gradient_ckpt, use_cache=not gradient_ckpt)
        input_ids = tokenizerfile.encode(input_str, add_special_tokens=False, return_tensors="pt")
        attention_mask = torch.where(input_ids == tokenizerfile.eos_token_id, torch.zeros_like(input_ids), torch.ones_like(input_ids)).to(model.device)
        output_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=750, num_return_sequences=1)
        output_str = tokenizerfile.decode(output_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return(output_str)


parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=False)
phrases = []
phrases.append(input("insert phrase to search: "))
put_text(phrases)
for phrase in phrases:
    para_phrases = []
    para_phrases = parrot.augment(input_phrase=phrase, use_gpu=False)
    for p in para_phrases:
        selected = p[0]

sa = Generator().test_generate(selected)
print(sa)

if __name__ == "__main__":
    app.run()
