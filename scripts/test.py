import torch
from peft import PeftModel, PeftTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# specify how to quantize the model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
)

checkpoint = "./ckpts/Mistral-7B-Instruct-v0.2/"
model = AutoModelForCausalLM.from_pretrained(checkpoint, quantization_config=quantization_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

lora_ckpt = "./output/asset-generation-sft-qlora/checkpoint-4000/"
model = PeftModel.from_pretrained(model, lora_ckpt, adapter_name="asset_generation")
model.set_adapter("asset_generation")
messages = [
    {"role": "user",
     "content": "Please generate 4 Ad Headline in English language, based on the following information:\nFinalUrl: https://ripack.com/en/shrinking-gun/?mtm_campaign=Notoriete_EN&mtm_source=BingAds&mtm_medium=ppc \nDomain: ripack.com \nCategory: Retail -- Home & Garden \nLandingPage:  . Extensions for Ripack heat shrink tools | Leader for professional packaging | Ripack;  . Accueil . Heat Shrink Gun . Assistance . Distributor . Register a product . Ask for a quote . Products . Heat Shrink Guns . Ripack 3000 + . Ripack 3000 . Ripack 2500 . Ripack 2100 . Film sealing machines . Multicover 935 film dispenser . Multicover 955 . Multicover 960 . Gas bottle trolleys . Automated flam \nCharacterLimit: between 10 to 20 characters. \n"},
]
'''
messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant",
     "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]
'''
model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
print(tokenizer.batch_decode(generated_ids)[0])



