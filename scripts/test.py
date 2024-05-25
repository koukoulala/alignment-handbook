from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "../ckpts/Mistral-7B-Instruct-v0.2/"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)  # You may want to use bfloat16 and/or move to GPU here

messages = [
    {
        "role": "system",
        "content": "",
    },
    {"role": "user", "content": "Please generate 4 Ad Headline in English language, based on the following information:\nFinalUrl: https://ripack.com/en/shrinking-gun/?mtm_campaign=Notoriete_EN&mtm_source=BingAds&mtm_medium=ppc \nDomain: ripack.com \nCategory: Retail -- Home & Garden \nLandingPage:  . Extensions for Ripack heat shrink tools | Leader for professional packaging | Ripack;  . Accueil . Heat Shrink Gun . Assistance . Distributor . Register a product . Ask for a quote . Products . Heat Shrink Guns . Ripack 3000 + . Ripack 3000 . Ripack 2500 . Ripack 2100 . Film sealing machines . Multicover 935 film dispenser . Multicover 955 . Multicover 960 . Gas bottle trolleys . Automated flam \nCharacterLimit: between 10 to 20 characters. \n"},
 ]
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
print(tokenizer.decode(tokenized_chat[0]))

outputs = model.generate(tokenized_chat, max_new_tokens=128)
print(tokenizer.decode(outputs[0]))