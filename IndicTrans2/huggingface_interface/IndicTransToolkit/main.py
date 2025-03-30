import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoTokenizer
from IndicTransToolkit import IndicProcessor

# Constants
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
quantization = None

def initialize_model_and_tokenizer(ckpt_dir, quantization):
    """Initialize the model and tokenizer with optional quantization."""
    if quantization == "4-bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8-bit":
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        qconfig = None

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=qconfig,
    )

    if qconfig is None:
        model = model.to(DEVICE)
        if DEVICE == "cuda":
            model.half()

    model.eval()
    return tokenizer, model


def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    """Batch translate sentences from src_lang to tgt_lang."""
    translations = []
    
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i : i + BATCH_SIZE]

        # Preprocess the batch and extract entity mappings
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

        # Tokenize the batch and generate input encodings
        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)

        # Generate translations using the model
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        # Decode the generated tokens into text
        with tokenizer.as_target_tokenizer():
            generated_tokens = tokenizer.batch_decode(
                generated_tokens.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

        # Postprocess the translations, including entity replacement
        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)

        del inputs
        torch.cuda.empty_cache()

    return translations


# Initialize the model and processor
indic_en_ckpt_dir = "ai4bharat/indictrans2-indic-en-1B"  
indic_en_tokenizer, indic_en_model = initialize_model_and_tokenizer(indic_en_ckpt_dir, quantization)

ip = IndicProcessor(inference=True)

# Sample sentences
ml_sents = [
    """ഹലോ ഫ്രണ്ട്സ് കോളേജ് സ്കൂളിൻറെ മറ്റൊരു അധ്യായത്തിലേക്ക് ഏവർക്കും സ്വാഗതം ഇന്ന് ഞാൻ വന്നിരിക്കുന്നത് ചെറിയ കുട്ടികൾക്കായുള്ള ഒരു മലയാളം പ്രസംഗവും ആയിട്ടാണ് പ്രസംഗ വിഷയം ഇന്ത്യ എൻറെ രാജ്യം ആയിരക്കണക്കിന് വർഷങ്ങളുടെ പാരമ്പര്യം പേറുന്ന മഹത്തായ രാജ്യമാണ് ഇന്ത്യ 1947 ൽ ബ്രിട്ടീഷുകാരിൽ നിന്നും സ്വാതന്ത്ര്യം നേടിയ നമ്മുടെ ഭാരതം അനേകം നാട്ടുരാജ്യങ്ങൾ ചേർന്ന് ഏറ്റവും വലിയ ജനാധിപത്യ രാജ്യമായി ആശയുടെ അടിസ്ഥാനത്തിൽ നല്ല ഭരണത്തിന് സഹായകമാകും വിധം സംസ്ഥാനങ്ങൾ രൂപം കൊണ്ടും എന്ന് 28 സംസ്ഥാനങ്ങൾ ആണ് ഇന്ത്യയിൽ ഉള്ളത് നാനാത്വത്തിലെ ഏകത്വം എന്ന ചിന്ത വിവിധ ഭാഷകളും ജാതികളും മതങ്ങളും ആചാരങ്ങളും ജീവിതരീതികളും ഉള്ള ഒരു വലിയ ജനതയെ ഒറ്റക്കെട്ടായി നിർത്തുന്നു അതാണ് ഭാരതത്തിൻറെ വിജയം നേടിയ ലോകമേ തറവാട് എന്നതാണ് ഭാരത സംസ്കാരം അതുകൊണ്ട് തന്നെ ഇന്ത്യക്കാരെ മാത്രമല്ല ലോകം മുഴുവനും ഉള്ള എല്ലാവരെയും ഭാരതം സന്തോഷത്തോടെ ഉൾക്കൊള്ളുകയും സ്നേഹിക്കുകയും ചെയ്യുന്ന പ്രസിഡണ്ടും പ്രധാനമന്ത്രിയും മന്ത്രിമാരും ചേർന്ന് നമ്മുടെ രാജ്യം ഭരിക്കുന്നു മുഖ്യമന്ത്രിയും മന്ത്രിമാരും ചേർന്ന് സംസ്ഥാനങ്ങളെയും പരിപാലിക്കുന്നു എൻറെ ഇന്ത്യ അഭിമാനമാണ് സംസ്കാരങ്ങൾ ചേർന്ന് മനോഹരിയായി പുഞ്ചിരിക്കുന്ന എൻറെ അമ്മ ഭാരതമെന്നു കേട്ടാൽ തിളക്കണം ചോര നമുക്ക് ഞരമ്പുകളിൽ"""
]

# Translation
src_lang, tgt_lang = "mal_Mlym", "eng_Latn"
en_translations = batch_translate(ml_sents, src_lang, tgt_lang, indic_en_model, indic_en_tokenizer, ip)

# Print translations
print(f"\n{src_lang} - {tgt_lang}")
for input_sentence, translation in zip(ml_sents, en_translations):
    print(f"{src_lang}: {input_sentence}")
    print(f"{tgt_lang}: {translation}")

# flush the models to free the GPU memory
del indic_en_tokenizer, indic_en_model
