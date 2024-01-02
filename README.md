# Tích hợp hệ thống thông tin

## Enviroment

Colab google, Kaggle + ngrok
> Our team utilizes the GPU resources on Kaggle for llama2 and the GPU on Google Colab for Falcon. Therefore, we also need to create two Ngrok accounts to generate two APIs.

## Llama2-7b using Kaggle(GPU P100) + ngrok

Install all dependencies using [pip](https://pip.pypa.io/en/stable/installation/)

```
!pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7
!pip install transformers torch
!pip install googletrans==4.0.0rc1 diffusers transformers accelerate xformers pyngrok
```

Load fine-tuning model (Hugging-face)
```
device_map = {"": 0}
model_name='Dan2205/Llama-2-7b-chat-finetune'
model=AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
```
```
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=256)
```
Config flask ngrok
```
os.environ["FLASK_ENV"] = "development"
app = Flask(__name__)
port = 5000

app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

ngrok.set_auth_token("2aOzkYCgk7Iz61PVhVcZIGWwwLG_6EQ8cn9Ln5zoJfsroi9Cr")
public_url = ngrok.connect(port).public_url
print(" * ngrok \"{}\" - \"http://127.0.0.1:{}\"".format(public_url,port))

app.config["BASE_URL"] = public_url
```
integrate model
```
@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        content = request.json
        prompt = content['prompt']

        result = pipe(prompt)
        list_result = []
        for tmp in result:
            utf8_str = tmp['generated_text'].encode('utf-8')

            # Mã hóa Base64
            base64_encoded = base64.b64encode(utf8_str)

            # Chuyển bytes thành chuỗi
            base64_encoded_text = base64_encoded.decode("utf-8")
            
            list_result.append(base64_encoded_text)
        return jsonify(list_result)

    except Exception as e:
        return jsonify({'error': str(e)})

threading.Thread(target = app.run, kwargs={"use_reloader": False}).start()
```


## Falcon-7b using Colab google + ngrok

Install all dependencies using [pip](https://pip.pypa.io/en/stable/installation/)

```
!pip install transformers torch accelerate tensorflow_probability==2.14 safetensors pyngrok googletrans==4.0.0rc1
!pip -qqq install bitsandbytes accelerate
```

Load pre-trained model
```
model = "tiiuae/falcon-7b"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
```
Write function generate text
```
def gen_text(text):
    text = translate_to_english(text)
    sequences = pipeline(
        text,
        max_length=256,
        do_sample=True,
        top_k=10,
        top_p=0.95,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        eos_token_id=tokenizer.eos_token_id,
        temperature=0.8
    )

    return sequences
```
Because our team is utilizing a pre-trained model suitable for English, we are using the Googletrans library to translate both the input and output sentences.
```
from googletrans import Translator

def translate_to_english(text):
    translator = Translator()
    translation = translator.translate(text, src='vi', dest='en')
    return translation.text

def translate_to_vietnamese(text):
    translator = Translator()
    translation = translator.translate(text, src='en', dest='vi')
    return translation.text
```

Config flask ngrok
```
os.environ["FLASK_ENV"] = "development"
app = Flask(__name__)
port = 5000

app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

ngrok.set_auth_token("2Y1Su8y7lLj3zxQKUHuQBxxScYL_5RouZaHebSgjpPi7jcKhR")
public_url = ngrok.connect(port).public_url
print(" * ngrok \"{}\" - \"http://127.0.0.1:{}\"".format(public_url,port))

app.config["BASE_URL"] = public_url
```
integrate model
```
@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        content = request.json
        prompt = content['prompt']

        result = gen_text(prompt)
        list_result = []
        for tmp in result:
            utf8_str = translate_to_vietnamese(tmp['generated_text']).encode('utf-8')

            # Mã hóa Base64
            base64_encoded = base64.b64encode(utf8_str)

            # Chuyển bytes thành chuỗi
            base64_encoded_text = base64_encoded.decode("utf-8")

            list_result.append(base64_encoded_text)
        return jsonify(list_result)

    except Exception as e:
        return jsonify({'error': str(e)})

threading.Thread(target = app.run, kwargs={"use_reloader": False}).start()
```
