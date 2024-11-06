from transformers import LlamaTokenizer, LlamaForCausalLM
import torch

# Загрузка модели и токенизатора
tokenizer = LlamaTokenizer.from_pretrained("path/to/vikhr-llama3.1-8b")
model = LlamaForCausalLM.from_pretrained("path/to/vikhr-llama3.1-8b")

def generate_hashtags(text):
    # Подготовка входных данных
    input_text = f"Generate relevant hashtags for the following text: {text}"
    inputs = tokenizer(input_text, return_tensors="pt")

    # Генерация текста
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50)

    # Декодирование результата
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Извлечение хэштегов
    hashtags = extract_hashtags(generated_text)
    return hashtags

def extract_hashtags(generated_text):
    # Извлечение слов, начинающихся с #
    return [word for word in generated_text.split() if word.startswith('#')]

# Пример использования
text_input = "Artificial intelligence is transforming technology and healthcare with groundbreaking innovations."
hashtags = generate_hashtags(text_input)
print(hashtags)
