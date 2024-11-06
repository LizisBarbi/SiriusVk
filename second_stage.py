import requests
import pandas as pd
import re

# Пример запроса к API «ВКонтакте»
def fetch_vk_data(token, version, domain, count, offset):
    response = requests.get(
        'https://api.vk.com/method/wall.get',
        params={
            'access_token': token,
            'v': version,
            'domain': domain,
            'count': count,
            'offset': offset
        }
    )
    return response.json()

# Пример сбора данных из нескольких источников
domains = ['domain1', 'domain2', 'domain3']
token = 'YOUR_ACCESS_TOKEN'
version = '5.131'
count = 100
offset = 0

data = []
for domain in domains:
    response = fetch_vk_data(token, version, domain, count, offset)
    data.extend(response['response']['items'])

df = pd.DataFrame(data)

# Предобработка данных
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Удаление URL
    text = re.sub(r'[^а-яА-Яa-zA-Z0-9\s]', '', text)  # Удаление специальных символов
    return text.lower()

df['text'] = df['text'].apply(clean_text)

# Анализ данных (EDA)
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x='type', data=df)
plt.show()

# Фильтрация и очистка данных
df = df.dropna(subset=['text', 'type'])
df = df[df['text'] != '']

# Разделение данных на обучающую и тестовую выборки
from sklearn.model_selection import train_test_split

train_texts, test_texts, train_labels, test_labels = train_test_split(df['text'], df['type'], test_size=0.2, random_state=42)

import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report

# Токенизация текста
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True, max_length=512)

# Преобразование меток в числовые значения
label_map = {label: idx for idx, label in enumerate(df['type'].unique())}
train_labels = [label_map[label] for label in train_labels]
test_labels = [label_map[label] for label in test_labels]

# Создание датасетов
class PostDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = PostDataset(train_encodings, train_labels)
test_dataset = PostDataset(test_encodings, test_labels)

# Обучение модели
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=len(label_map))

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

# Оценка модели
predictions = trainer.predict(test_dataset)
pred_labels = torch.argmax(predictions.predictions, dim=-1)
print(classification_report(test_labels, pred_labels, target_names=label_map.keys()))
