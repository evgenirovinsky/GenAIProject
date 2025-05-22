from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained('cointegrated/rubert-tiny2-cedr-emotion-detection')
print('id2label:', model.config.id2label)
print('label2id:', model.config.label2id) 