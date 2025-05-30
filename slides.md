---
# Слайди презентації фінального проєкту: Ukrainian Text Emotion Analysis System

---

## Слайд 1: Титульний слайд

# Ukrainian Text Emotion Analysis System

*Автор: [Ваше ім'я]*

---

## Слайд 2: Motivation

### Мотивація та опис задачі

- **Проблема:** Емоційний аналіз українською мовою є важливим для соцмереж, підтримки, освіти, але існує дефіцит відкритих інструментів.
- **Мета:** Розробити систему, яка автоматично визначає емоції та сентимент у текстах українською, використовуючи сучасні трансформери.

---

## Слайд 3: Introduction

### Огляд існуючих підходів і релевантних робіт

- **Існуючі підходи:**
  - Класичні методи (словники, SVM, Naive Bayes).
  - Сучасні трансформери (BERT, RoBERTa) для інших мов.
- **Релевантні роботи:**
  - Hugging Face моделі (англійська, російська, багатомовні).
  - Відсутність відкритих рішень для української мови з підтримкою емоційної класифікації.

---

## Слайд 4: Description

### Технічний підхід

- **Архітектура системи:**
  - Модульна структура (Sentiment Analyzer, Emotion Analyzer, REST API).
  - Sentiment Analyzer: модель `cointegrated/rubert-tiny2-cedr-emotion-detection` (класифікація на позитивний, негативний, нейтральний).
  - Emotion Analyzer: модель `j-hartmann/emotion-english-distilroberta-base` (визначення емоцій: радість, сум, злість, страх, здивування, нейтрально).
  - REST API (FastAPI) для інтеграції.
- **Інструменти:** Python, Hugging Face Transformers, FastAPI, requests, pandas, numpy.
- **Алгоритми:** Трансформери для класифікації тексту, мапінг емоцій у категорії сентименту.

---

## Слайд 5: Demo

### Демонстрація роботи системи

- **Використання через API:**
  - POST-запит на `/analyze` з текстом українською.
  - Відповідь: сентимент, домінуюча емоція, розподіл емоцій.
- **Приклад:**
  ```
  Запит: "Я дуже радий зустріти вас!"
  Відповідь:
    Sentiment: neutral
    Dominant Emotion: радість
    Emotion Mixture: радість, злість, здивування
  ```
- **Візуалізація:** (Скріншоти Swagger UI, Jupyter Notebook, або відео роботи API.)

---

## Слайд 6: Results

### Аналіз отриманих результатів

- **Якість роботи:**
  - Висока точність на тестових прикладах.
  - Підтримка багатьох емоцій, адаптація під українську мову.
- **Сильні сторони:**
  - Використання сучасних трансформерів.
  - Гнучкість (API, пакетна обробка).
  - Відкритий код, простота розгортання.
- **Слабкі сторони:**
  - Залежність від якості датасетів для української.
  - Можливі помилки на складних або іронічних текстах.
- **Метрики:** (Якщо є — навести точність, recall, F1 на тестових даних або прикладах.)

---

## Слайд 7: Conclusions

### Висновки та майбутня робота

- **Висновки:**
  - Система дозволяє ефективно аналізувати емоції та сентимент українських текстів.
  - Може бути використана для моніторингу соцмереж, підтримки клієнтів, освітніх цілей.
- **Майбутні напрямки:**
  - Додавання підтримки інших мов.
  - Покращення точності через донавчання на українських датасетах.
  - Додавання візуалізації (графіки, інтерактивні дашборди).
  - Інтеграція з іншими сервісами (Telegram-бот, веб-інтерфейс).

---

## Слайд 8: Thank You

# Дякую за увагу!

*Готовий відповідати на питання.*

--- 