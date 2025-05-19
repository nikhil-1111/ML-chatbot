# ML-chatbot
This project is a machine learning-based customer support chatbot that uses Natural Language Processing (NLP) to classify user queries into predefined intents. It handles greetings, FAQs, order status, and more — with escalation to a human agent for unknown inputs

# 🤖 Customer Support Chatbot (ML + NLP)

A machine learning-based chatbot that uses **Natural Language Processing (NLP)** to classify customer queries into intents. This project simulates a simple yet smart virtual assistant capable of answering FAQs, providing mock order statuses, collecting contact info, and escalating unknown issues to a human.

---

## 📌 Features

- ✅ Greets users and maintains a friendly tone
- ✅ Answers **20+ frequently asked questions**
- ✅ Provides **mock order status** using randomly generated order IDs
- ✅ Collects user **name and email** when needed
- ✅ Escalates unrecognized queries to a human support agent
- ✅ Built with a **Linear SVM classifier** using TF-IDF vectors
- ✅ Reaches **82%+ accuracy** with hyperparameter tuning
- ✅ Easily expandable with more intents

---

## 🧠 Technologies Used

- Python 3
- Jupyter Notebook
- Scikit-learn (LinearSVC, GridSearchCV)
- Pandas & NumPy
- TF-IDF Vectorizer (from sklearn)

---

## 🛠 Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/chatbot-intent-classifier.git
cd chatbot-intent-classifier
