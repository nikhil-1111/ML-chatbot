from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
import re

# Load trained chatbot model
model = joblib.load(r'chatbot_model.pkl')

app = Flask(__name__)
CORS(app)
# ----------- Text cleaning ----------
def clean_input(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ----------- Intent prediction ----------
def predict_intent(text):
    probs = model.predict_proba([text])[0]
    max_index = probs.argmax()
    intent = model.classes_[max_index]
    confidence = probs[max_index]
    return intent, confidence

# ----------- Chat response generator ----------
def get_response(intent, user_input):
    if intent == 'greet':
        return "Hello there! How can I help you today?"
    elif intent == 'ok':
        return "Okay! If you have any questions, feel free to ask or type 'quit' to exit."
    elif intent == "order_status":
        order_id = re.findall(r'#?\d{4,6}', user_input)
        order_id = order_id[0].lstrip('#') if order_id else None
        mock_orders = {
            "12345": "📦 Order 12345 is out for delivery.",
            "67890": "✅ Order 67890 has been delivered."
        }
        return mock_orders.get(order_id, "⚠️ I couldn't find that order ID.") if order_id else "Please provide your order number."
    elif intent == "faq_payment":
        return "💳 We accept Visa, MasterCard, and PayPal."
    elif intent == "faq_hours":
        return "🕘 Our store hours are 9 AM to 9 PM, Monday to Saturday."
    elif intent == "faq_returns":
        return "🔄 You can return items within 30 days of purchase with a receipt."
    elif intent == "human_help":
        return "☎️ Please hold while I connect you to a human representative."
    elif intent == "faq_shipping":
        return "🚚 We offer free shipping on orders over $50. Standard shipping takes 3-5 business days."
    elif intent == "faq_contact":
        return "📞 You can reach us at ABC@company.com or 1-800-555-01**"
    elif intent == "thanking":
        return "🙏 You're welcome! If you have any more questions, just ask."
    else:
        return "❓ I'm not sure how to help with that. Please rephrase or contact support at 1-800-555-01**."

# ----------- Flask API route ----------
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    cleaned = clean_input(user_input)
    intent, confidence = predict_intent(cleaned)

    if confidence < 0.4:
        response = f"🤖 Hmm... I'm not sure I understood that (Intent: {intent}, Confidence: {confidence:.2f}). Could you rephrase?"
    else:
        response = get_response(intent, user_input)

    return jsonify({
        'response': response,
        'intent': intent,
        'confidence': round(confidence, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
