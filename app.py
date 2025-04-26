from flask import Flask, render_template, send_from_directory, request, jsonify
from cosine_sim import compute_cosine_similarity

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/send-message', methods=['POST'])
def send_message():
    data = request.get_json()
    user_message = data.get('message')

    reply = compute_cosine_similarity(user_message)
    return jsonify({'reply': reply})

if __name__ == "__main__":
    app.run(debug=True)