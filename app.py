import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


#
def player_position():
    """
       For rendering results on HTML GUI
       """
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    # print(output)

    player = output
    if player == 0:
        player = "Defensive Midfielder"
    if player == 1:
        player = "Central Defender"
    if player == 2:
        player = "Attacking Midfielder"
    if player == 3:
        player = " GoalKeeper"
    if player == 4:
        player = "Striker"
    if player == 5:
        player = "Winger "
    if player == 6:
        player = "Wing Back"
    if player == 7:
        player = "Zone Midfielder"
    return player


@app.route('/predict', methods=['POST'])
def predict():
    # """
    # For rendering results on HTML GUI
    # """
    # int_features = [int(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)
    #
    # output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='PLayer Position is a   {}'.format(player_position()))


@app.route('/predict_api', methods=['POST'])
def predict_api():
    """
    For direct API calls trought request
    """
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
