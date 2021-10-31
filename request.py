import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url, json={'Height': 2, 'Crossing': 9, 'Finishing': 6,
                             'Dribbling': 78, 'Curve': 67, 'Agility': 56,
                             'Shotpower': 89, 'JUmping': 78, 'Aggression': 90,
                             'Positioning': 80, 'Vision': 84})

print(r.json())
