import json

# Load data from JSON file
with open('qa-dataset.json', 'r') as file:
    data = json.load(file)

# ACTS:

# greeting 0
# inquiry 1
# farewell 2
# compliment 3
# gratitude 4
# apology 5
# request 6
# confirmation 7
# suggestion 8
# expression 9