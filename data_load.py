import json
import os

path1 = os.path.join(os.path.dirname(__file__), 'general-dataset.json')
path2 = os.path.join(os.path.dirname(__file__), 'wikiqa-dataset.json')
path3 = os.path.join(os.path.dirname(__file__), 'sciq-dataset.json')

# Load data from JSON file
with open(path1, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Load wiki qa dataset
with open(path2, 'r', encoding='utf-8') as file:
    wiki_data = json.load(file)

# Load SciQ qa dataset
with open(path3, 'r', encoding='utf-8') as file:
    sciq_data = json.load(file)


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
# question 10
# answer 11