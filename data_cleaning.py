from data_load import dataset
from data_proceeding import dialog_separator
import nltk
from nltk.tokenize import word_tokenize
import string

# nltk.download("punkt")
# nltk.download('punkt_tab')

#  Tokenizing the dialogs
train = dialog_separator(dataset["train"])
tokenized_words = [ ( word_tokenize(dialog.lower()) ) for dialog, act in train ]

print(tokenized_words[:15])