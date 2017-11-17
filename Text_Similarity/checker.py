import nltk
import os
from Text_Similarity import spell

# obtains individual words from text file
curr_dir = os.pardir
store = []
with open(curr_dir + "\\Text_Similarity\\Check_Text.txt") as f:
    store = [word for line in f for word in line.split()]

# calculates average error of words compared to dictionary
average_error = 0
cumulative_error = 0
num_words = len(store)
for word in store:
    distance = nltk.edit_distance(word, spell.correction(word))
    cumulative_error = cumulative_error + distance/len(word)

# prints average error
average_error = cumulative_error/num_words
print(average_error)