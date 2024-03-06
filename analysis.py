import pandas as pd
import numpy as np
import re
import string
import os 
import warnings
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from collections import Counter, defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ignore warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

# Download NLTK POS tagger and stop words data
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')

# Get English stop words
stop_words = set(stopwords.words('english'))

# Read sentences from file
sentences = []
with open('kendrick.txt') as file:
    sentences += file.readlines()

# Function to get POS tags for a sentence
def get_pos_tags(sentence):
    words = nltk.word_tokenize(sentence)
    pos_tags = pos_tag(words)
    return pos_tags

# Modify the clean function to filter out stop words
def clean(sentence):
    clean = sentence.strip().lower().encode("utf8").decode("ascii", 'ignore')
    clean = re.sub(r"[^a-zA-Z0-9]", ' ', sentence)
    words = nltk.word_tokenize(clean)
    #filtered_words = [word for word in words if word not in stop_words]
    #pos_tags = pos_tag(filtered_words)
    pos_tags = pos_tag(words)
    clean_with_pos = " ".join([f"{word}_{pos}" for word, pos in pos_tags])
    return clean_with_pos

# Apply the function to clean sentences
cleaned_sentences = [clean(sentence) for sentence in sentences if sentence.strip() != '']

# Filler words to be removed
filler_words = ["uh", "um", "like", "you know", "well", "so", "actually", "basically"]

# Function to remove filler words
def remove_filler_words(sentence):
    words = sentence.split()
    filtered_words = [word for word in words if word.lower() not in filler_words]
    return " ".join(filtered_words)

# Apply the function to remove filler words
cleaned_sentences_without_fillers = [remove_filler_words(sentence) for sentence in cleaned_sentences]


# Function to get last word and its POS
# def get_last_word_pos(sentence):
#     if len(sentence.split()) > 1:
#         last_word, pos = sentence.split(" ")[-1].split('_')
#         return last_word, pos

# Get the most common last word POS for each sentence
# last_word_pos_list = [get_last_word_pos(sentence) for sentence in cleaned_sentences if get_last_word_pos(sentence)]

# Count the occurrences of each last word POS
# pos_counter = Counter(last_word_pos_list)

# Display the results
# for pos, count in pos_counter.most_common():
#     print(f"{pos}: {count} occurrences")
    
# Combine all cleaned sentences into a single string
def get_word_pos(sentence):
    for word in sentence.split():
        if len(sentence.split()) > 1:
            word, pos = word.split('_')
            return word, pos
words_pos = [get_word_pos(sentence) for sentence in cleaned_sentences_without_fillers]
words = [item[0] for item in words_pos if item is not None]
tags = [item[1] for item in words_pos if item is not None]
word_text = " ".join(words)

# Initialize a dictionary to store counts of occurrences
# pos_counts = defaultdict(lambda: defaultdict(int))

# Loop through each (word, tag) tuple in the dataset
# for i in range(len(words) - 1):
#     current_word = words[i]
#     current_pos = tags[i]
#     next_word = words[i + 1]
#     next_pos = tags[i + 1]
#     pos_counts[current_pos][next_pos] += 1

# Compute probabilities
# pos_probabilities = {}
# for current_pos, next_pos_count in pos_counts.items():
#     total_count = sum(next_pos_count.values())
#     probabilities = {next_pos: count / total_count for next_pos, count in next_pos_count.items()}
#     pos_probabilities[current_pos] = probabilities

noun_tags = {'NN', 'NNS', 'NNP', 'NNPS'}
verb_tags = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
adjective_tags = {'JJ', 'JJR', 'JJS'}
adverb_tags = {'RB', 'RBR', 'WRB'}
pronoun_tags = {'PRP', 'PRP$'}
preposition_tags = {'IN'}
determiner_tags = {'DT'}
conjunction_tags = {'CC'}
numeral_tags = {'CD'}
modal_tags = {'MD'}
interjection_tags = {'UH'}
to_tags = {'TO'}
wh_determiner_tags = {'WDT'}
wh_pronoun_tags = {'WP'}

# Function to extract specific POS tags from a sentence
def extract_specific_pos(sentence, pos_tags):
    words = nltk.word_tokenize(sentence)
    pos_tags_full = pos_tag(words)
    specific_pos_tags = [word for word, pos in pos_tags_full if pos in pos_tags]
    return specific_pos_tags

# Extract specific POS tags from all cleaned sentences
nouns_list = [nouns for nouns in words_pos if nouns is not None and nouns[1] in noun_tags]
verbs_list = [verbs for verbs in words_pos if verbs is not None and verbs[1] in verb_tags]
adjectives_list = [adjectives for adjectives in words_pos if adjectives is not None and adjectives[1] in adjective_tags]
adverb_list = [adverbs for adverbs in words_pos if adverbs is not None and adverbs[1] in adverb_tags]
pronoun_list = [pronouns for pronouns in words_pos if pronouns is not None and pronouns[1] in pronoun_tags]
preposition_list = [prepositions for prepositions in words_pos if prepositions is not None and prepositions[1] in preposition_tags]
determiner_list = [determiners for determiners in words_pos if determiners is not None and determiners[1] in determiner_tags]
conjunction_list = [conjunctions for conjunctions in words_pos if conjunctions is not None and conjunctions[1] in conjunction_tags]
numeral_list = [numerals for numerals in words_pos if numerals is not None and numerals[1] in numeral_tags]
modal_list = [modals for modals in words_pos if modals is not None and modals[1] in modal_tags]
interjection_list = [interjections for interjections in words_pos if interjections is not None and interjections[1] in interjection_tags]
to_list = [tos for tos in words_pos if tos is not None and tos[1] in to_tags]
wh_determiner_list = [wh_determiners for wh_determiners in words_pos if wh_determiners is not None and wh_determiners[1] in wh_determiner_tags]
wh_pronoun_list = [wh_pronouns for wh_pronouns in words_pos if wh_pronouns is not None and wh_pronouns[1] in wh_pronoun_tags]

# Create a DataFrame for each POS category
df_nouns = pd.DataFrame(set(nouns_list), columns=['nouns', 'noun_tag'])
df_verbs = pd.DataFrame(set(verbs_list), columns=['verbs', 'verb_tag'])
df_adjectives = pd.DataFrame(set(adjectives_list), columns=['adjectives', 'adjective_tag'])
df_adverbs = pd.DataFrame(set(adverb_list), columns=['adverbs', 'adverb_tag'])
df_pronouns = pd.DataFrame(set(pronoun_list), columns=['pronouns', 'pronoun_tag'])
df_prepositions = pd.DataFrame(set(preposition_list), columns=['prepositions', 'preposition_tag'])
df_determiners = pd.DataFrame(set(determiner_list), columns=['determiners', 'determiner_tag'])
df_conjunctions = pd.DataFrame(set(conjunction_list), columns=['conjunctions', 'conjunction_tag'])
df_numerals = pd.DataFrame(set(numeral_list), columns=['numerals', 'numeral_tag'])
df_modals = pd.DataFrame(set(modal_list), columns=['modals', 'modal_tag'])
df_interjections = pd.DataFrame(set(interjection_list), columns=['interjections', 'interjection_tag'])
df_tos = pd.DataFrame(set(to_list), columns=['tos', 'to_tag'])
df_wh_determiners = pd.DataFrame(set(wh_determiner_list), columns=['wh_determiners', 'wh_determiner_tag'])
df_wh_pronouns = pd.DataFrame(set(wh_pronoun_list), columns=['wh_pronouns', 'wh_pronoun_tag'])


# Combine all DataFrames
dfs = [df_nouns, df_verbs, df_adjectives, df_adverbs, df_pronouns, df_prepositions,
       df_determiners, df_conjunctions, df_numerals, df_modals, df_interjections,
       df_tos, df_wh_determiners, df_wh_pronouns]

# Concatenate DataFrames along columns (axis=1)
df_combined = pd.concat(dfs, axis=1)

# Save the DataFrame to a CSV file
df_combined.to_csv('pos_tags_data.csv', index=False)

# Display the DataFrame
print(df_combined.head())
# Create DataFrames for each POS category
# Continue with the previous code...


# This part of the code is processing the cleaned sentences to extract specific parts of speech (POS)
# tags such as nouns, verbs, adjectives, adverbs, pronouns, prepositions, determiners, conjunctions,
# numerals, modals, interjections, "to" tags, wh-determiners, and wh-pronouns.
# df_nouns = pd.DataFrame(nouns_list, columns=['Nouns'])
# df_verbs = pd.DataFrame(verbs_list, columns=['Verbs'])
# df_adjectives = pd.DataFrame(adjectives_list, columns=['Adjectives'])
# df_adverbs = pd.DataFrame(adverb_list, columns=['Adverbs'])
# df_pronouns = pd.DataFrame(pronoun_list, columns=['Pronouns'])
# df_prepositions = pd.DataFrame(preposition_list, columns=['Prepositions'])
# df_determiners = pd.DataFrame(determiner_list, columns=['Determiners'])
# df_conjunctions = pd.DataFrame(conjunction_list, columns=['Conjunctions'])
# df_numerals = pd.DataFrame(numeral_list, columns=['Numerals'])
# df_modals = pd.DataFrame(modal_list, columns=['Modals'])
# df_interjections = pd.DataFrame(interjection_list, columns=['Interjections'])
# df_to = pd.DataFrame(to_list, columns=['To'])
# df_wh_determiners = pd.DataFrame(wh_determiner_list, columns=['Wh_Determiners'])
# df_wh_pronouns = pd.DataFrame(wh_pronoun_list, columns=['Wh_Pronouns'])

# # Combine all DataFrames
# dfs = [df_nouns, df_verbs, df_adjectives, df_adverbs, df_pronouns, df_prepositions,
#        df_determiners, df_conjunctions, df_numerals, df_modals, df_interjections,
#        df_to, df_wh_determiners, df_wh_pronouns]

# # Ensure all DataFrames have the same number of rows by padding with NaN if needed
# max_rows = max(df.shape[0] for df in dfs)
# dfs_padded = [df.reindex(range(max_rows)).fillna('') for df in dfs]

# # Concatenate DataFrames along columns (axis=1)
# df_combined = pd.concat(dfs_padded, axis=1)

# # Save the DataFrame to a CSV file
# df_combined.to_csv('pos_tags_data.csv', index=False)

# # Display the DataFrame
# print(df_combined.head())



# Generate and display a word cloud
# wordcloud = WordCloud(width=800, height=400, background_color="white").generate(word_text)
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()

# plt.savefig('wordcloud_plot.png')



# # Flatten the lists
# flat_nouns = [item for sublist in nouns_list for item in sublist]
# flat_verbs = [item for sublist in verbs_list for item in sublist]
# flat_adjectives = [item for sublist in adjectives_list for item in sublist]

# # Display the distribution of each POS tag
# print("Nouns Distribution:")
# print(Counter(flat_nouns).most_common())

# print("\nVerbs Distribution:")
# print(Counter(flat_verbs).most_common())

# print("\nAdjectives Distribution:")
# print(Counter(flat_adjectives).most_common())


# words, pos_tags, counts = zip(*[(word, pos, count) for (word, pos), count in pos_counter.items()])


# # Create the 3D bar plot using scatter3d
# fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

# fig.add_trace(go.Scatter3d(
#     x=words,
#     y=pos_tags,
#     z=counts,
#     mode='markers',
#     marker=dict(size=8, color=counts, colorscale='viridis', opacity=0.7, line=dict(color='rgb(0,0,0)', width=0.5)),
#     text=[f'{word} ({pos}): {count} occurrences' for word, pos, count in zip(words, pos_tags, counts)],
#     hoverinfo='text'
# ))

# # Set axis titles and layout
# fig.update_layout(scene=dict(xaxis_title='Words', yaxis_title='POS Tags', zaxis_title='Counts'),
#                   title='Interactive 3D Bar Plot of POS Tags',
#                   width=800, height=600)

# # Show the plot
# fig.show()