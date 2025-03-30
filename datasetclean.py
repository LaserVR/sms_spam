
import random
import nltk
from nltk.corpus import wordnet
import pandas as pd

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    synonyms.discard(word)  # Remove original word
    return list(synonyms)

def synonym_replacement(sentence, n=1):
    words = sentence.split()
    new_words = words.copy()

    for _ in range(n):
        word_choices = [word for word in words if get_synonyms(word)]
        
        if not word_choices:
            continue  # No words have synonyms, skip augmentation
        
        word_to_replace = random.choice(word_choices)
        synonyms = get_synonyms(word_to_replace)

        if synonyms:
            synonym = random.choice(synonyms)
            index = new_words.index(word_to_replace)
            new_words[index] = synonym

    return " ".join(new_words)

def random_word_swap(text, n=2):
    words = text.split()
    if len(words) < 2:
        return text  # Skip short texts
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return " ".join(words)


file_path = r"C:\Users\Senthil Anand\Documents\sms spam\combined12_sms_spam.csv"
df = pd.read_csv(file_path, encoding="utf-8")


spam_messages = df[df["label"] == 1]["text"]

augmented_texts = [synonym_replacement(text) for text in spam_messages] + \
                  [random_word_swap(text) for text in spam_messages]

augmented_df = pd.DataFrame({"label": 1, "text": augmented_texts})
df = pd.concat([df, augmented_df], ignore_index=True)

# Save the new dataset
df.to_csv(r"C:\Users\Senthil Anand\Documents\sms spam\augmented_sms_spam.csv", index=False)
print("✅ Data augmentation complete. New dataset saved.")

import random
import nltk
from nltk.corpus import wordnet
import pandas as pd

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    synonyms.discard(word)  
    return list(synonyms)

def synonym_replacement(sentence, n=1):
    words = sentence.split()
    new_words = words.copy()

    for _ in range(n):
        word_choices = [word for word in words if get_synonyms(word)]
        
        if not word_choices:
            continue  
        
        word_to_replace = random.choice(word_choices)
        synonyms = get_synonyms(word_to_replace)

        if synonyms:
            synonym = random.choice(synonyms)
            index = new_words.index(word_to_replace)
            new_words[index] = synonym

    return " ".join(new_words)

def random_word_swap(text, n=2):
    words = text.split()
    if len(words) < 2:
        return text  # Skip short texts
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return " ".join(words)

file_path = r"C:\Users\Senthil Anand\Documents\sms spam\combined12_sms_spam.csv"
df = pd.read_csv(file_path, encoding="utf-8")

spam_messages = df[df["label"] == 1]["text"]

augmented_texts = [synonym_replacement(text) for text in spam_messages] + \
                  [random_word_swap(text) for text in spam_messages]

augmented_df = pd.DataFrame({"label": 1, "text": augmented_texts})
df = pd.concat([df, augmented_df], ignore_index=True)

df.to_csv(r"C:\Users\Senthil Anand\Documents\sms spam\augmented_sms_spam.csv", index=False)
print("✅ Data augmentation complete. New dataset saved.")
