
import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def normalize_text(text):
    text = re.sub(r"https?://\S+|www\.\S+", "", str(text))
    text = text.lower()
    text = "".join(
        char if char not in string.punctuation else " "
        for char in text
    )
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    text = " ".join(
        word for word in text.split()
        if word not in STOP_WORDS
    )

    text = " ".join(
        LEMMATIZER.lemmatize(word)
        for word in text.split()
    )

    return text
