import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re

spam_messages = [
    "Congratulations! You’ve won a FREE iPhone 15! Click here to claim: [spam-link]",
    "Urgent! Your Netflix subscription is about to expire! Renew now at: [spam-link]",
    "FREE 50GB data from your provider! Claim now:",
    "Claim your FREE vacation to Hawaii! Call now: 1-800-SPAM-TRAP",
    "Win big now! You have been selected for a $1,000 gift card. Click here: [spam-link]",
    "Congratulations,Your personal loan of up to Rs 25 Lacs can be pre-approved check Now",
    "Get 2Star Frost - Free Refrigerator starting Rs 19990* with Easy EMI Benefit. Visit Croma store now T&C"
]

spam_text = " ".join(spam_messages)
spam_text = re.sub(r'[^A-Za-z0-9 ]+', '', spam_text).lower()

wordcloud = WordCloud(
    width=1000, height=500,
    background_color='black',
    colormap='coolwarm',
    max_words=100,
    contour_color='white',
    contour_width=1
).generate(spam_text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Most Common Words in Spam Messages", fontsize=16)
plt.show()
