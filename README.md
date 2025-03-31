With the increasing number of spam messages, users are frequently exposed to fraud, phishing attempts, and unwanted advertisements. Traditional spam filters often rely on keyword-based detection or server-side processing, which may compromise privacy. This project focuses on developing a mobile-friendly spam detection system using DistilBERT, a lightweight transformer model, to classify SMS messages as spam or not spam in real-time while maintaining data privacy. This model is optimized for real-time classification on mobile devices. The model ensures efficient, privacy-preserving, and accurate spam filtering without requiring cloud-based computation.

The Datasets and the respective python files used to run the model are shared here.

Model link-> https://drive.google.com/drive/folders/1C9-KHUQpu1rfr0bxlc-rog1rSxg3BKmA?usp=sharing

![image](https://github.com/user-attachments/assets/e6f570db-8de5-4d1a-af9c-3cd6fcc45225)
![image](https://github.com/user-attachments/assets/6d2ac2a0-12bc-4251-a759-b1a1ed1668a9)
![image](https://github.com/user-attachments/assets/ced8b891-361f-480c-bbe5-1481f8ec2281)

The system uses DistilBERT, a lightweight version of BERT, for text classification. The model architecture consists of:
•	Pre-trained DistilBERT Transformer Layer
•	Fully Connected Layer (Classification Head)
•	Softmax Activation for Binary Classification (Spam/Not Spam)
Key design choices include:
•	Using Bidirectional Context Learning for better language understanding.
•	Fine-tuning DistilBERT instead of training from scratch to leverage transfer learning.
•	Reduced Transformer Layers (6 instead of 12 in BERT) for efficiency.
BERT Model (Bidirectional Encoder Representations from Transformers)
BERT is a Transformer-based deep learning model developed by Google AI. It is designed for Natural Language Processing (NLP) tasks such as text classification, sentiment analysis, and question answering.

Key Features of BERT:
Bidirectional Understanding
Unlike traditional NLP models that process text left-to-right (like LSTMs), BERT reads both directions (left-to-right and right-to-left) at the same time.
This helps BERT understand context more accurately. Pre-trained on a Large Corpus. BERT is trained on massive text datasets like Wikipedia and BookCorpus using self-supervised learning. Uses the Transformer Architecture
BERT is based on the Transformer model, which relies on the self-attention mechanism to process words in relation to each other. Masked Language Model (MLM) Training
During training, random words are masked, and BERT learns to predict them based on surrounding words.
After pre-training, BERT can be fine-tuned for tasks like spam detection, sentiment analysis, question answering, etc.
Limitations of BERT:
Computationally Expensive – Requires a lot of memory and processing power.
Slow Inference Time – Not ideal for mobile or real-time applications.
DistilBERT – A Lighter, Faster Version of BERT
DistilBERT (Distilled BERT) is a smaller, optimized version of BERT, developed by Hugging Face. It is designed to provide similar performance while being faster and more efficient.
