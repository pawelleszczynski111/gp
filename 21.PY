import os 
import pickle 
from google_auth_oauthlib.flow import InstalledAppFlow 
from googleapiclient.discovery import build
import base64
import pickle
from google.auth.transport.requests import Request
from googleapiclient.discovery import build


import json
import random
import math

SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

def authorize_and_save_token():
    if os.path.exists('token.pkl'):
        print("Plik token.pkl już istnieje.")
        return

    flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
    creds = flow.run_local_server(port=0)

    with open('token.pkl', 'wb') as token_file:
        pickle.dump(creds, token_file)
    print("Autoryzacja zakończona i token zapisany jako token.pkl.")

    service = build('gmail', 'v1', credentials=creds)
    results = service.users().labels().list(userId='me').execute()
    print("Twoje etykiety Gmail:")
    for label in results.get('labels', []):
        print("•", label['name'])

if __name__ == '__main__':
    authorize_and_save_token()



def load_email_data(spam_path, ham_path):
    with open(spam_path, encoding="utf-8") as f1, open(ham_path, encoding="utf-8") as f2:
        spam_data = json.load(f1)
        ham_data = json.load(f2)
    return spam_data + ham_data


def train_test_split(data, test_ratio=0.2):
    random.shuffle(data)
    cut = int(len(data) * (1 - test_ratio))
    return data[:cut], data[cut:]

def preprocess(text):
    return text.lower().replace("–", " ").replace("-", " ").replace(",", " ")\
        .replace(".", " ").replace("!", " ").replace("?", " ").split()

def train_naive_bayes(train_data, alpha=1.0):
    class_counts = {}
    word_counts = {}
    total_words = {}

    for rec in train_data:
        label = rec["label"]
        class_counts[label] = class_counts.get(label, 0) + 1
        word_counts.setdefault(label, {})
        total_words.setdefault(label, 0)

        words = preprocess(rec["text"])
        for word in words:
            word_counts[label][word] = word_counts[label].get(word, 0) + 1
            total_words[label] += 1

    vocab = set()
    for wc in word_counts.values():
        vocab.update(wc.keys())

    return {
        "class_counts": class_counts,
        "word_counts": word_counts,
        "total_words": total_words,
        "vocab": vocab,
        "alpha": alpha,
        "total_docs": len(train_data)
    }


def log_prob(model, words, class_name):
    logp = math.log(model["class_counts"][class_name] / model["total_docs"])
    V = len(model["vocab"])
    a = model["alpha"]
    for word in words:
        wc = model["word_counts"][class_name].get(word, 0)
        logp += math.log((wc + a) / (model["total_words"][class_name] + a * V))
    return logp

def predict(model, text):
    words = preprocess(text)
    best_class, best_log = None, -float("inf")
    for c in model["class_counts"]:
        lp = log_prob(model, words, c)
        if lp > best_log:
            best_class, best_log = c, lp
    return best_class

def evaluate_model(model, test_data):
    correct = 0
    for rec in test_data:
        prediction = predict(model, rec["text"])
        if prediction == rec["label"]:
            correct += 1
    accuracy = correct / len(test_data)
    print(f"Skuteczność na zbiorze testowym: {accuracy * 100:.2f}%")
    return accuracy



# ////////////////////////////////////////////////
def get_gmail_service():
    with open("token.pkl", "rb") as token:
        creds = pickle.load(token)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())

    service = build("gmail", "v1", credentials=creds)
    return service


def fetch_unread_emails_from_label(model, label_name='Test'):
    service = get_gmail_service()

    label_test = get_label_id(service, label_name)
    label_ham = get_label_id(service, 'ham')
    label_spam = get_label_id(service, 'spam2')

    if not label_test:
        print(f"Etykieta '{label_name}' nie została znaleziona.")
        return []

    response = service.users().messages().list(
        userId='me',
        labelIds=[label_test, 'UNREAD'],
        maxResults=100
    ).execute()

    messages = response.get('messages', [])
    email_list = []

    for msg in messages:
        msg_id = msg['id']
        message = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
        payload = message.get('payload', {})
        headers = payload.get('headers', [])

        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '')
        body = get_message_body(payload)
        full_text = f"{subject} {body.strip()}"

        prediction = predict(model, full_text)

        add_labels = [label_spam if prediction == 'spam' else label_ham]
        remove_labels = [label_test]

        service.users().messages().modify(
            userId='me',
            id=msg_id,
            body={
                'addLabelIds': add_labels,
                'removeLabelIds': remove_labels
            }
        ).execute()

        print(f"[ZMIANA] Wiadomość '{subject[:40]}...' → {prediction.upper()} (etykieta zmieniona)")



def get_label_id(service, label_name):
    labels = service.users().labels().list(userId='me').execute().get('labels', [])
    for label in labels:
        if label['name'].lower() == label_name.lower():
            return label['id']
    return None


def get_message_body(payload):
    parts = payload.get('parts')
    if parts:
        for part in parts:
            if part['mimeType'] == 'text/plain':
                data = part['body'].get('data')
                if data:
                    return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
    else:
        body_data = payload['body'].get('data')
        if body_data:
            return base64.urlsafe_b64decode(body_data).decode('utf-8', errors='ignore')
    return "(brak treści)"



from pprint import pprint

def main():
    data = load_email_data("spam_emails.json", "ham_emails.json")

    train, test = train_test_split(data)

    model = train_naive_bayes(train)

    evaluate_model(model, test)

    fetch_unread_emails_from_label(model, label_name='Test')

# ------------------------------
# 5. Uruchomienie
# ------------------------------
if __name__ == "__main__":
    main()