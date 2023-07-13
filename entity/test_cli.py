import argparse

from entity_recognizer import EntityRecognizer

parser = argparse.ArgumentParser()
parser.add_argument("--lang", required=True)
args = parser.parse_args()

entity_recognizer = EntityRecognizer(lang=args.lang)
while True:
    sentence = input("Enter sentence to test:")
    if len(sentence) == 0:
        break
    entities = entity_recognizer.predict(sentence)
    for ent in entities:
        print(ent)
