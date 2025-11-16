import weaviate
import json

client = weaviate.Client(url="http://localhost:8080")

with open("KJV.json", "r") as f:
    bible = json.load(f)

for book in bible['books']:
    for chapter in book['chapters']:
        for verse in chapter['verses']:
            obj = {
                "name": verse['name'],
                "chapter": verse['chapter'],
                "verse": verse['verse'],
                "text": verse['text']
            }

            client.data_object.create(
                data_object=obj,
                class_name="Verse"
            )
