import csv
import sys
import json
clotho = sys.argv[1]
audio_caps= sys.argv[2]
def read_json(path):

  with open(path, 'r') as fd:
      data = fd.read()

  return json.loads(data)

def get_captions_from_json(data):

  captions = []
  for d in data['data']:
    captions.append(d['caption'])
  return captions


captions = []
with open(clotho, 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    captions += row[1:]

with open('clotho.json', 'w') as f:
    json.dump(captions, f)

captions = []
with open(audio_caps, 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    captions.append(row[1])

with open('audiocaps.json', 'w') as f:
    json.dump(captions, f)



