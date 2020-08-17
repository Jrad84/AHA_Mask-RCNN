import json
import os
import re


with open('aha_labels.json', 'r') as myfile:
    data=myfile.read()

input = json.loads(data)
annotations = list(input)

train_dir = os.listdir('dataset/train')
test_dir = os.listdir('dataset/test')

train = []
test = []

for image in train_dir:
    train.append(image)

for image in test_dir:
    test.append(image)

output = []
d = {}
for key,value in input.items():

    # replace directory as needed
    for im in train:
        r = re.match(im, key)
        if r:
            d[key] = value
            # temp = [key,value]
            # output.append(temp)

# itemDict = {out[0]: out[0:] for out in output}


newfile = json.dumps(d)

f = open('aha_labels1.json', "x")
f.write(newfile)
f.close()
