import requests

# Fetch ImageNet labels
url = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
response = requests.get(url)
labels = response.json()

# Get label for Category ID 192
category_id = 446
label = labels[category_id]
print(f"Category ID {category_id} corresponds to: {label}")