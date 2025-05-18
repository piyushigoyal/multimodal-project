from PIL import Image
import json
import os

# Load Anole's model and tokenizer
from anole.model import AnoleModel
from anole.tokenizer import AnoleTokenizer

model = AnoleModel.from_pretrained('path_to_pretrained_model')
tokenizer = AnoleTokenizer.from_pretrained('path_to_tokenizer')

# Load MathVista sample
with open('MathVista/sample.json', 'r') as f:
    sample = json.load(f)

image_path = sample['image']
question = sample['question']

# Preprocess image
image = Image.open(image_path).convert('RGB')
image_tensor = model.preprocess_image(image)

# Tokenize question
question_tokens = tokenizer.encode(question)

# Create interleaved input
input_tokens = model.create_interleaved_input(image_tensor, question_tokens)

# Generate answer
output = model.generate(input_tokens)
answer = tokenizer.decode(output)

print(f"Question: {question}")
print(f"Answer: {answer}")
