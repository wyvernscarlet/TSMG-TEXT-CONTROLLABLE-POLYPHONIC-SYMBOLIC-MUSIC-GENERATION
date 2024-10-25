from datasets import load_dataset
from transformers import BertTokenizer, BertModel
import torch
import muspy
import clip
import openai

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

ds = load_dataset("amaai-lab/MidiCaps")
train_ds = ds['train']
captionset = train_ds['caption']
midiset = train_ds['location']


test_caption = captionset[0]
test_midi_path = midiset[0]

test_midi = muspy.read_midi(test_midi_path)


test_token = tokenizer(test_caption, return_tensors='pt')

with torch.no_grad():
    outputs = model(**test_token)

#bert


embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess = clip.load("ViT-B/32", device=device)

clip_token = clip.tokenize([test_caption]).to(device)

with torch.no_grad():
    text_features = model_clip.encode_text(clip_token)

text_features = text_features.cpu().numpy()

#clip

openai.api_key = ''

response = openai.Embedding.create(
    model = "text-embedding-ada-002",
    input = "Fuck"
)

embedding = response['data'][0]['embedding']


#print("Text Embedding shape:", text_features.shape)
#print("Text Embedding:", text_features)
print(f"embedding")
