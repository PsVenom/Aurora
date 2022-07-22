
from transformers import AutoTokenizer, AutoModel,pipeline
from transformers import BartTokenizer, BartModel
from transformers import CLIPProcessor, CLIPModel

import numpy as np
from sklearn.decomposition import PCA
import subprocess

model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
summarize = pipeline("summarization", model = "facebook/bart-large")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
model = BartModel.from_pretrained('facebook/bart-large')
pca =  PCA(n_components = 3)

def BART_t2t(text):
 return summarize(text, min_length = 3,max_length = 20, do_sample = False)

def BART_pl(text):
  text = BART_t2t(text)[0]['summary_text']
  inputs = tokenizer(text, return_tensors="pt")
  return model(**inputs)



def reduce_channels(encoding): #dimensionality reduction using PCA and mean correction
 array = encoding.detach().cpu().numpy().reshape(-1,1024).T
 X_new = array - np.mean(array)
 return pca.fit_transform(X_new)

def text2PCA(word):
  return reduce_channels(BART_pl(word)[0])

#now we can finally convert this array to (32,32,3) image array
def t2i(word):
  return text2PCA(word).reshape(32,32,3)
 
def L_clip(Xt,Yi): #Xt is a string, Yi is an image array
  inputs = processor(text= Xt, images= Yi, return_tensors="pt", padding=True)
  outputs = model_clip(**inputs)
  logits_per_image = outputs.logits_per_image
  return 1 - (logits_per_image/100) #returns a pytorch tensor
