
from transformers import AutoTokenizer, AutoModel,pipeline
from transformers import BartTokenizer, BartModel
import numpy as np
from sklearn.decomposition import PCA
import subprocess

def BART_t2t(text):
 summarize = pipeline("summarization", model = "facebook/bart-large")
 return summarize(text, min_length = 3,max_length = 20, do_sample = False)

def BART_pl(text):
  tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
  model = BartModel.from_pretrained('facebook/bart-large')
  text = BART_t2t(text)[0]['summary_text']
  inputs = tokenizer(text, return_tensors="pt")
  return model(**inputs)

pca =  PCA(n_components = 3)
def reduce_channels(encoding): #dimensionality reduction using PCA and mean correction
 array = encoding.detach().cpu().numpy().reshape(-1,1024).T
 X_new = array - np.mean(array)
 return pca.fit_transform(X_new)

def text2PCA(word):
  return reduce_channels(BART_pl(word)[0])

#now we can finally convert this array to (32,32,3) image array
def t2i(word):
  return text2PCA(word).reshape(32,32,3)
