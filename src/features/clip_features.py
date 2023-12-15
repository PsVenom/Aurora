import torch
from clip import *
from PIL import Image
import tensorflow as tf
from src.data.make_dataset import *
import torch.nn as nn
import numpy as np

class CLIP2SG(nn.Module):
    def __init__(self):
        super(CLIP2SG, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fc = nn.Linear(in_features=77, out_features= 512)

    def forward(self, x):
        x = self.fc(x)
        x = nn.Sigmoid()(x)

        return x



@tf.function
def L_clip(Xt,Yi, model, preprocess, device): #Xt is a string array, Yi is an image array (both tensorflow)
    #print("Yi:", Yi.shape)

    # Xt = BART_t2t(Xt)
    Xt = clip.tokenize(Xt.split("'")[1::2]).to(device)
    Yi = Yi.numpy()
    #print("Code works till here")
    #print(f"Yi numpy: {Yi}")
    #print(f"Yi[0] shape: {Yi[0].shape}")
    typed_np = [Yi[i].astype(np.uint8) for i in range(0, Yi.shape[0])]
    img = [Image.fromarray(typed_np[i]) for i in range(0, Yi.shape[0])]
    Yi = [preprocess(img[i]) for i in range(0, Yi.shape[0])]
    with torch.no_grad():
        image_features = model.encode_image(torch.tensor(np.stack(Yi)).cuda())
        text_features = model.encode_text(Xt)
    #  similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
    #  loss = 1- np.sum(np.abs(np.diag(similarity)))
    loss = tf.keras.losses.MeanAbsoluteError(
        name='huber_loss'
    )
    return loss(text_features.cpu().numpy(), image_features.cpu().numpy())
tf.Graph
