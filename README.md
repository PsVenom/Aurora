# Aurora
## So here are all the helper functions that we will use for this project, based on priority
### L_clip - returns the CLIP loss (it's a modified cosine similarity formula). As it turns out, CLIP returns a text-image similarity score between 0 and 100. THis is perfect for text-image loss
### t2i - converts a given string S into a tensor of shape(32,32,3). This will be used as the stylegan input, and the goal is to map all latent spaces on our text vector
### text2PCA - converts a given string S into a vector of shape (1024,3). It can be used as an input for AdaIN layer
### BART_pl - converts a given string S into an embedding vector of shape(n,1024), where n can vary from 3 to 20. We'll use this to tokenize our text, which can then be used to calculate L(clip). We will also use this as a summarizer for unreasonably long text
 
# I've also given some inferences from using text2PCA on some bird captions. As you might've noticed, there are some recurring patterns that we can use to our advantage
