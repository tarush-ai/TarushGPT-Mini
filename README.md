# TarushGPT-Mini


# Model Dimensions

The model is 5.7M parameters, with:

Embeddings
vocab_size = 4096
d_model = 256

total_embeddings = 4096 * 256 = 1,048,576

Model
layers = 6
heads = 8 
d_head = 256/8 = 32

total_model = 6 * 12 * 256^2 = 4,718,592

total_parameters = total_embeddings + total_model = 5,767,168 parameters

By Chinchilla scaling laws, for general performance, the ratio of tokens to parameters should be 20-1, meaning we need 115,343,360 tokens.


Given that one token is about 4 bytes:

115,343,360 * 4 = 461,373,440 bytes (which translates to 461.37344 MB).