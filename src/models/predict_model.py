import os
from Stylegan import StyleGAN
def call_model_for_train(START_RES = 4, TARGET_RES = 128, BETA = 0.99, gen_per_epoch = 1, weight_dir = None):
    style_gan = StyleGAN(start_res=START_RES, target_res=TARGET_RES, beta = BETA, gen_per_epoch = gen_per_epoch)
    if weight_dir:
        style_gan.load_weights(os.path.join(weight_dir))
    return style_gan
def call_model( weight_dir, WEIGHT_RES = 4, TARGET_RES = 128, BETA = 0.0001):
    style_gan = StyleGAN(start_res=WEIGHT_RES, target_res=TARGET_RES, beta = BETA, gen_per_epoch = 1)
    if not weight_dir:
        print("No weight directory")
        exit()
    style_gan.load_weights(os.path.join(weight_dir))