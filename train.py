import click
from src.models.train_model import train_seq
import tensorflow as tf
from src.models.predict_model import call_model_for_train
from src.data.make_dataset import make_dataset
import os
@click.command()
@click.argument('input_filepath', type=click.Path(exists=True),  default = os.getcwd())
@click.option('-s', type = click.INT, default = 4, help = 'START_RESOLUTION for base model, default is 4' )
@click.option('-t', type = click.INT, default = 128, help = 'TARGET_RESOLUTION for base, default is 128' )
@click.option('-step',  type = click.INT, default = 2000, help = 'STEPS for base model training, default is 2000' )
@click.option('-w',  type = click.STRING, default = None, help = 'WEIGHTS for base model training, default is None' )
def train(input_filepath,s, t, step, w):
    cap_vector, img_vectors = make_dataset(input_filepath)
    tf.config.run_functions_eagerly(True)
    style_gan =  call_model_for_train(START_RES=s, TARGET_RES=t, BETA=0.99, gen_per_epoch=1,weight_dir=w)
    train_seq(start_res=s, target_res=t, cv = cap_vector, iv = img_vectors, steps_per_epoch = step, style_gan=style_gan)

if __name__ == "__main__":
    train()