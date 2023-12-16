import click
from src.models.train_model import train_seq
import tensorflow as tf
from src.models.predict_model import call_model_for_train
from src.data.make_dataset import make_dataset
import os
@click.command()
@click.argument('input_filepath', type=click.Path(exists=True),  default = os.get_cwd())
@click.argument('START_RES', type = click.INT, default = 4, help = 'START_RESOLUTION for base model, default is 4' )
@click.argument('TARGET_RES', type = click.INT, default = 128, help = 'TARGET_RESOLUTION for base, default is 128' )
@click.argument('STEPS', type = click.INT, default = 2000, help = 'STEPS for base model training, default is 2000' )
@click.argument('WEIGHTS', type = click.STRING, default = None, help = 'WEIGHTS for base model training, default is None' )
@click.argument('SHOW_IMG', type = click.BOOL, default = False,help ='shows image for base model training')
def train(input_filepath,start_res, target_res, steps, weights, see_img = False):
    cap_vector, img_vectors = make_dataset(input_filepath)
    tf.config.run_functions_eagerly(True)
    style_gan =  call_model_for_train(START_RES=start_res, TARGET_RES=target_res, BETA=0.99, gen_per_epoch=1,weight_dir=weights)
    train_seq(start_res=start_res, target_res=target_res, display_images=False, steps_per_epocj = steps, style_gan=style_gan)

if __name__ == "__main__":
    train()