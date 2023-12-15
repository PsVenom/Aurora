import numpy as np
import click
from src.models.predict_model import call_model_for_train
from src.data.make_dataset import *
from src.models.helper import plot_images
import os
import tensorflow as tf
import keras
@click.command()
@click.argument('input_filepath', type=click.Path(exists=True),  default = os.get_cwd()+'caption_dataframe.csv')
@click.argument('START_RES', type = click.INT, default = 4, help = 'START_RESOLUTION for base model, default is 4' )
@click.argument('TARGET_RES', type = click.INT, default = 128, help = 'TARGET_RESOLUTION for base, default is 128' )
@click.argument('STEPS', type = click.INT, default = 2000, help = 'STEPS for base model training, default is 2000' )
@click.argument('WEIGHTS', type = click.STRING, default = None, help = 'WEIGHTS for base model training, default is None' )
def train(input_filepath,start_res, target_res, steps, weights):
    cap_vector, img_vectors = make_dataset(input_filepath)
    tf.config.run_functions_eagerly(True)
    style_gan =  call_model_for_train(START_RES=4, TARGET_RES=64, BETA=0.99, gen_per_epoch=1,weight_dir=weights)
    train_seq(start_res=32, target_res=64, display_images=False, style_gan=style_gan)
def train_seq(
        start_res=4,
        target_res=128,
        steps_per_epoch=2000,
        display_images=False,
        style_gan = call_model_for_train(START_RES=4, TARGET_RES=64, BETA=0.99, gen_per_epoch=1,weight_dir=None)
):
    opt_cfg = {"learning_rate": 1e-3, "beta_1": 0.0, "beta_2": 0.99, "epsilon": 1e-8}
    val_batch_size = 16
    val_z = tf.random.normal((val_batch_size, style_gan.z_dim))
    val_noise = style_gan.generate_noise(val_batch_size)

    start_res_log2 = int(np.log2(start_res))
    target_res_log2 = int(np.log2(target_res))

    for res_log2 in range(start_res_log2, target_res_log2 + 1):
        res = 2 ** res_log2
        for phase in ["TRANSITION", "STABLE"]:
            if res == start_res and phase == "TRANSITION":
                continue

            train_dl, caption_dl = create_dataloader(res)
            dataset = tf.data.Dataset.zip((train_dl, caption_dl))

            steps = int(train_step_ratio[res_log2] * steps_per_epoch) if phase == 'TRANSITION' else int(train_step_ratio[res_log2] * steps_per_epoch)/2

            style_gan.compile(
                d_optimizer=tf.keras.optimizers.Adam(**opt_cfg),
                g_optimizer=tf.keras.optimizers.Adam(**opt_cfg),
                loss_weights={"gradient_penalty": 10, "drift": 0.001},
                steps_per_epoch=steps,
                res=res,
                phase=phase,
                run_eagerly=True,
            )

            prefix = f"res_{res}x{res}_{style_gan.phase}"

            ckpt_cb = keras.callbacks.ModelCheckpoint(
                f"checkpoints/stylegan_{res}x{res}.ckpt",
                save_weights_only=True,
                verbose=0,
            )
            print(phase)
            style_gan.fit(
                dataset, epochs=1, steps_per_epoch=steps, callbacks=[ckpt_cb]
            )

            if display_images:
                images = style_gan({ "noise": val_noise, "alpha": 1.0, "v": "A flower"})
                plot_images(images, res_log2)
if __name__ == "__main__":
    train()