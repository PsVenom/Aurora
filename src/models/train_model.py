import numpy as np
import click
from .predict_model import call_model_for_train
from src.data.make_dataset import create_dataloader
from .helper import plot_images
import os
import tensorflow as tf
import keras
batch_sizes = {2: 16, 3: 16, 4: 16, 5: 16, 6: 16, 7: 8, 8: 4, 9: 2, 10: 1}

train_step_ratio = {k: batch_sizes[2] / v for k, v in batch_sizes.items()}

def train_seq(
        cv, iv,
        start_res=4,
        target_res=128,

        steps_per_epoch=2000,
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

            train_dl, caption_dl = create_dataloader(res, iv, cv)
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


