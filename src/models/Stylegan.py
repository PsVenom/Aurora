import tensorflow as tf
from src.data.make_dataset import *
from src.models.helper import *
import torch
import matplotlib.pyplot as plt
from clip import clip
class StyleGAN(tf.keras.Model):
    def __init__(self, z_dim=512, target_res=64, start_res=4, beta=1, gen_per_epoch = 1, weight = 10):
        super(StyleGAN, self).__init__()
        self.z_dim = z_dim
        self.beta_val = beta #DECOMPOSITION CONSTANT INITIALISER
        self.target_res_log2 = log2(target_res)
        self.start_res_log2 = log2(start_res)
        self.current_res_log2 = self.target_res_log2
        self.num_stages = self.target_res_log2 - self.start_res_log2 + 1
        self.alpha = tf.Variable(1.0, dtype=tf.float32, trainable=False, name="alpha")
        self.weight = weight
        self.beta = tf.Variable(beta, dtype=tf.float32, trainable=False, name="beta")
        self.a = 3.166186e20
        self.x = -8.6349
        self.gen_per_epoch = gen_per_epoch
        self.mapping = Mapping(num_stages=self.num_stages)
        self.d_builder = Discriminator(self.start_res_log2, self.target_res_log2)
        self.g_builder = Generator(self.start_res_log2, self.target_res_log2)
        self.g_input_shape = self.g_builder.input_shape
        self.map_optim = torch.optim.Adam(clip2sg.parameters(), lr=0.001)
        self.phase = None
        self.train_step_counter = tf.Variable(0, dtype=tf.int32, trainable=False)

        self.loss_weights = {"gradient_penalty": 10, "drift": 0.001}

    def grow_model(self, res):
        tf.keras.backend.clear_session()
        res_log2 = log2(res)
        self.generator = self.g_builder.grow(res_log2)
        self.discriminator = self.d_builder.grow(res_log2)
        self.current_res_log2 = res_log2
        print(f"\nModel resolution:{res}x{res}")

    def compile(
            self, steps_per_epoch, phase, res, d_optimizer, g_optimizer, *args, **kwargs
    ):
        self.loss_weights = kwargs.pop("loss_weights", self.loss_weights)
        self.steps_per_epoch = steps_per_epoch
        if res != 2 ** self.current_res_log2:
            self.grow_model(res)
            self.d_optimizer = d_optimizer
            self.g_optimizer = g_optimizer

        self.train_step_counter.assign(0)
        self.phase = phase
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")
        super(StyleGAN, self).compile(*args, **kwargs)

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def generate_noise(self, batch_size):
        noise = [
            tf.random.normal((batch_size, 2 ** res, 2 ** res, 1))
            for res in range(self.start_res_log2, self.target_res_log2 + 1)
        ]
        return noise

    def gradient_loss(self, grad):
        loss = tf.square(grad)
        loss = tf.reduce_sum(loss, axis=tf.range(1, tf.size(tf.shape(loss))))
        loss = tf.sqrt(loss)
        loss = tf.reduce_mean(tf.square(loss - 1))
        return loss

    def img_to_numpy(img):
        pass

    #THE ACTUAL TRAINING STEP
    def train_step(self, data):

        real_images, real_text = data
        #print("real_text ", real_text)
        #print(f"real_text shape: {real_text.shape}")

        real_text_string = tf.compat.as_str_any(real_text)

        self.train_step_counter.assign_add(1)

        if self.phase == "TRANSITION":
            self.alpha.assign(
                tf.cast(self.train_step_counter / self.steps_per_epoch, tf.float32)
            )
            self.beta.assign(self.beta)

        elif self.phase == "STABLE":
            self.alpha.assign(tf.cast(1.0,tf.float32))
            self.beta.assign(
                self.weight* tf.cast((tf.cast(self.train_step_counter,tf.float32))/(2*tf.cast(self.steps_per_epoch,tf.float32)),tf.float32)
            )


        elif self.phase == 'TRANSITION2':
            self.alpha.assign(1.0)
            self.beta.assign(
                self.weight* tf.cast(float(self.a)*tf.math.exp((-3)*float(self.x)*(self.train_step_counter+1.5*self.steps_per_epoch)/ self.steps_per_epoch),tf.float32)
            )
        else:
            raise NotImplementedError

        alpha = tf.expand_dims(self.alpha, 0)
        batch_size = tf.shape(real_images)[0]
        real_labels = tf.ones(batch_size)
        fake_labels = -tf.ones(batch_size)

        # z = tf.random.normal((batch_size, self.z_dim))
        const_input = tf.ones(tuple([batch_size] + list(self.g_input_shape)))
        noise = self.generate_noise(batch_size)


        # generator
        for i in range(self.gen_per_epoch):
            with tf.GradientTape() as g_tape:
                #real text string is actually a list of strings
                v = clip2sg(clip.tokenize(real_text_string.split("'")[1::2]).type(torch.float32).to(device))
                #now you just need to convert it to a pytorch tensor
                v = tf.expand_dims(v.cpu().data.numpy(), axis=0)
                #print(f"v: shape in train step {v.shape}")
                # v = self.mapping(v)
                #print(f"w: shape in train step {w.shape}")

                fake_images = self.generator([const_input, noise, alpha, v])


                if self.train_step_counter.numpy() % 500 == 0:
                    plt.imshow(fake_images[1])
                    plt.figtext(0.5, 0.01, real_text_string.split("'")[1::2][1], wrap=True, horizontalalignment='center', fontsize=12)
                    plt.show()
                print("Generated Images with input shape ", fake_images.shape, type(v))
                pred_fake = self.discriminator([fake_images, alpha])
                g_loss = wasserstein_loss(real_labels, pred_fake)

                trainable_weights = (
                        self.mapping.trainable_weights + self.generator.trainable_weights
                )
                if self.current_res_log2 >=3:
                    l_clip = L_clip(real_text_string,tf.Variable(fake_images * 255)) #.astype(np.uint8)
                    # self.map_optim.zero_grad()
                    # torch.from_numpy(np.array(l_clip.numpy())).backward()
                    # optimiser.numpy().step()
                    g_loss = g_loss + (1-self.beta)*tf.cast(l_clip,tf.float32)
                gradients = g_tape.gradient(g_loss, trainable_weights)
                self.g_optimizer.apply_gradients(zip(gradients, trainable_weights))

        # discriminator
        with tf.GradientTape() as gradient_tape, tf.GradientTape() as total_tape:
            # forward pass
            pred_fake = self.discriminator([fake_images, alpha])
            #print("pred_fake ", pred_fake)
            pred_real = self.discriminator([real_images, alpha])
            #print("pred_real", pred_real)

            epsilon = tf.random.uniform((batch_size, 1, 1, 1))
            interpolates = epsilon * real_images + (1 - epsilon) * fake_images
            gradient_tape.watch(interpolates)
            pred_fake_grad = self.discriminator([interpolates, alpha])

            # calculate losses
            loss_fake = wasserstein_loss(fake_labels, pred_fake)
            #print("loss fake: ", loss_fake)
            loss_real = wasserstein_loss(real_labels, pred_real)
            #print("loss_real: ", loss_real)
            loss_fake_grad = wasserstein_loss(fake_labels, pred_fake_grad)
            #print("real_text_string: ", real_text_string)
            #print("fake images shape", tf.Variable(fake_images * 255).shape)
            #print("Code works before clip loss")
            #print("L_clip success")

            # gradient penalty
            gradients_fake = gradient_tape.gradient(loss_fake_grad, [interpolates])
            gradient_penalty = self.loss_weights["gradient_penalty"] * self.gradient_loss(gradients_fake)

            # drift loss
            all_pred = tf.concat([pred_fake, pred_real], axis=0)
            drift_loss = self.loss_weights["drift"] * tf.reduce_mean(all_pred ** 2)

            d_loss = loss_fake + loss_real + gradient_penalty + drift_loss
            if self.current_res_log2 >=3:
                # l_clip = L_clip(real_text_string,tf.Variable(fake_images * 255)) #.astype(np.uint8)
                # self.map_optim.zero_grad()
                # torch.from_numpy(np.array(l_clip.numpy())).backward()
                # optimiser.numpy().step()
                d_loss+=(1-self.beta)*tf.cast(l_clip,tf.float32)
            LOSS.append(d_loss)
            #print(d_loss)

            gradients = total_tape.gradient(
                d_loss, self.discriminator.trainable_weights
            )
            self.d_optimizer.apply_gradients(
                zip(gradients, self.discriminator.trainable_weights)
            )

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
            "d_loss_stylegan": loss_fake + loss_real + gradient_penalty + drift_loss,
            "g_loss_clip": 0 if self.current_res_log2<3 else l_clip,
            "beta": self.beta
        }

    def call(self, inputs: dict()):
        style_code = inputs.get("style_code", None)
        v = inputs.get("v", None)
        batch_size = len(v)
        v = clip2sg(clip.tokenize(v).type(torch.float32).to(device))
        v = v = tf.expand_dims(v.cpu().data.numpy(), axis=0)
        print("v.shape", v.shape)
        # z = inputs.get("z", None)
        noise = inputs.get("noise", None)
        alpha = inputs.get("alpha", 1.0)
        alpha = tf.expand_dims(alpha, 0)
        if noise is None:
            noise = self.generate_noise(batch_size)
        # v= self.mapping(v)
        #self.alpha.assign(alpha)
        #print("Code works till here")
        const_input = tf.ones(tuple([batch_size] + list(self.g_input_shape)))
        print("const_input",const_input.shape)
        images = self.generator([const_input, noise, alpha, v])
        images = tf.keras.backend.clip((images * 0.5 + 0.5) * 255, 0, 255)

        return images