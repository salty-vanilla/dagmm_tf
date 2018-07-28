import argparse
import os
import sys
sys.path.append(os.getcwd())
from dagmm import DAGMM
from image_sampler import ImageSampler
from models import EstimationNetwork, GMM
from mnist.autoencoder import AutoEncoder
from mnist.dataset import get_digits
from utils.config import args_to_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--height', '-ht', type=int, default=32)
    parser.add_argument('--width', '-wd', type=int, default=32)
    parser.add_argument('--channel', '-ch', type=int, default=1)
    parser.add_argument('--batch_size', '-bs', type=int, default=64)
    parser.add_argument('--nb_epoch', '-e', type=int, default=1000)
    parser.add_argument('--latent_dim', '-ld', type=int, default=2)
    parser.add_argument('--save_steps', '-ss', type=int, default=1)
    parser.add_argument('--logdir', '-log', type=str, default="../logs")
    parser.add_argument('--upsampling', '-up', type=str, default="deconv")
    parser.add_argument('--downsampling', '-down', type=str, default="stride")
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('--naef', '-naef', type=int, default=16)
    parser.add_argument('--distances', '-d', nargs='+', default=['mse'])
    parser.add_argument('--nb_components', '-c', type=int, default=4)

    args = parser.parse_args()

    args_to_csv(os.path.join(args.logdir, 'config.csv'), args)

    train_x = get_digits([0, 1], [4000, 10])
    image_sampler = ImageSampler(target_size=(args.width, args.height),
                                 color_mode='gray' if args.channel == 1 else 'rgb',
                                 is_training=True).flow(train_x, batch_size=args.batch_size)
    nb_features = args.latent_dim+len(args.distances)

    autoencoder = AutoEncoder((args.height, args.width, args.channel),
                              latent_dim=args.latent_dim,
                              first_filters=args.naef,
                              downsampling=args.downsampling,
                              upsampling=args.upsampling,
                              distances=args.distances)
    estimator = EstimationNetwork((nb_features, ),
                                  dense_units=[256, args.nb_components])
    gmm = GMM(args.nb_components, nb_features)

    dagmm = DAGMM(autoencoder,
                  estimator,
                  gmm)

    dagmm.fit(image_sampler,
              nb_epoch=args.nb_epoch,
              save_steps=args.save_steps,
              logdir=args.logdir)


if __name__ == '__main__':
    main()