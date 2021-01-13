import argparse

import tensorflow as tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        type=str,
                        default="checkpoints/weights.best.hdf5",
                        help="Input keras hdf5 model file.")
    parser.add_argument("--output",
                        type=str,
                        default="model.tflite",
                        help="Output tensorflow pb file.")
    args = parser.parse_args()
    converter = tf.lite.TFLiteConverter.from_keras_model_file(args.input)
    tflite_model = converter.convert()
    with open(args.output, "wd") as of:
        of.write(tflite_model)


if __name__ == "__main__":
    main()
