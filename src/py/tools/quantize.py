# Copyright 2020 Xiaomi
# Usage: python3 har/quantize.py --dir=dataset/swimming/sensor-data/ --model-file=path/to/model.hdf5 --int8=true --out-dir=/path/to/save/output/models
import itertools
import random
import logging
import numpy as np
import argparse
import os
import csv

import matplotlib.pyplot as plt
import tabulate
from common import *
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

try:
    # %tensorflow_version only exists in Colab.
    import tensorflow.compat.v2 as tf
except Exception:
    pass
tf.enable_v2_behavior()

from tensorflow import keras
import pathlib


def random_pick(some_list, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluating DNN Models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-file',
                        type=str,
                        help='model file path.',
                        required=True)
    parser.add_argument("--dir",
                        required=True,
                        dest='dir',
                        help="Dataset directory")
    parser.add_argument("--out-dir",
                        required=True,
                        dest='out_dir',
                        help="Dataset directory")
    parser.add_argument('--int8',
                        type=str2bool,
                        help="Use full int8 oor danymic range quantization.",
                        dest='int8',
                        default=True)

    args = parser.parse_args()
    return args


def quantize(model_file, out_dir, data_dir, use_int8=False):
    print("Loading model...")
    model = keras.models.load_model(model_file, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    model_filename = model_file.split('/')[-1]
    # write it out to a tflite file
    tflite_models_dir = pathlib.Path(out_dir + "/tflite_models/")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)

    tflite_model_filename = model_filename.replace('.hdf5', '.tflite')
    tflite_model_file = tflite_models_dir / tflite_model_filename
    tflite_model_file.write_bytes(tflite_model)

    # converter.post_training_quantize = True
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    x_test, y_test = load_dataset(data_dir,
                                  downsample=2,
                                  duration=8,
                                  shift=2,
                                  use_amp=True)
    frames = tf.cast(x_test, tf.float32)
    logging.debug("sensor data shape:", frames.shape)
    frames_ds = tf.data.Dataset.from_tensor_slices((frames)).batch(1)
    logging.debug("sensor batch dataset:", frames_ds)

    def representative_data_gen():
        for input_value in frames_ds.take(10000):
            yield [input_value]

    tflite_model_quant_filename = model_filename.replace(
        '.hdf5', '_quant.tflite')
    if use_int8:
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model_quant_filename = model_filename.replace(
            '.hdf5', '_quant_int8.tflite')

    tflite_model_quant = converter.convert()
    tflite_model_quant_file = tflite_models_dir / tflite_model_quant_filename
    tflite_model_quant_file.write_bytes(tflite_model_quant)

    return tflite_model_file, tflite_model_quant_file


def evaluate_model(interpreter, x_test, y_test):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    # Run predictions on every image in the "test" dataset.
    prediction_classes = []
    logging.debug("start process...")
    for test_data in x_test:
        # Pre-processing: add batch dimension and convert to float32 to match with
        # the model's input data format.
        test_data = np.expand_dims(test_data, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, test_data)

        # Run inference.
        interpreter.invoke()

        # Post-processing: remove batch dimension and find the digit with highest
        # probability.
        probs = interpreter.tensor(output_index)
        predict = np.argmax(probs()[0])
        prediction_classes.append(predict)
    logging.debug("start statistic.")
    # Compare prediction results with ground truth labels to calculate accuracy.
    accurate_count = 0
    for index in range(len(prediction_classes)):
        if prediction_classes[index] == y_test[index]:
            accurate_count += 1
    accuracy = accurate_count * 1.0 / len(prediction_classes)

    acc, conf_mat = stats_confusion_matrix(y_test, prediction_classes)
    print("Accuracy: ", acc)
    print(
        classification_report(y_test,
                              prediction_classes,
                              target_names=CategoryNames))
    tabulate_conf_mat(conf_mat)
    return accuracy


def eval(tflite_model_file, tflite_quant_model_file, eval_dir):
    logging.debug('Loading data...')
    x_test, y_test = load_dataset(eval_dir,
                                  downsample=2,
                                  duration=8,
                                  shift=2,
                                  use_amp=True)
    logging.debug('predicting...')

    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
    interpreter.allocate_tensors()

    interpreter_quant = tf.lite.Interpreter(
        model_path=str(tflite_quant_model_file))
    interpreter_quant.allocate_tensors()

    logging.info("####### Evaluate TFLite model ###########")
    evaluate_model(interpreter, x_test, y_test)
    logging.info("####### Evaluate TFLite Quant Model ############")
    evaluate_model(interpreter_quant, x_test, y_test)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        level=logging.INFO)
    args = parse_args()
    if os.path.isfile(args.model_file) and os.path.isdir(args.dir):
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
        tflite_model_file, tflite_quant_model_file = quantize(
            args.model_file, args.out_dir, args.dir, use_int8=args.int8)
        eval(tflite_model_file, tflite_quant_model_file, args.dir)
    else:
        raise Exception(
            "invalid invalid model file: {0} or invalid dataset dir {1}.".
            format(args.model_file, args.dir))
