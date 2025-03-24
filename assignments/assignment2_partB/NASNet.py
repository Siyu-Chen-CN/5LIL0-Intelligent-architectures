import os
import tensorflow as tf
import tensorflow_datasets as tfds
import json
import csv
import numpy as np
from typing import List
from functools import partial

class NASNet:
    def __init__(
        self,
        epochs: int,
        lr: float, 
        exp_dir: str,
        objectives: List[str],
    ):
        self.epochs = epochs
        self.lr = lr
        self.objectives = objectives
        self.exp_dir = exp_dir

    def __call__(self, trial):
        # create work directory for storing trial artifacts
        self.work_dir = os.path.join(self.exp_dir, f"trial_{trial.number}")
        os.makedirs(self.work_dir, exist_ok=True)

        # get trial hyperparameters
        self.trial_hp = self._search_space_by_func(trial)
        with open(os.path.join(self.work_dir, "trial_hps.json"), "w") as outfile:
            json.dump(self.trial_hp, outfile, indent=1)

        # instantiate model
        model = self._get_CNN()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        # get MNIST dataset for training
        ds_train, ds_test = self._get_mnist_dataset()

        # train model
        model.fit(ds_train, epochs=self.epochs, validation_data=ds_test)
        
        # save model on trial directory
        model.save(os.path.join(self.work_dir, "mnist_trained_model.keras"))
        
        # calculate floating-point model metrics
        self.fp32_accuracy = model.history.history["val_accuracy"][-1]
        self.num_params = model.count_params()

        # quantization

        # reload model with batch size 1 before TFLite export for
        # compatibility with ethos vela compiler
        model = self._get_CNN(batch_size=1)
        model.load_weights(os.path.join(self.work_dir, "mnist_trained_model.keras"))

        # get dataset with batch size 1 for TFLite model evaluation
        ds_train_calib, ds_q_test = self._get_mnist_dataset(batch_size=1)
        tflite_model = self._quantize_model(ds_train_calib, model)
        tflite_model_path = os.path.join(self.work_dir, "mnist_nas_full_int8.tflite")
        open(tflite_model_path, "wb").write(tflite_model)
        
        # calculate quantized model metrics
        self.int8_accuracy = self._evaluate_quantized_model(tflite_model_path, ds_q_test)
        self.int8_model_size = os.path.getsize(tflite_model_path) / 1024
        self.latency = self._profile_model_latency(self.work_dir, tflite_model_path)
        
        # Log this to optuna for post-processing
        trial.set_user_attr("fp32_accuracy", self.fp32_accuracy)
        trial.set_user_attr("num_params", self.num_params)
        trial.set_user_attr("int8_accuracy", self.int8_accuracy)
        trial.set_user_attr("int8_model_size", self.int8_model_size)
        trial.set_user_attr("int8_latency", self.latency)

        # report metrics
        return [getattr(self, m) for m in self.objectives]

    def _normalize_img(self, image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label

    def _get_mnist_dataset(self, batch_size=256):
        """
        Returns MNIST dataset for training.
        Code borrowed from: https://www.tensorflow.org/datasets/keras_example
        """
        (ds_train, ds_test), ds_info = tfds.load('mnist', split=['train', 'test'], shuffle_files=True, as_supervised=True, with_info=True)
        ds_train = ds_train.map(self._normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        ds_train = ds_train.cache()
        ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
        ds_train = ds_train.batch(batch_size)
        ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
        ds_test = ds_test.map(self._normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = ds_test.batch(batch_size)
        ds_test = ds_test.cache()
        ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
        return ds_train, ds_test

    def _quantize_model(self, calib_ds, model):
        """Quantizes Keras model to TFLite using data generator"""
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = self._get_calibration_set(ds=calib_ds, num_samples=100)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # or tf.uint8
        converter.inference_output_type = tf.int8  # or tf.uint8
        tflite_model = converter.convert()
        return tflite_model
    
    def _evaluate_quantized_model(self, model_path, ds):
        model = tf.lite.Interpreter(model_path=model_path)
        model.allocate_tensors()
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        input_type = input_details[0]['dtype']

        iter_ds = iter(ds)
        n_correct_preds = 0
        n_total_preds = 0

        for inputs, targets in iter_ds:
            # forward pass
            if input_type == np.int8:
                input_scale, input_zero_point = input_details[0]['quantization']
                inputs = (inputs / input_scale) + input_zero_point
                inputs = np.around(inputs)

            # Convert features to NumPy array of expected type
            inputs = inputs.astype(input_type)
            model.set_tensor(input_details[0]['index'], inputs)
            model.invoke()
            output = model.get_tensor(output_details[0]['index'])

            output_type = output_details[0]['dtype']
            if output_type == np.int8:
                output_scale, output_zero_point = output_details[0]['quantization']
                output = output_scale * (output.astype(np.float32) - output_zero_point)
            # endof fwd
            predicted = np.argmax(output, axis=1)
            n_total_preds += targets.numpy().shape[0]
            n_correct_preds += np.equal(predicted, targets.numpy()).sum()
            #progress_bar.update(1) # update progress
        acc = n_correct_preds / n_total_preds
        return acc   
    
    def _profile_model_latency(self, work_dir, tflite_model_path):
        """Profiles TFLite latency (in ms) using the EthosU65 vela compiler."""
        os.system(
            f"vela --accelerator-config ethos-u55-32 {tflite_model_path} --output-dir {os.path.join(work_dir, 'vela_output')} --verbose-performance",
        )
        # load vela csv file
        csv_file_name = "mnist_nas_full_int8_summary_internal-default.csv"
        with open(os.path.join(work_dir, "vela_output", csv_file_name), 'r') as file:
            csv_reader = csv.DictReader(file)
            prof_results = [row for row in csv_reader][0]
        latency = float(prof_results["inference_time"]) * 1e3  # s --> ms
        return latency

    def _get_calibration_set(self, ds, num_samples=100):
        """Returns a calibration set generator for TFLite quantization"""
        def data_gen(num_samples):
            ds_iter = iter(ds)
            for i, batch in enumerate(ds_iter):
                if i == num_samples:
                    break
                x, _ = batch
                x = tf.constant(x)
                yield [x]
        return partial(data_gen, num_samples)