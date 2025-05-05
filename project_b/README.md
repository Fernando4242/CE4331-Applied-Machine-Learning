### **Overview**

This project focuses on training, fine-tuning, pruning, and quantizing a deep learning model for deployment on an **Espressif ESP32** device. The process uses Edge Impulse for model creation, followed by optimization with pruning and quantization, and finally deployment or creation of the C++ Library.

### **1. Model Creation and Training**

The model was created using Edge Impulse, with the dataset split using the data_splitter to organize the data into the appropriate categories for training. This split was then uploaded via the Edge Impulse UI for model creation. The base model used MobileNetV2 weights (`mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_160`), and transfer learning was applied to customize it for the dataset by using the python locally option in the Edge Impulse step. This allowed the use of the docker container for further customization during training.

The dataset can be found and managed within the data_splitter script, which splits the data into training and validation sets for model training that were uploaded into edge impulse to received the transformed into an easy to use for edge impulse format.

```python
train_dataset, validation_dataset, samples_dataset, X_train, X_test, Y_train, Y_test, \
    has_samples, X_samples, Y_samples = ei_tensorflow.training.get_dataset_from_folder(
        input,
        args.data_directory,
        RANDOM_SEED,
        online_dsp_config,
        MODEL_INPUT_SHAPE,
        args.ensure_determinism
    )
```

**Model Definition:**

```python
model = Sequential()
model.add(InputLayer(input_shape=INPUT_SHAPE, name='x_input'))
# Don't include the base model's top layers
last_layer_index = -3
model.add(Model(inputs=base_model.inputs, outputs=base_model.layers[last_layer_index].output))
model.add(Reshape((-1, model.layers[-1].output.shape[3])))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(classes, activation='softmax'))
```

**Training Parameters:**

```python
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
```

### **2. Fine-Tuning the Model**

The model was fine-tuned with:

* **10 Epochs** of training
* **70%** of the layers fine-tuned

### **3. Pruning the Model**

To optimize the model further, **pruning** was applied to reduce weights and enhance efficiency during fine-tuning.

**Pruning Code:**

```python
end_step = np.ceil(train_sample_count / BATCH_SIZE).astype(np.int32) * EPOCHS

pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.50,
        final_sparsity=0.80,
        begin_step=0,
        end_step=end_step
    )
}

model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_dataset,
    epochs=FINE_TUNE_EPOCHS,
    verbose=2,
    validation_data=validation_dataset,
    callbacks=callbacks,
    class_weight=None
)

model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
```

### **4. Quantization to INT8**

After pruning, the model was automatically quantized to **INT8** by the main script provided in the downloaded docker project from Edge Impulse when training is done, reducing model size and improving inference speed.

### **5. Deployment to ESP32**

The **INT8 quantized model** was saved as a **TFLite** file and packaged with a **C++ library** for deployment on the **ESP32**.

* **Deployment Output:** `ml-group-8-project-b-{version}.zip`

### **6. Edge Impulse Testing**

The model was deployed back to **Edge Impulse UI** for additional testing and result verification.

