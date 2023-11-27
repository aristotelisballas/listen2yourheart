def my_train_challenge_model_old(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding data files...')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    classes = ['Present', 'Unknown', 'Absent']
    num_classes = len(classes)

    loader = BaseLoader(data_folder)  # TODO fix this
    train_ds, val_ds = loader.create_loaders(32, model_folder + "/snapshots")

    ts_input: Input = Input((WSIZE, 1), None, "ts_input")

    # model: Model = BioResNet(ts_input, num_classes)
    # model: Model = get_model(layers_5sec(num_classes), ts_input)
    blocks_per_layer = [2, 2, 2, 2]
    outputs = resnet_builder(x=ts_input, blocks_per_layer=blocks_per_layer, num_classes=num_classes)
    model: Model = keras.Model(ts_input, outputs)

    # Optimizer
    optimizer = SGDW(learning_rate=1e-1, weight_decay=1e-4, momentum=0.9)
    metrics = [
        CategoricalAccuracy(name='Accuracy'),
        Recall(name='Recall'),
        Precision(name='Precision'),
        AUC(name='AUC', multi_label=True)
    ]
    # AUC(num_thresholds=200, curve="ROC", summation_method="interpolation", name="AUC", dtype=None, thresholds=None, multi_label=False, label_weights=None)

    model.compile(optimizer, CategoricalCrossentropy(), metrics)
    model.summary(120)

    # Callbacks
    # 1. Reduce Learning Rate on Plateau
    # def scheduler(epoch, lr):
    #     if epoch < 42:
    #         return lr
    #     else:
    #         return lr * tf.math.exp(-0.1)

    # reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

    # 2. Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)

    # 3. Checkpoint
    checkpoint_path = Path(model_folder) / 'checkpoint' / 'checkpoint.cpt'
    model_checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, save_freq='epoch')

    # 4. Tensorboard
    tensor_board_log_dir: str = str(Path(model_folder) / 'tensorboard')
    tensor_board = TensorBoard(tensor_board_log_dir, 1, update_freq='batch')
    callbacks = [reduce_lr, early_stopping, model_checkpoint, tensor_board]

    epochs = 50

    # Train the model.
    if verbose >= 1:
        print('Training model...')
    model.fit(train_ds,
              epochs=epochs,
              steps_per_epoch=loader.train_steps,
              validation_data=val_ds,
              validation_steps=loader.val_steps,
              validation_freq=1,
              callbacks=callbacks)

    # Save the model.
    my_save_challenge_model(model_folder, model)

    if verbose >= 1:
        print('Done.')
