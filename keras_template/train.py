import sys
import os
import time
import argparse
import pickle
import logging
import numpy as np
from tensorflow.contrib.keras.python.keras.preprocessing import image
from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.contrib.keras.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.contrib.keras.python.keras.models import load_model
from tensorflow.contrib.keras.python.keras.callbacks import ModelCheckpoint
from tensorflow.contrib.keras.python.keras.callbacks import EarlyStopping, TensorBoard

MODEL_SAVE_FILE = 'save.h5'
MODEL_CKPT_FILE = 'ckpt.h5'

CLASS_INDEX_FILE = 'output_indices.pkl'

IMG_SRC_SIZE = (500, 500)
IMG_TGT_SIZE = (32, 32)

IMG_COLORMODE = 'rgb' # rgb, grayscale

BATCH_SIZE = {
    'train' : 32,
    'test'  : 1,
    'infer' : 1,
}

EPOCHS = {
    'train' : 50,
    'test'  : 1,
    'infer' : 1,
}

def find_num_samples(data_dir):
    """find the number of samples in the directory """
    path, dirs, files = os.walk(data_dir).next()
    assert path == data_dir
    samples =[x for x in files if x.endswith('.jpg')]
    numsample = len(samples)
    for subdir in dirs:
        numsample += find_num_samples(data_dir + '/' + subdir)
    return numsample

def build_train_gen(train_data_dir):
    train_datagen = ImageDataGenerator(
            featurewise_center = False,
            samplewise_center = False,
            featurewise_std_normalization = False,
            samplewise_std_normalization = False,
            zca_whitening = False,
            rescale = 1./ 255,
            shear_range = 0.0,
            zoom_range = 0.0,
            width_shift_range = 0.1,
            height_shift_range = 0.1,
            horizontal_flip = False,
            vertical_flip = False)

    traingen = train_datagen.flow_from_directory(
            train_data_dir,
            target_size = IMG_TGT_SIZE,
            batch_size = BATCH_SIZE['train'],
            color_mode = IMG_COLORMODE,
            class_mode = "categorical")
    # dump class index into serial file so that test can read it 
    pickle.dump(traingen.class_indices, open(CLASS_INDEX_FILE, 'wb'))
    return traingen

def build_test_gen(test_data_dir):
    test_datagen = ImageDataGenerator(
            featurewise_center = False,
            samplewise_center = False,
            featurewise_std_normalization = False,
            samplewise_std_normalization = False,
            zca_whitening = False,
            rescale = 1./ 255,
            shear_range = 0.0,
            zoom_range = 0.0, 
            width_shift_range = 0.1,
            height_shift_range = 0.1,
            horizontal_flip = False,
            vertical_flip = False)

    testgen = test_datagen.flow_from_directory(
            test_data_dir,
            target_size = IMG_TGT_SIZE,
            color_mode = IMG_COLORMODE,
            class_mode = "categorical")
    saved_indices = pickle.load(open(CLASS_INDEX_FILE, 'rb'))
    assert saved_indices == testgen.class_indices
    return testgen

def build_model(num_output_classes):
    conv1size = 32
    conv2size = 64
    convfiltsize = 4
    densesize = 128
    poolsize = (2, 2)
    imgdepth = 3
    dropout = 0.3
    if IMG_COLORMODE == 'grayscale':
        imgdepth = 1
    inpshape = IMG_TGT_SIZE + (imgdepth,)
    inputs = Input(shape=inpshape)
    conv1 = Convolution2D(conv1size, convfiltsize,
                        strides=(1,1),
                        padding='valid',
                        activation='relu',
                        name='conv1',
                        data_format='channels_last')(inputs)
    pool1 = MaxPooling2D(pool_size=poolsize, name='pool1')(conv1)
    drop1 = Dropout(dropout)(pool1)
    conv2 = Convolution2D(conv2size, convfiltsize,
                        strides=(1,1),
                        padding='valid',
                        activation='relu',
                        name='conv2',
                        data_format='channels_last')(drop1)
    pool2 = MaxPooling2D(pool_size=poolsize, name='pool2')(conv2)
    drop2 = Dropout(dropout)(pool2)
    flat2 = Flatten()(drop2)
    dense = Dense(densesize, name='dense')(flat2)
    denseact = Activation('relu')(dense)
    output = Dense(num_output_classes, name='output')(denseact)
    outputact = Activation('softmax')(output)

    model = Model(inputs=inputs, outputs=outputact)
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    return model


def train_mode(datadir):
    traindatadir = datadir + '/train/'
    traingen = build_train_gen(traindatadir)
    num_output_classes = len(traingen.class_indices)
    model = build_model(num_output_classes)
    numsamples = find_num_samples(traindatadir)

    checkpointer = ModelCheckpoint(filepath=MODEL_CKPT_FILE, verbose=1, save_best_only=True)
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
    #tfboard = TensorBoard(log_dir='./tfboard', histogram_freq=1, 
    #        write_graph=True, write_images=True)
    logging.info("starting fit_generator with num samples: %d, epochs: %d", numsamples, EPOCHS['train'])
    num_steps = numsamples / BATCH_SIZE['train']
    hist = model.fit_generator(
            traingen,
            num_steps,
            EPOCHS['train'],
            validation_data = traingen,
            validation_steps = num_steps,
            #callbacks=[checkpointer, earlystop, tfboard])
            callbacks=[checkpointer, earlystop])
    model.save(MODEL_SAVE_FILE)
    logging.info("Done training model, saved to file %s", MODEL_SAVE_FILE)
    logging.info("Training history: epoch, val_loss, val_accuracy")
    for epoch in range(len(hist.history['val_acc'])):
        logging.info('%d, %f, %f', epoch,
                hist.history['val_loss'][epoch],
                hist.history['val_acc'][epoch]
                )

def test_mode(datadir):
    testdatadir = datadir + '/test/'
    testgen = build_test_gen(testdatadir)
    numsamples = find_num_samples(testdatadir)
    model = False
    if os.path.isfile(MODEL_SAVE_FILE):
        model = load_model(MODEL_SAVE_FILE)
    elif os.path.isfile(MODEL_CKPT_FILE):
        model = load_model(MODEL_CKPT_FILE)
    if model:
        scores = model.evaluate_generator(testgen, numsamples)
        logging.info('Accuracy on test data is: %g', scores[1])
    else:
        logging.error("Error: cannot load tranied model")


def infer_mode(data):
    class_indices = pickle.load(open(CLASS_INDEX_FILE, 'rb'))
    img = image.load_img(data, target_size=IMG_TGT_SIZE)
    inp_data = image.img_to_array(img)
    inp_data = np.expand_dims(inp_data, axis=0)
    model = False
    result = -1
    if os.path.isfile(MODEL_SAVE_FILE):
        model = load_model(MODEL_SAVE_FILE)
    elif os.path.isfile(MODEL_CKPT_FILE):
        model = load_model(MODEL_CKPT_FILE)

    if model:
        predictions = model.predict(inp_data, verbose=1)
        prediction = predictions[0]
        logging.info("Prediction: %s", prediction)
        for classname in class_indices:
            logging.info("Prediction that sample is an '%s': %g",
                    classname, prediction[class_indices[classname]])
        result = prediction
    else:
        logging.error("Error: cannot load tranied model")

    return result

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s| %(message)s')
    parser = argparse.ArgumentParser(description=__doc__, 
                                formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("phase",
            help="Phase to perform: train, test, or infer")
    parser.add_argument("data",
            help="The data to read.")
            
    args = parser.parse_args()
    funcs = {
        'train' : train_mode,
        'test'  : test_mode,
        'infer' : infer_mode,
        }
    start_time = time.time()
    result = funcs[args.phase](args.data)
    elapsed_time = time.time() - start_time
    logging.info("total runtime in %f second", elapsed_time)
    return result


if __name__ == '__main__':
    sys.exit(main())


