import sys,os
import numpy as np
import quicklens as ql
import tensorflow as tf
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import backend as K 
from tensorflow.python.keras import callbacks
from shutil import copyfile

import config
import utilities
import networks
import losses
import trainingdata

#run e.g.: python train.py ~/mlcmb/mlcmb/config_master.ini



def run_training(configpath):
    
    params = config.Parameters(configpath)
    
    #save config for this training to be sure
    copyfile(configpath, params.folder_path_run+"config_backup_train.ini") 

    datasetid = params.datasetid
    
    if params.wf_mode=="T":
        channels_in = 2 #2
        channels_out = 1

    if params.wf_mode=="QU":
        channels_in = 3 #2
        channels_out = 2    

    npad = params.npad 
    img_shape = (params.imgsizepix+2*npad, params.imgsizepix+2*npad, channels_in)
    batch_size = params.batch_size #1
    epochs = params.epochs #1000 #5

    num_train_examples = params.nsims_train #10000 #1000 #10000 #should/could math the TFrecord file. this defines how long an epoch is.
    num_valid_examples = params.nsims_valid #1000 #300 #1000
    nx = params.nx

    save_model_path = params.folder_path_run+'model.ckpt'

    print ("save as", save_model_path) 
    
    restore_model = os.path.exists(save_model_path+".index") #whether or not to restore a previous training and training from there

    ################### NETWORK
    
    wienernet = networks.WienerNet(params)
    inputs,outputs = getattr(wienernet,params.network)(img_shape,channels_out) 
    
    lossfunctions = losses.Lossfunctions(params)
    
    if params.loss_mode == "J2":
        #lossfunc = lossfunctions.mean_squared_error
        #lossfunc = lossfunctions.loss_wiener_J2
        #lossfunc = lossfunctions.loss_pixelMSE_ellfiltered
        lossfunc = lossfunctions.loss_pixelMSE_unfiltered
        lossfuncname = 'loss_pixelMSE_unfiltered' #'loss_pixelMSE_ellfiltered' 'loss_wiener_J2' #needed to load the model

    if params.loss_mode == "J3":
        #lossfunc = lossfunctions.loss_wiener_J3_wfweighting
        #lossfuncname = 'loss_wiener_J3_wfweighting'   
        lossfunc = lossfunctions.loss_wiener_J3
        lossfuncname = 'loss_wiener_J3'    
        #lossfunc = lossfunctions.realspace_loss_noisediag
        #lossfuncname = 'realspace_loss_noisediag'

    if params.loss_mode == "J4":  
        lossfunc = lossfunctions.loss_wiener_J4
        lossfuncname = 'loss_wiener_J4'  

        
    model = models.Model(inputs=[inputs], outputs=[outputs])
    #optim = optimizers.RMSprop(learning_rate=1e-3)
    if params.optimizer=='Adam':
        optim = optimizers.Adam(lr=params.learning_rate) #https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
    model.compile(optimizer=optim, loss=lossfunc) #, metrics=['mean_squared_error']
    model.summary()  
    
    #cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_loss' ,save_best_only=True, verbose=1)
    cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_loss', save_weights_only=True ,save_best_only=True, verbose=1)  
    callback_csv = callbacks.CSVLogger(params.folder_path_run+'training_history_log.csv', append=True)
    #https://stackoverflow.com/questions/50127527/how-to-save-training-history-on-every-epoch-in-keras
    
    #check whether or not we want to load a previous model
    if restore_model:
        print ("WE TRAIN FROM PREVIOUS CHECKPOINT.")
    else:
        print ("WE TRAIN FROM START.")        
    
    if restore_model:
        print ("loading weights from", save_model_path)
        model.load_weights(save_model_path)
    
    
    ################### DATA SET
    
    dataset_train_raw = tf.data.TFRecordDataset(params.datapath+"datasets/dataset_wf_train_"+str(datasetid)+".tfrecords")
    dataset_valid_raw = tf.data.TFRecordDataset(params.datapath+"datasets/dataset_wf_valid_"+str(datasetid)+".tfrecords")

    dataset_train_parsed = dataset_train_raw.map(lambda x: trainingdata.tfrecord_parse_function(x, npad,params), num_parallel_calls=8)
    dataset_valid_parsed = dataset_valid_raw.map(lambda x: trainingdata.tfrecord_parse_function(x, npad,params), num_parallel_calls=8)

    #https://stackoverflow.com/questions/53514495/what-does-batch-repeat-and-shuffle-do-with-tensorflow-dataset
    dataset_train_parsed = dataset_train_parsed.shuffle(buffer_size=100,reshuffle_each_iteration=True).repeat().batch(batch_size)
    dataset_valid_parsed = dataset_valid_parsed.repeat().batch(batch_size)

    # Create an iterator for the dataset and the above modifications.
    iterator_train = dataset_train_parsed.make_one_shot_iterator()
    iterator_valid = dataset_valid_parsed.make_one_shot_iterator()
    
    
    #################### TRAINING
    history = model.fit(iterator_train, 
                   steps_per_epoch=int(np.ceil(num_train_examples / float(batch_size))),
                   epochs=epochs,
                   validation_data=iterator_valid,
                   validation_steps=int(np.ceil(num_valid_examples / float(batch_size))),verbose=2,callbacks=[cp,callback_csv])
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    np.savez(params.folder_path_run+"loss",loss=loss,val_loss=val_loss)
    
    
    #tf.keras.backend.clear_session()
     

    
    
 
def main():
    configpath=sys.argv[1]
    run_training(configpath)

        
        
if __name__ == "__main__":
    main()       
        