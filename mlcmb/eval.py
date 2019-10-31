import sys,os
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import models
from shutil import copyfile

import config
import utilities
import networks


#run e.g.: python eval.py ~/mlcmb/mlcmb/config_master.ini


 
def eval_training(configpath):
    
    params = config.Parameters(configpath)

    datasetid = params.datasetid    
    
    if params.wf_mode=="T":
        channels_in = 2 #2
        channels_out = 1
    if params.wf_mode=="QU":
        channels_in = 3 #2
        channels_out = 2  
        
    npad = params.npad
    img_shape = (params.imgsizepix+2*npad, params.imgsizepix+2*npad, channels_in)

    wienernet = networks.WienerNet(params)
    inputs,outputs = getattr(wienernet,params.network)(img_shape,channels_out) 
    
    model = models.Model(inputs=[inputs], outputs=[outputs])

    save_model_path = params.folder_path_run+'model.ckpt'
    print ("loading and evaluating model", save_model_path)
    model.load_weights(save_model_path)    
    
    data_test = np.load(params.datapath+"/datasets/dataset_wf_test_"+str(datasetid)+".npy")

    if params.wf_mode=="T":
        images_test = data_test[:,:,:,[1,2]]
    if params.wf_mode=="QU":
        images_test = data_test[:,:,:,[10,11,2]]    

    #pad the images
    images_test = utilities.periodic_padding(images_test,npad)

    #renormalize images before analysing
    images_test[...,0] *= params.map_rescale_factor
    if params.wf_mode=="QU": 
        images_test[...,1] *= params.map_rescale_factor

    #analyse images
    result = model.predict(images_test, batch_size=params.batch_size)

    #ell filter the maps
    result[:,:,:,0] = utilities.ell_filter_maps(result[:,:,:,0], params.nx, params.dx, params.lmax, lmin=2)
    #undo renormalisation
    result[:,:,:,0] = (result[:,:,:,0])/params.map_rescale_factor

    if params.wf_mode=="QU":    
        result[:,:,:,1] = utilities.ell_filter_maps(result[:,:,:,1], params.nx, params.dx, params.lmax, lmin=2)
        result[:,:,:,1] = (result[:,:,:,1])/params.map_rescale_factor    

    #save
    fname = params.folder_path_run+"dataset_wf_test_"+str(datasetid)+"_results.npy"
    print ("saving results to", fname)
    np.save(fname,result)   


    
    

def main():
    configpath=sys.argv[1]
    eval_training(configpath)
        
        
if __name__ == "__main__":
    main() 