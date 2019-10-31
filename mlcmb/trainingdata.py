import numpy as np
import quicklens as ql
import tensorflow as tf
import sys
from shutil import copyfile

import config
import utilities

#run e.g.: python trainingdata.py ~/mlcmb/mlcmb/config_master.ini

##################### make data set
#creates tfrecords for training and validation, and a numpy array for test

def make_dataset(configpath):
    
    params = config.Parameters(configpath)

    datasetid = params.datasetid

    nsims_train = params.nsims_train
    nsims_valid = params.nsims_valid
    nsims_test = params.nsims_test
    nsims      = nsims_train+nsims_valid+nsims_test
    wf_trainvalid = False #whether or not also to wiener filter training and validation data


    multigrid  = True #False                                 # use a multigrid preconditioner.
    eps_min    = params.eps_min #1.e-6 #1.e-4
    lmax       = params.lmax                                 # maximum multipole.
    nside = params.nx
    nx         = params.nx
    dx         = params.dx

    nlev_t     = params.nlev_t                                 # 10. temperature map noise level, in uK.arcmin.
    nlev_p     = params.nlev_p                                  # 10. polarization map noise level (Q, U), in uK.arcmin.
    bl         = ql.spec.bl(fwhm_arcmin=params.fwhm_arcmin, lmax=lmax) # instrumental beam transfer function.

    cl_len     = ql.spec.get_camb_lensedcl(lmax=lmax)  # cmb theory spectra.

    #TEST: set bb and ee to same power
    #cl_len.clbb[:] = cl_len.clee[:]

    mask = params.mask

    rescale_factor =  1000. #1000. #internal rescaling. undone in the end. seems to be needed to get correct results in some configurations

    #RESCALE
    nlev_t *= rescale_factor
    nlev_p *= rescale_factor
    cl_len.clbb *= rescale_factor**2.
    cl_len.clee *= rescale_factor**2.
    cl_len.cltt *= rescale_factor**2.

    nltt       = (nlev_t*np.pi/180./60.)**2 / bl**2
    nlee       = (nlev_p*np.pi/180./60.)**2 / bl**2

    # diagonal approximation of filter.
    flt        = 1.0/(nltt[0:lmax+1] + cl_len.cltt[0:lmax+1]); flt[0:2] = 0.0
    fle        = 1.0/(nlee[0:lmax+1] + cl_len.clee[0:lmax+1]); fle[0:2] = 0.0
    flb        = 1.0/(nlee[0:lmax+1] + cl_len.clbb[0:lmax+1]); flb[0:2] = 0.0

    pix        = ql.maps.pix(nx, dx)

    c = ql.spec.clmat_teb(cl_len)
    
    
    #make a TFrecord for training set and valid set, but not for test set
    filename_train = params.datapath+"datasets/dataset_wf_train_"+str(datasetid)+".tfrecords"
    filename_valid = params.datapath+"datasets/dataset_wf_valid_"+str(datasetid)+".tfrecords"
    print('Writing', filename_train,filename_valid)

    def _bytes_feature_image(image):
        value = tf.compat.as_bytes(image.tostring())
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    dataset_test = np.zeros( (nsims_test,nside,nside,16) ) 

    with tf.python_io.TFRecordWriter(filename_train) as writer_train, tf.python_io.TFRecordWriter(filename_valid) as writer_valid:

        for map_id in range(nsims):
            print ("map", map_id)

            # simulate input sky
            teb_sky  = ql.sims.tebfft(pix, cl_len)
            tqu_sky  = teb_sky.get_tqu()

            # simulate observed sky
            teb_obs  = ql.spec.blmat_teb(bl) * teb_sky
            tqu_obs  = teb_obs.get_tqu() + ql.sims.tqumap_homog_noise( pix, nlev_t, nlev_p )

            tsky = tqu_sky.tmap/rescale_factor #T sky
            tobs = mask*tqu_obs.tmap/rescale_factor  #T obs
            esky = teb_sky.get_ffts()[1].get_rmap().map/rescale_factor #E sky
            bsky = teb_sky.get_ffts()[2].get_rmap().map/rescale_factor #B sky
            qsky = teb_sky.get_tqu().qmap/rescale_factor #Q sky
            usky = teb_sky.get_tqu().umap/rescale_factor #U sky
            eobs = (tqu_obs*mask).get_teb().get_ffts()[1].get_rmap().map/rescale_factor #E obs
            bobs = (tqu_obs*mask).get_teb().get_ffts()[2].get_rmap().map/rescale_factor #B obs
            qobs = mask*tqu_obs.qmap/rescale_factor #Q obs
            uobs = mask*tqu_obs.umap/rescale_factor #U obs           

            #wiener filter only the test data, or all
            if (map_id>=nsims_train+nsims_valid) or wf_trainvalid:
                # diagonal filter
                teb_filt_diag = ql.spec.blmat_teb(1./bl) * (tqu_obs * mask).get_teb()
                teb_filt_diag = ql.spec.clmat_teb( ql.util.dictobj( {'lmax' : lmax, 'cltt' : flt, 'clee' : fle, 'clbb' : flb} ) ) * teb_filt_diag


                # construct cinv filter
                ninv_filt = ql.cinv.opfilt_teb.ninv_filt(ql.spec.blmat_teb(bl),
                                                         ql.maps.tqumap(nx, dx,
                                                                        [ (180.*60./np.pi*dx)**2 / nlev_t**2 * np.ones( (nx,nx) ) * mask,
                                                                          (180.*60./np.pi*dx)**2 / nlev_p**2 * np.ones( (nx,nx) ) * mask,
                                                                          (180.*60./np.pi*dx)**2 / nlev_p**2 * np.ones( (nx,nx) ) * mask ]))
                sinv_filt = ql.cinv.opfilt_teb.sinv_filt(cl_len)

                pre_op = ql.cinv.opfilt_teb.pre_op_diag( sinv_filt, ninv_filt )


                # run solver
                if multigrid == True:
                    chain = ql.cinv.multigrid.chain( 0, ql.cinv.opfilt_teb, sinv_filt, ninv_filt, plogdepth=2, eps_min=eps_min )
                    teb_filt_cinv = chain.solve( ql.maps.tebfft( nx, dx ), tqu_obs )
                else:
                    monitor = ql.cinv.cd_monitors.monitor_basic(ql.cinv.opfilt_teb.dot_op(), iter_max=np.inf, eps_min=eps_min)

                    teb_filt_cinv = ql.maps.tebfft( nx, dx )
                    ql.cinv.cd_solve.cd_solve( x = teb_filt_cinv,
                                               b = ql.cinv.opfilt_teb.calc_prep(tqu_obs, sinv_filt, ninv_filt),
                                               fwd_op = ql.cinv.opfilt_teb.fwd_op(sinv_filt, ninv_filt),
                                               pre_ops = [pre_op], dot_op = ql.cinv.opfilt_teb.dot_op(),
                                               criterion = monitor, tr=ql.cinv.cd_solve.tr_cg, cache=ql.cinv.cd_solve.cache_mem() )
                    teb_filt_cinv = ql.cinv.opfilt_teb.calc_fini( teb_filt_cinv, sinv_filt, ninv_filt)

                twf = (c*teb_filt_cinv).get_ffts()[0].get_rmap().map/rescale_factor #T WF 
                ewf = (c*teb_filt_cinv).get_ffts()[1].get_rmap().map/rescale_factor #E WF
                bwf = (c*teb_filt_cinv).get_ffts()[2].get_rmap().map/rescale_factor #B WF 
                qwf = (c*teb_filt_cinv).get_tqu().qmap/rescale_factor #Q WF
                uwf = (c*teb_filt_cinv).get_tqu().umap/rescale_factor #U WF   

            #-----save record and test data

            #write records
            if wf_trainvalid:
                 example = tf.train.Example(
                  features=tf.train.Features(
                      feature={
                          'tsky': _bytes_feature_image(tsky),
                          'tobs': _bytes_feature_image(tobs),
                          'mask': _bytes_feature_image(mask),
                          'esky': _bytes_feature_image(esky),
                          'bsky': _bytes_feature_image(bsky),
                          'qsky': _bytes_feature_image(qsky),
                          'usky': _bytes_feature_image(usky),
                          'eobs': _bytes_feature_image(eobs),
                          'bobs': _bytes_feature_image(bobs), 
                          'qobs': _bytes_feature_image(qobs),                   
                          'uobs': _bytes_feature_image(uobs),
                          'twf': _bytes_feature_image(twf),
                          'ewf': _bytes_feature_image(ewf),
                          'bwf': _bytes_feature_image(bwf), 
                          'qwf': _bytes_feature_image(qwf),                   
                          'uwf': _bytes_feature_image(uwf)                      
                      }))           
            else:
                example = tf.train.Example(
                  features=tf.train.Features(
                      feature={
                          'tsky': _bytes_feature_image(tsky),
                          'tobs': _bytes_feature_image(tobs),
                          'mask': _bytes_feature_image(mask),
                          'esky': _bytes_feature_image(esky),
                          'bsky': _bytes_feature_image(bsky),
                          'qsky': _bytes_feature_image(qsky),
                          'usky': _bytes_feature_image(usky),
                          'eobs': _bytes_feature_image(eobs),
                          'bobs': _bytes_feature_image(bobs), 
                          'qobs': _bytes_feature_image(qobs),                   
                          'uobs': _bytes_feature_image(uobs)
                      }))
            if map_id<nsims_train:
                writer_train.write(example.SerializeToString())
            if map_id>=nsims_train and map_id<(nsims_train+nsims_valid):
                writer_valid.write(example.SerializeToString()) 

            #save test data
            if map_id>=nsims_train+nsims_valid:
                testmap_id=map_id-nsims_valid-nsims_train

                dataset_test[testmap_id,:,:,0] = tsky 
                dataset_test[testmap_id,:,:,1] = tobs
                dataset_test[testmap_id,:,:,2] = mask 
                dataset_test[testmap_id,:,:,4] = esky
                dataset_test[testmap_id,:,:,5] = bsky
                dataset_test[testmap_id,:,:,6] = qsky
                dataset_test[testmap_id,:,:,7] = usky
                dataset_test[testmap_id,:,:,8] = eobs
                dataset_test[testmap_id,:,:,9] = bobs
                dataset_test[testmap_id,:,:,10] = qobs
                dataset_test[testmap_id,:,:,11] = uobs                
                dataset_test[testmap_id,:,:,3] = twf         
                dataset_test[testmap_id,:,:,12] = ewf
                dataset_test[testmap_id,:,:,13] = bwf
                dataset_test[testmap_id,:,:,14] = qwf
                dataset_test[testmap_id,:,:,15] = uwf
    
    
    #save training set
    if nsims_test>0:
        np.save(params.datapath+"datasets/dataset_wf_test_"+str(datasetid)+".npy",dataset_test)

    #save config for this data set
    copyfile(configpath, params.datapath+"datasets/dataset_wf_config_backup_"+str(datasetid)+".ini") 
    
        
        
        
        
        
        
##################### TF record parser

#https://medium.com/@moritzkrger/speeding-up-keras-with-tfrecord-datasets-5464f9836c36
#https://www.tensorflow.org/guide/datasets#parsing_tfexample_protocol_buffer_messages
#https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/18_TFRecords_Dataset_API.ipynb
#Many input pipelines extract tf.train.Example protocol buffer messages from a TFRecord-format file (written, for example, using tf.python_io.TFRecordWriter). Each tf.train.Example record contains one or more "features", and the input pipeline typically converts these features into tensors.

def tfrecord_parse_function(proto,npad,params):
    if params.wf_mode == "T":
        # define your tfrecord again. Remember that you saved your image as a string.
        keys_to_features = {'tsky': tf.FixedLenFeature([], tf.string),
                            'tobs': tf.FixedLenFeature([], tf.string),
                            'mask': tf.FixedLenFeature([], tf.string)}

        # Load one example
        parsed_features = tf.parse_single_example(proto, keys_to_features)

        # Turn your saved image string into an array
        parsed_features['tsky'] = tf.decode_raw(parsed_features['tsky'], tf.float64)*params.map_rescale_factor
        parsed_features['tobs'] = tf.decode_raw(parsed_features['tobs'], tf.float64)*params.map_rescale_factor
        parsed_features['mask'] = tf.decode_raw(parsed_features['mask'], tf.float64)
        
        #reshape to original form
        parsed_features['tsky'] = tf.reshape(parsed_features['tsky'], [params.nx, params.nx, 1])
        parsed_features['tobs'] = tf.reshape(parsed_features['tobs'], [params.nx, params.nx, 1])
        parsed_features['mask'] = tf.reshape(parsed_features['mask'], [params.nx, params.nx, 1])
        
        #pad
        _padfunc = utilities.periodic_padding
        parsed_features['tobs_pad'] = tf.py_func(_padfunc, [parsed_features['tobs'],npad], tf.float64 )
        parsed_features['tsky_pad'] = tf.py_func(_padfunc, [parsed_features['tsky'],npad], tf.float64 )
        parsed_features['mask_pad'] = tf.py_func(_padfunc, [parsed_features['mask'],npad], tf.float64 )

        if params.loss_mode == 'J2':
            image = tf.concat([parsed_features['tobs_pad'],parsed_features['mask_pad']],axis=-1)
            label = parsed_features['tsky']
            return image, label
        
        if params.loss_mode == 'J3':
            image = tf.concat([parsed_features['tobs_pad'],parsed_features['mask_pad']],axis=-1)
            label = parsed_features['tobs']
            return image, label       
        
        if params.loss_mode == 'J4':
            image = tf.concat([parsed_features['tobs_pad'],parsed_features['mask_pad']],axis=-1)
            label = tf.concat([parsed_features['tobs'],parsed_features['tsky']],axis=-1)
            return image, label    

    if params.wf_mode == "QU":
        # define your tfrecord again. Remember that you saved your image as a string.
        keys_to_features = {'qsky': tf.FixedLenFeature([], tf.string),
                            'qobs': tf.FixedLenFeature([], tf.string),
                            'usky': tf.FixedLenFeature([], tf.string),
                            'uobs': tf.FixedLenFeature([], tf.string),
                            'mask': tf.FixedLenFeature([], tf.string)}

        # Load one example
        parsed_features = tf.parse_single_example(proto, keys_to_features)

        # Turn your saved image string into an array
        parsed_features['qsky'] = tf.decode_raw(parsed_features['qsky'], tf.float64)*params.map_rescale_factor
        parsed_features['qobs'] = tf.decode_raw(parsed_features['qobs'], tf.float64)*params.map_rescale_factor
        parsed_features['usky'] = tf.decode_raw(parsed_features['usky'], tf.float64)*params.map_rescale_factor
        parsed_features['uobs'] = tf.decode_raw(parsed_features['uobs'], tf.float64)*params.map_rescale_factor
        parsed_features['mask'] = tf.decode_raw(parsed_features['mask'], tf.float64)
        
        #reshape to original form
        parsed_features['qsky'] = tf.reshape(parsed_features['qsky'], [params.nx, params.nx, 1])
        parsed_features['qobs'] = tf.reshape(parsed_features['qobs'], [params.nx, params.nx, 1])
        parsed_features['usky'] = tf.reshape(parsed_features['usky'], [params.nx, params.nx, 1])
        parsed_features['uobs'] = tf.reshape(parsed_features['uobs'], [params.nx, params.nx, 1])
        parsed_features['mask'] = tf.reshape(parsed_features['mask'], [params.nx, params.nx, 1])
        
        #pad
        _padfunc = utilities.periodic_padding
        parsed_features['qobs_pad'] = tf.py_func(_padfunc, [parsed_features['qobs'],npad], tf.float64 )
        parsed_features['qsky_pad'] = tf.py_func(_padfunc, [parsed_features['qsky'],npad], tf.float64 )
        parsed_features['uobs_pad'] = tf.py_func(_padfunc, [parsed_features['uobs'],npad], tf.float64 )
        parsed_features['usky_pad'] = tf.py_func(_padfunc, [parsed_features['usky'],npad], tf.float64 )        
        parsed_features['mask_pad'] = tf.py_func(_padfunc, [parsed_features['mask'],npad], tf.float64 )

#         if params.loss_mode == 'J1':
#             image = tf.concat([parsed_features['qobs_pad'],parsed_features['uobs_pad'],parsed_features['mask_pad']],axis=-1)
#             label = tf.concat([parsed_features['qwf'],parsed_features['uwf']],axis=-1)
#             return image, label        
        
        if params.loss_mode == 'J2':
            image = tf.concat([parsed_features['qobs_pad'],parsed_features['uobs_pad'],parsed_features['mask_pad']],axis=-1)
            label = tf.concat([parsed_features['qsky'],parsed_features['usky']],axis=-1)
            return image, label
        
        if params.loss_mode == 'J3':
            image = tf.concat([parsed_features['qobs_pad'],parsed_features['uobs_pad'],parsed_features['mask_pad']],axis=-1)
            label = label = tf.concat([parsed_features['qobs'],parsed_features['uobs']],axis=-1)
            return image, label    
        
        if params.loss_mode == 'J4':
            image = tf.concat([parsed_features['qobs_pad'],parsed_features['uobs_pad'],parsed_features['mask_pad']],axis=-1)
            label = tf.concat([parsed_features['qobs'],parsed_features['uobs'],parsed_features['qsky'],parsed_features['usky']],axis=-1)
            return image, label           


        
        
        
        
        
        

##################### online feeder (not used so far since too slow)

class Quicklens_data_feeder():
    
    def __init__(self,npad,params):
        self.params = params
        self.pix = ql.maps.pix(self.params.nx, params.dx)
        self.cl_len = ql.spec.get_camb_lensedcl(lmax=params.lmax)  # cmb theory spectra.
        self.bl = ql.spec.bl(fwhm_arcmin=1., lmax=params.lmax) # instrumental beam transfer function.
        self.npad = npad
        self.mask = params.mask
        self.mask_pad = utilities.periodic_padding(self.mask,self.npad)

    def get_wiener_data(self):

        # simulate input sky
        teb_sky  = ql.sims.tebfft(self.pix, self.cl_len)
        tqu_sky  = teb_sky.get_tqu()

        # simulate observed sky
        teb_obs  = ql.spec.blmat_teb(self.bl) * teb_sky
        tqu_obs  = teb_obs.get_tqu() + ql.sims.tqumap_homog_noise( self.pix, params.nlev_t, params.nlev_p )

        if params.wf_mode=="T": 
            #images: t_obs, mask
            #labels: t_obs            
            images = np.zeros( (self.params.nx,self.params.nx,2), dtype=np.float32)
            labels = np.zeros( (self.params.nx,self.params.nx,1), dtype=np.float32)
            
            #TODO
        
        if params.wf_mode=="E":         
            images = np.zeros( (self.params.nx+self.npad,self.params.nx+self.npad,3), dtype=np.float32)
            labels = np.zeros( (self.params.nx+self.npad,self.params.nx+self.npad,2), dtype=np.float32)
            
            qobs = self.mask*tqu_obs.qmap
            uobs = self.mask*tqu_obs.umap
        
            #pad the images
            qobs = utilities.periodic_padding(qobs,self.npad)
            uobs = utilities.periodic_padding(uobs,self.npad) 
    
            #rescale the maps
            qobs *= params.map_rescale_factor_pol
            uobs *= params.map_rescale_factor_pol
        
            images[:,:,0] = qobs
            images[:,:,1] = uobs
            images[:,:,2] = self.mask_pad
            labels[:,:,0] = qobs
            labels[:,:,1] = uobs
                
        yield images,labels
        
        
        
        
        
        
 
def main():
    configpath=sys.argv[1]
    make_dataset(configpath)
        
        
if __name__ == "__main__":
    main()       
        
        