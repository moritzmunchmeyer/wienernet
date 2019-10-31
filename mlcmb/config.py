import scipy
import numpy as np
import os
import ConfigParser

r2d = 180./np.pi
d2r = np.pi/180.
rad2arcmin = 180.*60./np.pi

mlcmbpath = os.path.dirname(__file__)+"/../"


class Parameters():

    def __init__(self,configpath):

        Config = ConfigParser.ConfigParser()
        Config.read(configpath) 

        self.datapath = Config.get("Global", "datapath")  

        self.folder_path_run = self.datapath+Config.get("Global", "runpath_in_datapath") 
        if not os.path.exists(self.folder_path_run):
            os.makedirs(self.folder_path_run)

        self.datasetid = Config.getint("Global", "datasetid")     

        self.nlev_t = Config.getfloat("CMBexperiment", "nlev_t") # temperature map noise level, in uK.arcmin.
        self.nlev_p = Config.getfloat("CMBexperiment", "nlev_p") # polarization map noise level (Q, U), in uK.arcmin.
        self.fwhm_arcmin = Config.getfloat("CMBexperiment", "fwhm_arcmin") #1. 0.1 #beam

        self.lmax = Config.getint("Map", "lmax") #included
        self.nx = Config.getint("Map", "nx") #512, 128 
        self.imgsizepix = self.nx #TODO: remove duplicate

        self.mask = (scipy.ndimage.imread(mlcmbpath+"data/"+Config.get("Map", "fname_mask"))[:,:,0]/255).astype(np.float)  
        #mask = np.ones( (nx, nx) ) 

        self.dx = Config.getfloat("Map", "sidelength_deg")*d2r / float(self.nx) 

        self.noise_pix_t = self.nlev_t**2. / (self.dx**2 * rad2arcmin**2)
        self.noise_pix_pol = self.nlev_p**2. / (self.dx**2 * rad2arcmin**2)

        self.wf_mode = Config.get("Training", "wf_mode") #currently T or QU

        self.loss_mode = Config.get("Training", "loss_mode") #J2 J3 J4

        self.batch_size = Config.getint("Training", "batch_size")
        self.epochs = Config.getint("Training", "epochs")

        self.optimizer =  Config.get("Training", "optimizer")

        self.learning_rate = Config.getfloat("Training", "learning_rate")

        #these factors are applied internally in the NN to make the input pixel variance 1
        self.map_rescale_factor_t = Config.getfloat("Dataset", "map_rescale_factor_t")
        #we multiply the map by this factor. the inverse is the sigma of the original map.
        self.map_rescale_factor_pol = Config.getfloat("Dataset", "map_rescale_factor_pol")

        self.nsims_train = Config.getint("Dataset", "nsims_train")
        self.nsims_valid = Config.getint("Dataset", "nsims_valid")
        self.nsims_test = Config.getint("Dataset", "nsims_test")

        self.eps_min = Config.getfloat("Dataset", "eps_min")

        self.npad = Config.getint("NeuralNetwork", "npad")   

        self.kernelsize1 = Config.getint("NeuralNetwork", "kernelsize1") 
        
        self.network = Config.get("NeuralNetwork", "network")   
        
        self.network_paramset = Config.getint("NeuralNetwork", "network_paramset") 

        self.network_actifunc = Config.get("NeuralNetwork", "actifunc")  
        
        #unmutable:
        if self.wf_mode=="T":
            self.wf_signal_type = 'cl_tt'
            self.map_rescale_factor = self.map_rescale_factor_t
            self.noise_pix = self.noise_pix_t
        if self.wf_mode=="QU":
            self.wf_signal_type = 'cl_ee'
            self.map_rescale_factor = self.map_rescale_factor_pol
            self.noise_pix = self.noise_pix_pol    

