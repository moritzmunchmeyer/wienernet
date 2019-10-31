import tensorflow as tf
import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K 
import utilities
import scipy
import numpy as np
import quicklens as ql
import config




#from quicklens maps.py
def get_lxly(nx, dx, ny=None, dy=None):
    if ny is None:
        ny = nx
        dy = dx
    """ returns the (lx, ly) pair associated with each Fourier mode. """
    return np.meshgrid( np.fft.fftfreq( nx, dx )[0:nx/2+1]*2.*np.pi,
                            np.fft.fftfreq( ny, dy )*2.*np.pi )  

#from quicklens maps.py
def get_ell(nx, dx, ny=None, dy=None):
    """ returns the wavenumber l = \sqrt(lx**2 + ly**2) for each Fourier mode """
    lx, ly = get_lxly(nx, dx)
    return np.sqrt(lx**2 + ly**2)


def get_pixel_transfer(nx,dx, ny=None, dy=None):  
    if ny is None:
        ny = nx
        dy = dx
    """ return the FFT describing the map-level transfer function for the pixelization of this object. """
    lx, ly = get_lxly(nx,dx)
    fft = np.zeros( lx.shape )
    fft[ 0, 0] = 1.0
    fft[ 0,1:] = np.sin(dx*lx[ 0,1:]/2.) / (dx * lx[0,1:] / 2.)
    fft[1:, 0] = np.sin(dy*ly[1:, 0]/2.) / (dy * ly[1:,0] / 2.)
    fft[1:,1:] = np.sin(dx*lx[1:,1:]/2.) * np.sin(dy*ly[1:,1:]/2.) / (dx * dy * lx[1:,1:] * ly[1:,1:] / 4.)
    return fft


#adapted from quicklens
def get_rfft(rmap, nx, dx):
    """ return an rfft array containing the real fourier transform of this map. """
    #from quicklens.maps.rmap.get_rfft()
    tfac = np.sqrt((dx * dx) / (nx * nx))
    rfft = tf.spectral.rfft2d(rmap) * tfac
    return rfft



def get_lxly_cfft(nx, dx, ny=None, dy=None):
    """ returns the (lx, ly) pair associated with each Fourier mode. """
    if ny is None:
        ny = nx
        dy = dx    
    return np.meshgrid( np.fft.fftfreq( nx, dx )*2.*np.pi,
                        np.fft.fftfreq( ny, dy )*2.*np.pi )


def get_ell_cfft(nx, dx, ny=None, dy=None):
    """ returns the wavenumber l = \sqrt(lx**2 + ly**2) for each Fourier mode """ 
    lx, ly = get_lxly_cfft(nx, dx)
    return np.sqrt(lx**2 + ly**2)


def get_cfft(rmap, nx, dx):
    """ return the complex FFT. """    
#     rfft = get_rfft(rmap, nx, dx)
#     cfft = np.zeros( (nx, nx), dtype=np.complex )
#     cfft[:,0:(nx/2+1)] = rfft[:,:]
#     cfft[0,(nx/2+1):]  = np.conj(rfft[0,1:nx/2][::-1])
#     cfft[1:,(nx/2+1):]  = np.conj(rfft[1:,1:nx/2][::-1,::-1])    
    tfac = np.sqrt((dx * dx) / (nx * nx))
    rmap = tf.cast(rmap,tf.complex64)
    cfft = tf.spectral.fft2d(rmap) * tfac
    return cfft



#adapted from quicklens
def get_rmap(rfft, nx, dx):
    #from quicklens.maps.rfft.get_rmap()
    """ return the rmap given by this FFT. """
    tfac = np.sqrt((nx * nx) / (dx * dx))
    rmap = tf.spectral.irfft2d(rfft)*tfac    
    return rmap

def get_modepowers(y,nx,dx):

    #real space map, defined by rmap, nx, dx
    rmap = y[:,:,:,0] #need to strip last dimension, because fft2d acts on last two channels

    #fft
    rfft = get_rfft(rmap, nx, dx)
    rfft_shape = rfft.get_shape().as_list()

    #power of difference map
    power = tf.real((rfft * tf.conj(rfft))) #tf.math.conj in higher versions
    power = tf.reshape(power,[-1,rfft_shape[1]*rfft_shape[2]]) #flatten except batch dimension

    return power


#based on https://github.com/dhanson/quicklens/blob/master/quicklens/maps.py tqumap.get_teb(self)
def get_ebfft(qmap,umap, nx, dx):
    """ return e b containing the fourier transform of the Q,U maps. """

    lx, ly = get_lxly(nx,dx)
    tpi  = 2.*np.arctan2(lx, -ly)

    tfac = np.sqrt((dx * dx) / (nx * nx))
    qfft = tf.spectral.rfft2d(qmap) * tfac
    ufft = tf.spectral.rfft2d(umap) * tfac

    efft = (+np.cos(tpi) * qfft + np.sin(tpi) * ufft)
    bfft = (-np.sin(tpi) * qfft + np.cos(tpi) * ufft)
    return efft,bfft


#https://github.com/dhanson/quicklens/blob/master/quicklens/maps.py tebfft.get_tqu
def get_qumaps(efft,bfft,nx,dx):
    """ returns the tqumap given by the inverse Fourier transform of this object. """
    lx, ly = get_lxly(nx,dx)
    tpi  = 2.*np.arctan2(lx, -ly)
    tfac = np.sqrt((nx * nx) / (dx * dx))
    qmap = tf.spectral.irfft2d(np.cos(tpi)*efft - np.sin(tpi)*bfft) * tfac
    umap = tf.spectral.irfft2d(np.sin(tpi)*efft + np.cos(tpi)*bfft) * tfac
    return qmap, umap



def ell_filter(y,nx,dx):
    #real space map, defined by rmap, nx, dx
    rmap = y[:,:,:,0]    

    #fft
    rfft = get_rfft(rmap, self.params.nx, self.params.dx)

    #multiply tensor by a k-space mask
    mask = powerspectra.ellmask_nonzero_nonflat.astype(int)
    rfft = rfft*mask

    #get back real space map
    rmap = get_rmap(rfft, nx, dx)
    rmap = tf.expand_dims(rmap,axis=-1) #reintroduce the channel dimension

    return rmap






#class to carry power spectra used in loss with the right ell sampling for nx dx fourier space fields
#units: reads in muK and renormalizes to nn input map with var~1
class Powerspectra():
    def __init__(self,params):
        self.params = params
        
        self.ell = get_ell(self.params.nx,self.params.dx).flatten() #get ell sampling corresponding to our fourier modes for dx,nx
        self.ellmask_nonzero = np.logical_and(self.ell >= 2, self.ell <= self.params.lmax)

        self.pixel_transfer_function_flat = get_pixel_transfer(self.params.nx,self.params.dx).flatten()

        self.ell_nonflat = get_ell(self.params.nx,self.params.dx)
        self.ellmask_nonzero_nonflat = np.logical_and(self.ell_nonflat >= 2, self.ell_nonflat <= self.params.lmax)

        self.bl = ql.spec.bl(fwhm_arcmin=self.params.fwhm_arcmin, lmax=self.params.lmax) # instrumental beam transfer function.

        self.cl_len = ql.spec.get_camb_lensedcl(lmax=self.params.lmax)  # cmb theory spectra. in ell sampling [0:lmax]. unit muK!
        ell_ql = np.arange(0,self.cl_len.cltt.shape[0])

        #cl_tt = self.cl_len.cltt*self.params.map_rescale_factor_t**2. #standard, no beam. #unit: NN normalisation
        cl_tt = self.cl_len.cltt*self.params.map_rescale_factor_t**2. * self.bl**2. #modified, with beam
        nl_tt = (self.params.nlev_t*np.pi/180./60.)**2  * self.params.map_rescale_factor_t**2. * np.ones_like(self.bl) #/ self.bl**2

        #cl_ee = self.cl_len.clee*self.params.map_rescale_factor_pol**2. 
        cl_ee = self.cl_len.clee*self.params.map_rescale_factor_pol**2. * self.bl**2.
        nl_ee = (self.params.nlev_p*np.pi/180./60.)**2  * self.params.map_rescale_factor_pol**2. * np.ones_like(self.bl)#/ self.bl**2

        #cl_bb = self.cl_len.clbb*self.params.map_rescale_factor_pol**2. 
        cl_bb = self.cl_len.clbb*self.params.map_rescale_factor_pol**2. * self.bl**2. 
        nl_bb = (self.params.nlev_p*np.pi/180./60.)**2 * self.params.map_rescale_factor_pol**2. * np.ones_like(self.bl) #/ self.bl**2


        #we interpolate to the required ell sampling
        #------------V1: ell<2 = 0 (default)
#         interp = scipy.interpolate.interp1d(np.log(ell_ql[2:]),np.log(cl_tt[2:]), kind='linear',fill_value=0,bounds_error=False) 
#         self.cl_tt = np.zeros(self.ell.shape[0])
#         self.cl_tt[1:] = np.exp(interp(np.log(self.ell[1:]))) #* self.pixel_transfer_function_flat[1:]**2

#         interp = scipy.interpolate.interp1d(np.log(ell_ql[2:]),np.log(nl_tt[2:]), kind='linear',fill_value=0,bounds_error=False) 
#         self.nl_tt = np.zeros(self.ell.shape[0])
#         self.nl_tt[1:] = np.exp(interp(np.log(self.ell[1:])))     

#         interp = scipy.interpolate.interp1d(np.log(ell_ql[2:]),np.log(cl_ee[2:]), kind='linear',fill_value=0,bounds_error=False) 
#         self.cl_ee = np.zeros(self.ell.shape[0])
#         self.cl_ee[1:] = np.exp(interp(np.log(self.ell[1:]))) #* self.pixel_transfer_function_flat[1:]**2

#         interp = scipy.interpolate.interp1d(np.log(ell_ql[2:]),np.log(nl_ee[2:]), kind='linear',fill_value=0,bounds_error=False) 
#         self.nl_ee = np.zeros(self.ell.shape[0])
#         self.nl_ee[1:] = np.exp(interp(np.log(self.ell[1:])))

#         interp = scipy.interpolate.interp1d(np.log(ell_ql[2:]),np.log(cl_bb[2:]), kind='linear',fill_value=0,bounds_error=False) 
#         self.cl_bb = np.zeros(self.ell.shape[0])
#         self.cl_bb[1:] = np.exp(interp(np.log(self.ell[1:]))) #* self.pixel_transfer_function_flat[1:]**2

#         interp = scipy.interpolate.interp1d(np.log(ell_ql[2:]),np.log(nl_bb[2:]), kind='linear',fill_value=0,bounds_error=False) 
#         self.nl_bb = np.zeros(self.ell.shape[0])
#         self.nl_bb[1:] = np.exp(interp(np.log(self.ell[1:])))        

        #--------------V2: extend quadrupole to monopole and dipole to not leave modes unconstrained
        cl_tt[0:2] = cl_tt[2] #/100.
        interp = scipy.interpolate.interp1d(np.log(ell_ql[1:]),np.log(cl_tt[1:]), kind='linear',fill_value=0,bounds_error=False) 
        self.cl_tt = np.zeros(self.ell.shape[0])
        self.cl_tt[1:] = np.exp(interp(np.log(self.ell[1:])))
        self.cl_tt[0] = self.cl_tt[1]

        nl_tt[0:2] = nl_tt[2] #/100.
        interp = scipy.interpolate.interp1d(np.log(ell_ql[1:]),np.log(nl_tt[1:]), kind='linear',fill_value=0,bounds_error=False) 
        self.nl_tt = np.zeros(self.ell.shape[0])
        self.nl_tt[1:] = np.exp(interp(np.log(self.ell[1:])))     
        self.nl_tt[0] = self.nl_tt[1]

        cl_ee[0:2] = cl_ee[2] #/100.
        interp = scipy.interpolate.interp1d(np.log(ell_ql[1:]),np.log(cl_ee[1:]), kind='linear',fill_value=0,bounds_error=False) 
        self.cl_ee = np.zeros(self.ell.shape[0])
        self.cl_ee[1:] = np.exp(interp(np.log(self.ell[1:])))
        self.cl_ee[0] = self.cl_ee[1]

        nl_ee[0:2] = nl_ee[2] #/100.
        interp = scipy.interpolate.interp1d(np.log(ell_ql[1:]),np.log(nl_ee[1:]), kind='linear',fill_value=0,bounds_error=False) 
        self.nl_ee = np.zeros(self.ell.shape[0])
        self.nl_ee[1:] = np.exp(interp(np.log(self.ell[1:])))
        self.nl_ee[0] = self.nl_ee[1]

        cl_bb[0:2] = cl_bb[2] #/100.
        interp = scipy.interpolate.interp1d(np.log(ell_ql[1:]),np.log(cl_bb[1:]), kind='linear',fill_value=0,bounds_error=False) 
        self.cl_bb = np.zeros(self.ell.shape[0])
        self.cl_bb[1:] = np.exp(interp(np.log(self.ell[1:])))
        self.cl_bb[0] = self.cl_bb[1]

        nl_bb[0:2] = nl_bb[2] #/100.
        interp = scipy.interpolate.interp1d(np.log(ell_ql[1:]),np.log(nl_bb[1:]), kind='linear',fill_value=0,bounds_error=False) 
        self.nl_bb = np.zeros(self.ell.shape[0])
        self.nl_bb[1:] = np.exp(interp(np.log(self.ell[1:])))
        self.nl_bb[0] = self.nl_bb[1]        


    def inverse_ps_weight(self,data,spectype):
        #data is a tf tensor [batch_id,:]. 
        #where ell is ell < lmin OR ell > ellmax we want to set the data to zero
        #elsewhere we want to set it to 1/cl
        #https://stackoverflow.com/questions/48510741/how-to-conditionally-assign-values-to-tensor-masking-for-loss-function

        cl = self.get_power_spectrum(spectype)

        #test: try with booloean mask (not finished)
        #data = tf.boolean_mask(data, self.ellmask_nonzero) / cl[self.ellmask_nonzero]

        #test: try with scatter update (not finished)
        #data = data/cl
        #data = tf.scatter_update(data, tf.Variable(np.logical_not(self.ellmask_nonzero)), tf.Variable(0.))
        #https://stackoverflow.com/questions/54110085/using-scatter-update-with-feeded-data-in-tensorflow
        #https://stackoverflow.com/questions/53632837/tensorflow-assign-tensor-to-tensor-with-array-indexing
        #https://stackoverflow.com/questions/52872239/can-tf-scatter-update-or-tf-scatter-nd-update-be-used-to-update-column-slices-of

        #simply use div_no_nan, should work
        data = tf.div_no_nan(data,cl) #https://www.tensorflow.org/api_docs/python/tf/div_no_nan

        #for all cl non-zero:
        #data = tf.div(data,cl)

        return data


    def ps_weight(self,data,spectype):
        if spectype=='clee/(clee+nlee)':
            cl = self.get_power_spectrum(spectype='cl_ee')  
            nl = self.get_power_spectrum(spectype='nl_ee')  
            data = data * cl
            data = tf.div(data,cl+nl) #https://www.tensorflow.org/api_docs/python/tf/div_no_nan #div_no_nan
        if spectype=='clbb/(clbb+nlbb)':
            cl = self.get_power_spectrum(spectype='cl_bb')  
            nl = self.get_power_spectrum(spectype='nl_bb')  
            data = data * cl
            data = tf.div(data,cl+nl) #https://www.tensorflow.org/api_docs/python/tf/div_no_nan #div_no_nan

        return data    



    def get_power_spectrum(self,spectype):
        if spectype=='cl_tt':
            return self.cl_tt
        if spectype=='nl_tt':
            return self.nl_tt        
        if spectype=='cl_ee':
            return self.cl_ee
        if spectype=='nl_ee':
            return self.nl_ee
        if spectype=='cl_bb':
            return self.cl_bb
        if spectype=='nl_bb':
            return self.nl_bb        
        if spectype=='ell_square':
            return self.ell**2.
        if spectype=='ell':
            return self.ell






class Lossfunctions():
    
    def __init__(self,params):
        self.params = params
        self.powerspectra = Powerspectra(self.params)    


    def fourier_loss_diff(self,y_true, y_pred, spectype=None):

        #real space map, defined by rmap, nx, dx
        rmap_true = y_true[:,:,:,0] #need to strip last dimension, because fft2d acts on last two channels
        rmap_pred = y_pred[:,:,:,0]

        #fft
        rfft_true = get_rfft(rmap_true, self.params.nx, self.params.dx)
        rfft_pred = get_rfft(rmap_pred, self.params.nx, self.params.dx)
        rfft_shape = rfft_true.get_shape().as_list()

        #power of difference map
        rfft_diff = rfft_pred-rfft_true  #rfft_pred
        diffpower = tf.real((rfft_diff * tf.conj(rfft_diff))) #tf.math.conj in higher versions
        diffpower = tf.reshape(diffpower,[-1,rfft_shape[1]*rfft_shape[2]]) #flatten except batch dimension

        #weight by some power spectrum / noise spectrum
        if spectype is not None:
            diffpower = self.powerspectra.inverse_ps_weight(diffpower,spectype)

        score = tf.reduce_mean(diffpower)      
        return score



    def fourier_loss_auto(self,y, spectype=None):   
        if self.params.wf_mode=="T":    
            #real space map, defined by rmap, nx, dx
            rmap = y[:,:,:,0]    

            ######## rfft version. 
            #fft
            rfft = get_rfft(rmap, self.params.nx, self.params.dx)
            rfft_shape = rfft.get_shape().as_list()
            #print ("rfftshape", rfft_shape)
            #power of modes
            power = tf.real((rfft * tf.conj(rfft))) #tf.math.conj in higher versions
            power = tf.reshape(power,[-1,rfft_shape[1]*rfft_shape[2]]) #flatten except batch dimension

            ######## cfft version. ALSO change ell in powerspectra.
        #     cfft = get_cfft(rmap, self.params.nx, self.params.dx)
        #     cfft_shape = cfft.get_shape().as_list()
        #     power = tf.real((cfft * tf.conj(cfft))) #tf.math.conj in higher versions
        #     power = tf.reshape(power,[-1,cfft_shape[1]*cfft_shape[2]]) #flatten except batch dimension       

            #weight by some power spectrum / noise spectrum
            power = self.powerspectra.inverse_ps_weight(power,spectype='cl_tt')

            print ("power", power)
            loss = tf.reduce_mean(power) 
            print ("loss", loss)

        if self.params.wf_mode=="QU":           
            rmap_Q = y[:,:,:,0] 
            rmap_U = y[:,:,:,1]   

            #get E and B modes
            efft,bfft = get_ebfft(rmap_Q,rmap_U,self.params.nx,self.params.dx)
            efft_shape = efft.get_shape().as_list()
            bfft_shape = bfft.get_shape().as_list()

            #power of modes
            power_E = tf.real((efft * tf.conj(efft))) 
            power_E = tf.reshape(power_E,[-1,efft_shape[1]*efft_shape[2]]) #
            power_B = tf.real((bfft * tf.conj(bfft))) 
            power_B = tf.reshape(power_B,[-1,bfft_shape[1]*bfft_shape[2]]) 

            #weight by some power spectrum / noise spectrum
            power_E = self.powerspectra.inverse_ps_weight(power_E,spectype='cl_ee')
            power_B = self.powerspectra.inverse_ps_weight(power_B,spectype='cl_bb')  #cl_ee TEST

            loss = tf.reduce_mean(power_E) + tf.reduce_mean(power_B) 

        return loss


    #y_true is the cmbobs map
    def realspace_loss_noisediag(self,y_true,y_pred):
        if self.params.wf_mode=="T":
            y_true = y_true[:,:,:,0]
            y_pred = y_pred[:,:,:,0]

            #ell filter the map to remove high freq noise
            #y_pred = ell_filter(y_pred)

            #noise diagonal in pixel space
            #square map in real space and multiply by mask to remove masked pixels. weight the others by noise.

            loss = (y_pred-y_true)*(y_pred-y_true) / (self.params.noise_pix * self.params.map_rescale_factor**2.)
            print ("loss1", loss)

            loss = loss * self.params.mask
            print ("loss2", loss)

            loss = tf.reduce_mean(loss) 
            print ("loss3", loss)

        if self.params.wf_mode=="QU":     
            y_true_Q = y_true[:,:,:,0]
            y_pred_Q = y_pred[:,:,:,0]
            y_true_U = y_true[:,:,:,1]
            y_pred_U = y_pred[:,:,:,1]        

            loss_Q = (y_pred_Q-y_true_Q)*(y_pred_Q-y_true_Q) / (self.params.noise_pix * self.params.map_rescale_factor**2.)
            loss_Q = loss_Q * self.params.mask
            loss_U = (y_pred_U-y_true_U)*(y_pred_U-y_true_U) / (self.params.noise_pix * self.params.map_rescale_factor**2.)
            loss_U = loss_U * self.params.mask
            loss = tf.reduce_mean(loss_Q) + tf.reduce_mean(loss_U) 
            print ("loss", loss)              

        return loss



    #nn input: cmbobs,mask
    #nn output: ypred = cmb_predicted
    #ytrue: cmbobs #do not feed the mask here because otherwise the keras would make a channel for it
    def loss_wiener_J3(self,y_true, y_pred):

        #real space noise weighted difference
        term1 = self.realspace_loss_noisediag(y_true,y_pred)

        #prior on the input map
        term2 = self.fourier_loss_auto(y_pred)

        loss = term1 + term2   
        return loss


    def loss_wiener_J4(self,y_true, y_pred):

        #real space noise weighted difference
        J3term1 = self.realspace_loss_noisediag(y_true[...,0:2],y_pred)

        #prior on the input map
        J3term2 = self.fourier_loss_auto(y_pred)

        J2 = self.loss_pixelMSE_unfiltered(y_true[...,2:4], y_pred)

        loss = J3term1 + J3term2 + 10.*J2 
        return loss




    #take in Q,U. transform to E,B. weight by Cl/Cl+Nl. pass this to fourier loss term. re transfrom to QU for noise term.
    def loss_wiener_J3_wfweighting(self,y_true, y_pred):
        y_true_Q = y_true[:,:,:,0]
        y_pred_Q = y_pred[:,:,:,0]
        y_true_U = y_true[:,:,:,1]
        y_pred_U = y_pred[:,:,:,1] 

        #do the prefiltering
        y_pred_efft,y_pred_bfft = get_ebfft(y_pred_Q,y_pred_U, self.params.nx, self.params.dx)
        efft_shape = y_pred_efft.get_shape().as_list()
        bfft_shape = y_pred_bfft.get_shape().as_list()

        y_pred_efft = tf.reshape(y_pred_efft,[-1,efft_shape[1]*efft_shape[2]]) #flatten
        y_pred_bfft = tf.reshape(y_pred_bfft,[-1,bfft_shape[1]*bfft_shape[2]])

        y_pred_efft = self.powerspectra.ps_weight(y_pred_efft,spectype='clee/(clee+nlee)')    
        y_pred_bfft = self.powerspectra.ps_weight(y_pred_bfft,spectype='clbb/(clbb+nlee)')  

        #term 2
        if self.params.wf_mode=="QU":           
            #power of modes
            power_E = tf.real((y_pred_efft * tf.conj(y_pred_efft))) 
            #power_E = tf.reshape(power_E,[-1,efft_shape[1]*efft_shape[2]]) 
            power_B = tf.real((y_pred_bfft * tf.conj(y_pred_bfft))) 
            #power_B = tf.reshape(power_B,[-1,bfft_shape[1]*bfft_shape[2]]) 

            #weight by some power spectrum / noise spectrum
            power_E = self.powerspectra.inverse_ps_weight(power_E,spectype='cl_ee')
            power_B = self.powerspectra.inverse_ps_weight(power_B,spectype='cl_bb')  

            term2 = tf.reduce_mean(power_E) + tf.reduce_mean(power_B)  

        #term 1 
        if self.params.wf_mode=="QU": 
            y_pred_efft = tf.reshape(y_pred_efft,[-1,efft_shape[1],efft_shape[2]])
            y_pred_bfft = tf.reshape(y_pred_bfft,[-1,bfft_shape[1],bfft_shape[2]])
            y_pred_Q, y_pred_U = get_qumaps(y_pred_efft,y_pred_bfft,self.params.nx,self.params.dx)
            y_true_Q = y_true[:,:,:,0]
            y_true_U = y_true[:,:,:,1]       

            loss_Q = (y_pred_Q-y_true_Q)*(y_pred_Q-y_true_Q) / (self.params.noise_pix * self.params.map_rescale_factor**2.)
            loss_Q = loss_Q * self.params.mask
            loss_U = (y_pred_U-y_true_U)*(y_pred_U-y_true_U) / (self.params.noise_pix * self.params.map_rescale_factor**2.)
            loss_U = loss_U * self.params.mask
            term1 = tf.reduce_mean(loss_Q) + tf.reduce_mean(loss_U)        

        loss = term1 + term2   
        return loss


    #nn input: cmbobs,mask
    #nn output: ypred = cmb_predicted
    #ytrue: cmbtrue (unmasked, no noise)
    def loss_wiener_J2(self,y_true, y_pred):
        cmbtrue_map = y_true

        loss = self.fourier_loss_diff(y_pred,cmbtrue_map, spectype=None) #spectype=None,'ell_square','ell' try different weighting    
        return loss



    def loss_pixelMSE_ellfiltered(self,y_true, y_pred):
        rmap_true = y_true
        rmap_pred = ell_filter(y_pred,self.params.nx,self.params.dx)

        loss = tf.reduce_mean((rmap_true-rmap_pred)*(rmap_true-rmap_pred))     
        return loss


    def loss_pixelMSE_unfiltered(self,y_true, y_pred):
        rmap_true = y_true
        rmap_pred = y_pred

        loss = tf.reduce_mean((rmap_true-rmap_pred)*(rmap_true-rmap_pred))     
        return loss


