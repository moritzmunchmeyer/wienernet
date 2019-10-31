import numpy as np
import quicklens as ql
import scipy
import config

r2d = 180./np.pi
d2r = np.pi/180.


#pass array of form [img_id,:,:,channel], return same array normalized channel wise, and also return variances
def normalize_channelwise(images):
    
#     #remove mean per image
#     for img_id in range(images.shape[0]):
#         for channel_id in range(images.shape[-1]):     
#             avg = (images[img_id,:,:,channel_id]).sum() / images[img_id,:,:,channel_id].size 
#             images[img_id,:,:,channel_id] = images[img_id,:,:,channel_id]-avg
       
    #calculate variance over all images per channel
    variances = np.zeros(images.shape[-1])
    for channel_id in range(images.shape[-1]): 
        if len(images.shape) == 4:
            variances[channel_id] = (images[:,:,:,channel_id]*images[:,:,:,channel_id]).sum() / images[:,:,:,channel_id].size 
            images[:,:,:,channel_id] = (images[:,:,:,channel_id])/variances[channel_id]**(1./2.) 
        if len(images.shape) == 3:
            variances[channel_id] = (images[:,:,channel_id]*images[:,:,channel_id]).sum() / images[:,:,channel_id].size 
            images[:,:,channel_id] = (images[:,:,channel_id])/variances[channel_id]**(1./2.)             
    return images,variances




def ell_filter_maps(maps, nx, dx, lmax, lmin=0):
    nsims = maps.shape[0]
    
    ell_filter = np.ones(10000)   #itlib.lib_qlm.ellmax=5133 for some reason
    ell_filter[lmax:] = 0 #3500
    ell_filter[0:lmin] = 0
    
    for map_id in range(nsims): 
        fullmap_cfft = ql.maps.rmap(nx, dx,map=maps[map_id]).get_cfft()
        filteredmap_cfft = fullmap_cfft * ell_filter
        filteredmap_cfft.fft[0,0] = 0.
        filteredmap = filteredmap_cfft.get_rffts()[0].get_rmap().map
        maps[map_id] = filteredmap
    
    return maps
        
        

def estimate_ps(maps, binnr=30, lmin=2, lmax=3000):
    nmaps = maps.shape[0]
    lbins      = np.linspace(lmin, lmax, binnr)       
    ell_binned = lbins[:-1] + np.diff(lbins)
    power_avg = np.zeros(ell_binned.shape[0])        
    for map_id in range(nmaps): 
        rmap = maps[map_id,:,:]     
        cfft = ql.maps.rmap(config.nx, config.dx,map=rmap).get_cfft()
        power = cfft.get_cl(lbins)
        power_avg += power.cl.real
    power_avg = power_avg/nmaps
    return ell_binned, power_avg
    
    

#periodic padding for image array (img_id,x,y,channels)
def periodic_padding(images,npad):
    if len(images.shape)==4:
        images = np.pad(images,pad_width=((0,0),(npad,npad),(npad,npad),(0,0)),mode='wrap')
    if len(images.shape)==3:
        images = np.pad(images,pad_width=((npad,npad),(npad,npad),(0,0)),mode='wrap')        
    return images



#pass true kappa and max like kappa
#find the spectrum S and N.
#wiener filter S/(S+N)*kappa_true
def wiener_filter_kappa(data_input, deg, nx,dx):
    nsims = data_input.shape[0]
    
    #################### calc S and N power spectra needed for WF
    
    #calculate kappa correlation coeff
    lmax       = 3500 #3500
    lbins      = np.linspace(100, lmax, 20)       
    ell_binned = lbins[:-1] + np.diff(lbins)

    #kappa
    corr_coeff_qe_avg = np.zeros(ell_binned.shape[0])
    corr_coeff_it_avg = np.zeros(ell_binned.shape[0])
    auto_qe_avg = np.zeros(ell_binned.shape[0])
    auto_it_avg = np.zeros(ell_binned.shape[0])
    auto_true_avg = np.zeros(ell_binned.shape[0])
    R_qe_avg = np.zeros(ell_binned.shape[0])
    R_it_avg = np.zeros(ell_binned.shape[0])

    for map_id in range(nsims): 
        #load maps
        kappa_true_map = data_input[map_id,:,:,2]    
        kappa_qe_map = data_input[map_id,:,:,4]
        kappa_it_map = data_input[map_id,:,:,5]

        #make these rmaps and get cffts from which we can get cls and mls
        kappa_true_cfft = ql.maps.rmap(nx, dx,map=kappa_true_map).get_cfft()
        kappa_qe_cfft = ql.maps.rmap(nx, dx,map=kappa_qe_map).get_cfft()
        kappa_it_cfft = ql.maps.rmap(nx, dx,map=kappa_it_map).get_cfft()

        #cross powers
        cross_map_cfft_qe = ql.maps.cfft( nx, dx, fft=(kappa_qe_cfft.fft * np.conj(kappa_true_cfft.fft)) )
        cross_power_qe = cross_map_cfft_qe.get_ml(lbins)  #use ml because the cfft already is a power/multiple of two maps
        cross_map_cfft_it = ql.maps.cfft( nx, dx, fft=(kappa_it_cfft.fft * np.conj(kappa_true_cfft.fft)) )
        cross_power_it = cross_map_cfft_it.get_ml(lbins)  #use ml because the cfft already is a power/multiple of two maps    

        #auto powers
        auto_true = kappa_true_cfft.get_cl(lbins) #use cl because we really want the power of this map
        auto_qe = kappa_qe_cfft.get_cl(lbins) 
        auto_it = kappa_it_cfft.get_cl(lbins) 
        auto_true_avg += auto_true.cl.real
        auto_qe_avg += auto_qe.cl.real
        auto_it_avg += auto_it.cl.real

        #corr coeff from spectra
        corr_coeff_qe = cross_power_qe.specs['cl']/(auto_qe.specs['cl']*auto_true.specs['cl'])**(1./2) 
        corr_coeff_qe_avg += corr_coeff_qe.real
        corr_coeff_it = cross_power_it.specs['cl']/(auto_it.specs['cl']*auto_true.specs['cl'])**(1./2) 
        corr_coeff_it_avg += corr_coeff_it.real    

        #QE renormalisation
        R_qe = (cross_power_qe.specs['cl']/auto_true.specs['cl'])
        R_qe_avg += R_qe.real

        #IT renormalisation
        R_it = (cross_power_it.specs['cl']/auto_true.specs['cl'])
        R_it_avg += R_it.real

    #averages
    corr_coeff_qe_avg = corr_coeff_qe_avg/nsims
    corr_coeff_it_avg = corr_coeff_it_avg/nsims
    auto_qe_avg = auto_qe_avg/nsims
    auto_it_avg = auto_it_avg/nsims
    auto_true_avg = auto_true_avg/nsims
    R_qe_avg = R_qe_avg/nsims
    R_it_avg = R_it_avg/nsims

    #renormalized powers
    auto_qe_renorm = auto_qe_avg/R_qe_avg**2.
    auto_it_renorm = auto_it_avg/R_it_avg**2.

    #noises defined as  <est^2> = Cl + Nl
    noise_qe = auto_qe_renorm - auto_true_avg 
    noise_it = auto_it_renorm - auto_true_avg 


    
    #################### Wiener filter
    
    #get interpolated power spectra
    ell_all = np.arange(0,10000).astype(np.float)
    
#    nl_interp = scipy.interpolate.interp1d(ell_binned,noise_it*ell_binned**4., kind='linear',fill_value=0,bounds_error=False) 
#    nl_allell = nl_interp(ell_all)/ell_all**4.
    nl_interp = scipy.interpolate.interp1d(ell_binned,noise_it*ell_binned**0., kind='linear',fill_value=0,bounds_error=False) 
    nl_allell = nl_interp(ell_all)/ell_all**0.

    cl_interp = scipy.interpolate.interp1d(ell_binned,auto_true_avg*ell_binned**4., kind='linear',fill_value=0,bounds_error=False) 
    cl_allell = cl_interp(ell_all)/ell_all**4.    
    
    kappa_WF = np.zeros( (data_input.shape[0],data_input.shape[1], data_input.shape[2]), dtype=data_input.dtype)
    
    for map_id in range(nsims): 
        kappa_true_map = data_input[map_id,:,:,2]  
        kappa_true_cfft = ql.maps.rmap(nx, dx,map=kappa_true_map).get_cfft()
        kappa_wiener_cfft = kappa_true_cfft * np.nan_to_num(cl_allell/(cl_allell+nl_allell)) #*(1./ (1./nl_allell + 1./clpp_allell))
        kappa_wiener_cfft.fft[0,0] = 0.
        kappa_wiener_map = kappa_wiener_cfft.get_rffts()[0].get_rmap().map
        kappa_WF[map_id] = kappa_wiener_map

    return kappa_WF, ell_all, nl_allell, cl_allell



    
        
        
        
