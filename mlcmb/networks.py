import tensorflow as tf
import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K 

import config



###################### WienerNet: Linear-NonLinear networks for Wiener filtering with a mask




class WienerNet():
    
    def __init__(self,params):
        self.params = params
        

    def get_cropdim(self,input,target):
        inputdimx = input.get_shape().as_list()[-3]
        targetdimx = target.get_shape().as_list()[-2]

        if (inputdimx-targetdimx)%2 ==0:
            cropx = int((inputdimx-targetdimx)/2)
            cropy = cropx
        else: 
            cropx = int((inputdimx-targetdimx-1)/2)
            cropy = int((inputdimx-targetdimx+1)/2)
        return (cropx,cropy),(cropx,cropy) 

    
    

    #nx 128, ksize 5: npad = 128
    #nx 128, ksize 7: npad = 180
    def net_linear_nonlinear_doubleU_periodic_6encoders(self,img_shape,channels_out):  
        padding = 'valid'
        inputs = layers.Input(shape=img_shape)

        if self.params.wf_mode == "T":
            mapdata = layers.Lambda(lambda x: x[:,:,:,0:1])(inputs)
            maskdata = layers.Lambda(lambda x: x[:,:,:,1:2])(inputs)
        if self.params.wf_mode == "QU":
            mapdata = layers.Lambda(lambda x: x[:,:,:,0:2])(inputs)
            maskdata = layers.Lambda(lambda x: x[:,:,:,2:3])(inputs)

        
        if self.params.network_paramset == 0:
            nonlinear1 = False
            nonlinear2 = True 
            num_filters0 = 10 #1
            num_filters1 = 10 #1
            num_filters2 = 10 #1
            num_filters3 = 10 #1
            num_filters4 = 10 #1
            num_filters5 = 10 #1
            densebridge1 = False 
            densebridge2 = False 
            ksize = self.params.kernelsize1 #5             
            
        if self.params.network_paramset == 1:
            nonlinear1 = False
            nonlinear2 = True
            num_filters0 = 16 
            num_filters1 = 16 
            num_filters2 = 32 
            num_filters3 = 32 
            num_filters4 = 32 
            num_filters5 = 32 
            densebridge1 = False 
            densebridge2 = False 
            ksize = self.params.kernelsize1 #5 
            
        if self.params.network_paramset == 2:
            nonlinear1 = False
            nonlinear2 = True 
            num_filters0 = 8
            num_filters1 = 8
            num_filters2 = 16
            num_filters3 = 16
            num_filters4 = 32 
            num_filters5 = 32 
            densebridge1 = False 
            densebridge2 = False 
            ksize = self.params.kernelsize1 #5 
            
        actifunc = self.params.network_actifunc

        ######### non-linear U

        encoder0_nonlin = layers.Conv2D(filters=num_filters0, kernel_size=(ksize, ksize), padding=padding, strides=1)(maskdata) #needs stride 1
        #print ("E0", encoder0_nonlin)
        if nonlinear2:
            encoder0_nonlin = layers.Activation(activation=actifunc)(encoder0_nonlin) 

        encoder1_nonlin = encoderblock_1(encoder0_nonlin, num_filters=num_filters1,kernel_size=ksize,nonlinear=nonlinear2, padding=padding,activation=actifunc)
        #print ("E1", encoder1_nonlin)   
        encoder2_nonlin = encoderblock_1(encoder1_nonlin, num_filters=num_filters2,kernel_size=ksize,nonlinear=nonlinear2, padding=padding,activation=actifunc)
        #print ("E2", encoder2_nonlin)    
        encoder3_nonlin = encoderblock_1(encoder2_nonlin, num_filters=num_filters3,kernel_size=ksize,nonlinear=nonlinear2, padding=padding,activation=actifunc)

        encoder4_nonlin = encoderblock_1(encoder3_nonlin, num_filters=num_filters4,kernel_size=ksize,nonlinear=nonlinear2, padding=padding,activation=actifunc)    

        encoder5_nonlin = encoderblock_1(encoder4_nonlin, num_filters=num_filters5,kernel_size=ksize,nonlinear=nonlinear2, padding=padding,activation=actifunc)     

        #print ("E3", encoder3_nonlin)    
        if densebridge2:
            bridge_nonlin = bridge_sc_1(encoder5_nonlin,num_filters=num_filters5) #with linear layer

            if nonlinear2:
                bridge_nonlin = layers.Activation(activation=actifunc)(bridge_nonlin)    
        else:
            bridge_nonlin = encoder5_nonlin

        decoder5_nonlin = decoderblock_3(bridge_nonlin, num_filters=num_filters5,kernel_size=ksize,nonlinear=nonlinear2,padding=padding,activation=actifunc)          

        encoder4_nonlin = layers.Cropping2D(cropping=self.get_cropdim(encoder4_nonlin,decoder5_nonlin))(encoder4_nonlin) 
        skipcon4_nonlin = layers.concatenate([encoder4_nonlin, decoder5_nonlin], axis=3)     
        decoder4_nonlin = decoderblock_3(skipcon4_nonlin,num_filters=num_filters4,kernel_size=ksize,nonlinear=nonlinear2,padding=padding,activation=actifunc)          

        encoder3_nonlin = layers.Cropping2D(cropping=self.get_cropdim(encoder3_nonlin,decoder4_nonlin))(encoder3_nonlin) 
        skipcon3_nonlin = layers.concatenate([encoder3_nonlin, decoder4_nonlin], axis=3) 
        decoder3_nonlin = decoderblock_3(skipcon3_nonlin, num_filters=num_filters3,kernel_size=ksize,nonlinear=nonlinear2,padding=padding,activation=actifunc)         

        #print ("D3", decoder3_nonlin)
        encoder2_nonlin_crop = layers.Cropping2D(cropping=self.get_cropdim(encoder2_nonlin,decoder3_nonlin))(encoder2_nonlin) #((top_pad, bottom_pad), (left_pad, right_pad))
        skipcon2_nonlin = layers.concatenate([encoder2_nonlin_crop, decoder3_nonlin], axis=3) 
        decoder2_nonlin = decoderblock_3(skipcon2_nonlin, num_filters=num_filters2,kernel_size=ksize,nonlinear=nonlinear2, padding=padding,activation=actifunc) 
        #print ("D2", decoder2_nonlin)    
        encoder1_nonlin_crop = layers.Cropping2D(cropping=self.get_cropdim(encoder1_nonlin,decoder2_nonlin))(encoder1_nonlin)
        skipcon1_nonlin = layers.concatenate([encoder1_nonlin_crop, decoder2_nonlin], axis=3) 
        decoder1_nonlin = decoderblock_3(skipcon1_nonlin, num_filters=num_filters1,kernel_size=ksize,nonlinear=nonlinear2, padding=padding,activation=actifunc)
        #print ("D1", decoder1_nonlin)   



        ######### linear U

        encoder0_lin = layers.Conv2D(filters=num_filters0, kernel_size=(ksize, ksize), padding=padding, strides=1)(mapdata) #needs stride 1
        encoder0_lin = multi_block_quadraticandsemilinear(encoder0_lin,encoder0_nonlin,channels_out=num_filters1)

        if nonlinear1:
            encoder0_lin = layers.Activation(activation=actifunc)(encoder0_lin) 

        encoder1_lin = encoderblock_1(encoder0_lin, num_filters=num_filters1,kernel_size=ksize,nonlinear=nonlinear1, padding=padding,activation=actifunc)
        encoder1_lin = multi_block_quadraticandsemilinear(encoder1_lin,encoder1_nonlin,channels_out=num_filters1)

        encoder2_lin = encoderblock_1(encoder1_lin, num_filters=num_filters2,kernel_size=ksize,nonlinear=nonlinear1, padding=padding,activation=actifunc)
        encoder2_lin = multi_block_quadraticandsemilinear(encoder2_lin,encoder2_nonlin,channels_out=num_filters1)

        encoder3_lin = encoderblock_1(encoder2_lin, num_filters=num_filters3,kernel_size=ksize,nonlinear=nonlinear1, padding=padding,activation=actifunc)

        encoder4_lin = encoderblock_1(encoder3_lin, num_filters=num_filters4,kernel_size=ksize,nonlinear=nonlinear1, padding=padding,activation=actifunc)  

        encoder5_lin = encoderblock_1(encoder4_lin, num_filters=num_filters5,kernel_size=ksize,nonlinear=nonlinear1, padding=padding,activation=actifunc)  

        if densebridge1:
            bridge_lin = bridge_sc_1(encoder5_lin,num_filters=num_filters5) #with linear layer

            if nonlinear1:
                bridge_lin = layers.Activation(activation=actifunc)(bridge_lin)    
        else:
            bridge_lin = encoder5_lin

        decoder5_lin = decoderblock_3(bridge_lin, num_filters=num_filters5,kernel_size=ksize,nonlinear=nonlinear1, padding=padding,activation=actifunc)  
        decoder5_lin = multi_block_quadraticandsemilinear(decoder5_lin,decoder5_nonlin,channels_out=num_filters1)

        encoder4_lin_crop = layers.Cropping2D(cropping=self.get_cropdim(encoder4_lin,decoder5_lin))(encoder4_lin) #((top_pad, bottom_pad), (left_pad, right_pad))
        skipcon4_lin = layers.concatenate([encoder4_lin_crop, decoder5_lin], axis=3)     
        decoder4_lin = decoderblock_3(skipcon4_lin, num_filters=num_filters4,kernel_size=ksize,nonlinear=nonlinear1, padding=padding,activation=actifunc)  
        decoder4_lin = multi_block_quadraticandsemilinear(decoder4_lin,decoder4_nonlin,channels_out=num_filters1)

        encoder3_lin_crop = layers.Cropping2D(cropping=self.get_cropdim(encoder3_lin,decoder4_lin))(encoder3_lin) #((top_pad, bottom_pad), (left_pad, right_pad))
        skipcon3_lin = layers.concatenate([encoder3_lin_crop, decoder4_lin], axis=3) 
        decoder3_lin = decoderblock_3(skipcon3_lin, num_filters=num_filters3,kernel_size=ksize,nonlinear=nonlinear1, padding=padding,activation=actifunc)  
        decoder3_lin = multi_block_quadraticandsemilinear(decoder3_lin,decoder3_nonlin,channels_out=num_filters1)

        encoder2_lin_crop = layers.Cropping2D(cropping=self.get_cropdim(encoder2_lin,decoder3_lin))(encoder2_lin) #((top_pad, bottom_pad), (left_pad, right_pad))
        skipcon2_lin = layers.concatenate([encoder2_lin_crop, decoder3_lin], axis=3) 
        decoder2_lin = decoderblock_3(skipcon2_lin, num_filters=num_filters2,kernel_size=ksize,nonlinear=nonlinear1, padding=padding,activation=actifunc) 
        decoder2_lin = multi_block_quadraticandsemilinear(decoder2_lin,decoder2_nonlin,channels_out=num_filters1)

        encoder1_lin_crop = layers.Cropping2D(cropping=self.get_cropdim(encoder1_lin,decoder2_lin))(encoder1_lin)
        skipcon1_lin = layers.concatenate([encoder1_lin_crop, decoder2_lin], axis=3) 
        decoder1_lin = decoderblock_3(skipcon1_lin, num_filters=num_filters1,kernel_size=ksize,nonlinear=nonlinear1, padding=padding,activation=actifunc)
        decoder1_lin = multi_block_quadraticandsemilinear(decoder1_lin,decoder1_nonlin,channels_out=num_filters1)

        encoder0_lin_crop = layers.Cropping2D(cropping=self.get_cropdim(encoder0_lin,decoder1_lin))(encoder0_lin)
        skipcon0_lin = layers.concatenate([encoder0_lin_crop, decoder1_lin], axis=3) 
        decoder0_lin = layers.Conv2D(filters=channels_out, kernel_size=(ksize, ksize), padding=padding, strides=1)(skipcon0_lin) #needs stride 1
        outputs = decoder0_lin

        return inputs, outputs


    
    
    
    
    
    def net_linear_nonlinear_doubleU_periodic_7encoders(self,img_shape,channels_out):  
        padding = 'valid'
        inputs = layers.Input(shape=img_shape)

        if self.params.wf_mode == "T":
            mapdata = layers.Lambda(lambda x: x[:,:,:,0:1])(inputs)
            maskdata = layers.Lambda(lambda x: x[:,:,:,1:2])(inputs)
        if self.params.wf_mode == "QU":
            mapdata = layers.Lambda(lambda x: x[:,:,:,0:2])(inputs)
            maskdata = layers.Lambda(lambda x: x[:,:,:,2:3])(inputs)

        nonlinear1 = False
        nonlinear2 = True #True
        num_filters1 = 10 #10 1
        num_filters2 = 10 #10 1
        ksize = self.params.kernelsize1 #5 

        actifunc = self.params.network_actifunc
        
        ######### non-linear U

        encoder0_nonlin = layers.Conv2D(filters=num_filters2, kernel_size=(ksize, ksize), padding=padding, strides=1)(maskdata) #needs stride 1
        #print ("E0", encoder0_nonlin)
        if nonlinear2:
            encoder0_nonlin = layers.Activation(activation=actifunc)(encoder0_nonlin) 

        encoder1_nonlin = encoderblock_1(encoder0_nonlin, num_filters=num_filters2,kernel_size=ksize,nonlinear=nonlinear2, padding=padding)
        #print ("E1", encoder1_nonlin)   
        encoder2_nonlin = encoderblock_1(encoder1_nonlin, num_filters=num_filters2,kernel_size=ksize,nonlinear=nonlinear2, padding=padding)
        #print ("E2", encoder2_nonlin)    
        encoder3_nonlin = encoderblock_1(encoder2_nonlin, num_filters=num_filters2,kernel_size=ksize,nonlinear=nonlinear2, padding=padding)

        encoder4_nonlin = encoderblock_1(encoder3_nonlin, num_filters=num_filters2,kernel_size=ksize,nonlinear=nonlinear2, padding=padding)    

        encoder5_nonlin = encoderblock_1(encoder4_nonlin, num_filters=num_filters2,kernel_size=ksize,nonlinear=nonlinear2, padding=padding)     
        
        encoder6_nonlin = encoderblock_1(encoder5_nonlin, num_filters=num_filters2,kernel_size=ksize,nonlinear=nonlinear2, padding=padding)  
 
        decoder6_nonlin = decoderblock_3(encoder6_nonlin, num_filters=num_filters2,kernel_size=ksize,nonlinear=nonlinear2,padding=padding) 
        
        encoder5_nonlin = layers.Cropping2D(cropping=self.get_cropdim(encoder5_nonlin,decoder6_nonlin))(encoder5_nonlin) 
        skipcon5_nonlin = layers.concatenate([encoder5_nonlin, decoder6_nonlin], axis=3) 
        decoder5_nonlin = decoderblock_3(skipcon5_nonlin, num_filters=num_filters2,kernel_size=ksize,nonlinear=nonlinear2,padding=padding)          

        encoder4_nonlin = layers.Cropping2D(cropping=self.get_cropdim(encoder4_nonlin,decoder5_nonlin))(encoder4_nonlin) 
        skipcon4_nonlin = layers.concatenate([encoder4_nonlin, decoder5_nonlin], axis=3)     
        decoder4_nonlin = decoderblock_3(skipcon4_nonlin,num_filters=num_filters2,kernel_size=ksize,nonlinear=nonlinear2,padding=padding)          

        encoder3_nonlin = layers.Cropping2D(cropping=self.get_cropdim(encoder3_nonlin,decoder4_nonlin))(encoder3_nonlin) 
        skipcon3_nonlin = layers.concatenate([encoder3_nonlin, decoder4_nonlin], axis=3) 
        decoder3_nonlin = decoderblock_3(skipcon3_nonlin, num_filters=num_filters2,kernel_size=ksize,nonlinear=nonlinear2,padding=padding)         

        #print ("D3", decoder3_nonlin)
        encoder2_nonlin_crop = layers.Cropping2D(cropping=self.get_cropdim(encoder2_nonlin,decoder3_nonlin))(encoder2_nonlin) #((top_pad, bottom_pad), (left_pad, right_pad))
        skipcon2_nonlin = layers.concatenate([encoder2_nonlin_crop, decoder3_nonlin], axis=3) 
        decoder2_nonlin = decoderblock_3(skipcon2_nonlin, num_filters=num_filters2,kernel_size=ksize,nonlinear=nonlinear2, padding=padding) 
        #print ("D2", decoder2_nonlin)    
        encoder1_nonlin_crop = layers.Cropping2D(cropping=self.get_cropdim(encoder1_nonlin,decoder2_nonlin))(encoder1_nonlin)
        skipcon1_nonlin = layers.concatenate([encoder1_nonlin_crop, decoder2_nonlin], axis=3) 
        decoder1_nonlin = decoderblock_3(skipcon1_nonlin, num_filters=num_filters2,kernel_size=ksize,nonlinear=nonlinear2, padding=padding)
        #print ("D1", decoder1_nonlin)   



        ######### linear U

        encoder0_lin = layers.Conv2D(filters=num_filters1, kernel_size=(ksize, ksize), padding=padding, strides=1)(mapdata) #needs stride 1
        encoder0_lin = multi_block_quadraticandsemilinear(encoder0_lin,encoder0_nonlin,channels_out=num_filters1)

        if nonlinear1:
            encoder0_lin = layers.Activation(activation=actifunc)(encoder0_lin) 

        encoder1_lin = encoderblock_1(encoder0_lin, num_filters=num_filters1,kernel_size=ksize,nonlinear=nonlinear1, padding=padding)
        encoder1_lin = multi_block_quadraticandsemilinear(encoder1_lin,encoder1_nonlin,channels_out=num_filters1)

        encoder2_lin = encoderblock_1(encoder1_lin, num_filters=num_filters1,kernel_size=ksize,nonlinear=nonlinear1, padding=padding)
        encoder2_lin = multi_block_quadraticandsemilinear(encoder2_lin,encoder2_nonlin,channels_out=num_filters1)

        encoder3_lin = encoderblock_1(encoder2_lin, num_filters=num_filters1,kernel_size=ksize,nonlinear=nonlinear1, padding=padding)

        encoder4_lin = encoderblock_1(encoder3_lin, num_filters=num_filters1,kernel_size=ksize,nonlinear=nonlinear1, padding=padding)  

        encoder5_lin = encoderblock_1(encoder4_lin, num_filters=num_filters1,kernel_size=ksize,nonlinear=nonlinear1, padding=padding)  
        
        encoder6_lin = encoderblock_1(encoder5_lin, num_filters=num_filters1,kernel_size=ksize,nonlinear=nonlinear1, padding=padding)  


        decoder6_lin = decoderblock_3(encoder6_lin, num_filters=num_filters1,kernel_size=ksize,nonlinear=nonlinear1, padding=padding)  
        decoder6_lin = multi_block_quadraticandsemilinear(decoder6_lin,decoder6_nonlin,channels_out=num_filters1)        
        
        encoder5_lin_crop = layers.Cropping2D(cropping=self.get_cropdim(encoder5_lin,decoder6_lin))(encoder5_lin) 
        skipcon5_lin = layers.concatenate([encoder5_lin_crop, decoder6_lin], axis=3)         
        decoder5_lin = decoderblock_3(skipcon5_lin, num_filters=num_filters1,kernel_size=ksize,nonlinear=nonlinear1, padding=padding)  
        decoder5_lin = multi_block_quadraticandsemilinear(decoder5_lin,decoder5_nonlin,channels_out=num_filters1)

        encoder4_lin_crop = layers.Cropping2D(cropping=self.get_cropdim(encoder4_lin,decoder5_lin))(encoder4_lin) #((top_pad, bottom_pad), (left_pad, right_pad))
        skipcon4_lin = layers.concatenate([encoder4_lin_crop, decoder5_lin], axis=3)     
        decoder4_lin = decoderblock_3(skipcon4_lin, num_filters=num_filters1,kernel_size=ksize,nonlinear=nonlinear1, padding=padding)  
        decoder4_lin = multi_block_quadraticandsemilinear(decoder4_lin,decoder4_nonlin,channels_out=num_filters1)

        encoder3_lin_crop = layers.Cropping2D(cropping=self.get_cropdim(encoder3_lin,decoder4_lin))(encoder3_lin) #((top_pad, bottom_pad), (left_pad, right_pad))
        skipcon3_lin = layers.concatenate([encoder3_lin_crop, decoder4_lin], axis=3) 
        decoder3_lin = decoderblock_3(skipcon3_lin, num_filters=num_filters1,kernel_size=ksize,nonlinear=nonlinear1, padding=padding)  
        decoder3_lin = multi_block_quadraticandsemilinear(decoder3_lin,decoder3_nonlin,channels_out=num_filters1)

        encoder2_lin_crop = layers.Cropping2D(cropping=self.get_cropdim(encoder2_lin,decoder3_lin))(encoder2_lin) #((top_pad, bottom_pad), (left_pad, right_pad))
        skipcon2_lin = layers.concatenate([encoder2_lin_crop, decoder3_lin], axis=3) 
        decoder2_lin = decoderblock_3(skipcon2_lin, num_filters=num_filters1,kernel_size=ksize,nonlinear=nonlinear1, padding=padding) 
        decoder2_lin = multi_block_quadraticandsemilinear(decoder2_lin,decoder2_nonlin,channels_out=num_filters1)

        encoder1_lin_crop = layers.Cropping2D(cropping=self.get_cropdim(encoder1_lin,decoder2_lin))(encoder1_lin)
        skipcon1_lin = layers.concatenate([encoder1_lin_crop, decoder2_lin], axis=3) 
        decoder1_lin = decoderblock_3(skipcon1_lin, num_filters=num_filters1,kernel_size=ksize,nonlinear=nonlinear1, padding=padding)
        decoder1_lin = multi_block_quadraticandsemilinear(decoder1_lin,decoder1_nonlin,channels_out=num_filters1)

        encoder0_lin_crop = layers.Cropping2D(cropping=self.get_cropdim(encoder0_lin,decoder1_lin))(encoder0_lin)
        skipcon0_lin = layers.concatenate([encoder0_lin_crop, decoder1_lin], axis=3) 
        decoder0_lin = layers.Conv2D(filters=channels_out, kernel_size=(ksize, ksize), padding=padding, strides=1)(skipcon0_lin) #needs stride 1
        outputs = decoder0_lin

        return inputs, outputs    
    
    
    
    
    
    
    
    
    
    
    #npad = 62 (for ksize=5)
    #npad = 71 (for ksize=5), ksize_small=11
    def net_linear_nonlinear_doubleU_periodic_k5_5encoders(self,img_shape,channels_out):
        padding = 'valid'
        inputs = layers.Input(shape=img_shape)

        if self.params.wf_mode == "T":
            mapdata = layers.Lambda(lambda x: x[:,:,:,0:1])(inputs)
            maskdata = layers.Lambda(lambda x: x[:,:,:,1:2])(inputs)
        if self.params.wf_mode == "QU":
            mapdata = layers.Lambda(lambda x: x[:,:,:,0:2])(inputs)
            maskdata = layers.Lambda(lambda x: x[:,:,:,2:3])(inputs)

        nonlinear1 = False
        nonlinear2 = True #True
        num_filters1 = 10 #10 1
        num_filters2 = 10 #10 1
        densebridge1 = False #False
        densebridge2 = False #False
        ksize = 5 #ksize 5 -> npad 30
        ksize_small = 5 #ksize
        
        actifunc = self.params.network_actifunc

        ######### non-linear U

        encoder0_nonlin = layers.Conv2D(filters=num_filters2, kernel_size=(ksize_small, ksize_small), padding=padding, strides=1)(maskdata) #needs stride 1
        #print ("E0", encoder0_nonlin)
        if nonlinear2:
            encoder0_nonlin = layers.Activation(activation=actifunc)(encoder0_nonlin) 

        encoder1_nonlin = encoderblock_1(encoder0_nonlin, num_filters=num_filters2,kernel_size=ksize_small,nonlinear=nonlinear2, padding=padding)
        #print ("E1", encoder1_nonlin)   
        encoder2_nonlin = encoderblock_1(encoder1_nonlin, num_filters=num_filters2,kernel_size=ksize,nonlinear=nonlinear2, padding=padding)
        #print ("E2", encoder2_nonlin)    
        encoder3_nonlin = encoderblock_1(encoder2_nonlin, num_filters=num_filters2,kernel_size=ksize,nonlinear=nonlinear2, padding=padding)

        encoder4_nonlin = encoderblock_1(encoder3_nonlin, num_filters=num_filters2,kernel_size=ksize,nonlinear=nonlinear2, padding=padding)    

        #print ("E3", encoder3_nonlin)    
        if densebridge2:
            bridge_nonlin = bridge_sc_1(encoder4_nonlin,num_filters=num_filters2) #with linear layer

            if nonlinear2:
                bridge_nonlin = layers.Activation(activation=actifunc)(bridge_nonlin)    
        else:
            bridge_nonlin = encoder4_nonlin

        decoder4_nonlin = decoderblock_3(bridge_nonlin, num_filters=num_filters2,kernel_size=5,nonlinear=nonlinear2,padding=padding)          

        encoder3_nonlin = layers.Cropping2D(cropping=self.get_cropdim(encoder3_nonlin,decoder4_nonlin))(encoder3_nonlin) 
        skipcon3_nonlin = layers.concatenate([encoder3_nonlin, decoder4_nonlin], axis=3) 
        decoder3_nonlin = decoderblock_3(skipcon3_nonlin, num_filters=num_filters2,kernel_size=5,nonlinear=nonlinear2,padding=padding)         

        #print ("D3", decoder3_nonlin)
        encoder2_nonlin_crop = layers.Cropping2D(cropping=self.get_cropdim(encoder2_nonlin,decoder3_nonlin))(encoder2_nonlin) #((top_pad, bottom_pad), (left_pad, right_pad))
        skipcon2_nonlin = layers.concatenate([encoder2_nonlin_crop, decoder3_nonlin], axis=3) 
        decoder2_nonlin = decoderblock_3(skipcon2_nonlin, num_filters=num_filters2,kernel_size=ksize,nonlinear=nonlinear2, padding=padding) 

        #print ("D2", decoder2_nonlin)    
        encoder1_nonlin_crop = layers.Cropping2D(cropping=self.get_cropdim(encoder1_nonlin,decoder2_nonlin))(encoder1_nonlin)
        skipcon1_nonlin = layers.concatenate([encoder1_nonlin_crop, decoder2_nonlin], axis=3) 
        decoder1_nonlin = decoderblock_3(skipcon1_nonlin, num_filters=num_filters2,kernel_size=ksize_small,nonlinear=nonlinear2, padding=padding)
        #print ("D1", decoder1_nonlin)   



        ######### linear U

        encoder0_lin = layers.Conv2D(filters=num_filters1, kernel_size=(ksize_small, ksize_small), padding=padding, strides=1)(mapdata) #needs stride 1
        encoder0_lin = multi_block_quadraticandlinear(encoder0_lin,encoder0_nonlin,channels_out=num_filters1)

        if nonlinear1:
            encoder0_lin = layers.Activation(activation=actifunc)(encoder0_lin) 

        encoder1_lin = encoderblock_1(encoder0_lin, num_filters=num_filters1,kernel_size=ksize_small,nonlinear=nonlinear1, padding=padding)
        encoder1_lin = multi_block_quadraticandlinear(encoder1_lin,encoder1_nonlin,channels_out=num_filters1)

        encoder2_lin = encoderblock_1(encoder1_lin, num_filters=num_filters1,kernel_size=ksize,nonlinear=nonlinear1, padding=padding)
        encoder2_lin = multi_block_quadraticandlinear(encoder2_lin,encoder2_nonlin,channels_out=num_filters1)

        encoder3_lin = encoderblock_1(encoder2_lin, num_filters=num_filters1,kernel_size=ksize,nonlinear=nonlinear1, padding=padding)

        encoder4_lin = encoderblock_1(encoder3_lin, num_filters=num_filters1,kernel_size=ksize,nonlinear=nonlinear1, padding=padding)       

        if densebridge1:
            bridge_lin = bridge_sc_1(encoder4_lin,num_filters=num_filters1) #with linear layer

            if nonlinear1:
                bridge_lin = layers.Activation(activation=actifunc)(bridge_lin)    
        else:
            bridge_lin = encoder4_lin

        decoder4_lin = decoderblock_3(bridge_lin, num_filters=num_filters1,kernel_size=ksize,nonlinear=nonlinear1, padding=padding)  
        decoder4_lin = multi_block_quadraticandlinear(decoder4_lin,decoder4_nonlin,channels_out=num_filters1)

        encoder3_lin_crop = layers.Cropping2D(cropping=self.get_cropdim(encoder3_lin,decoder4_lin))(encoder3_lin) #((top_pad, bottom_pad), (left_pad, right_pad))
        skipcon3_lin = layers.concatenate([encoder3_lin_crop, decoder4_lin], axis=3) 
        decoder3_lin = decoderblock_3(skipcon3_lin, num_filters=num_filters1,kernel_size=ksize,nonlinear=nonlinear1, padding=padding)  
        decoder3_lin = multi_block_quadraticandlinear(decoder3_lin,decoder3_nonlin,channels_out=num_filters1)

        encoder2_lin_crop = layers.Cropping2D(cropping=self.get_cropdim(encoder2_lin,decoder3_lin))(encoder2_lin) #((top_pad, bottom_pad), (left_pad, right_pad))
        skipcon2_lin = layers.concatenate([encoder2_lin_crop, decoder3_lin], axis=3) 
        decoder2_lin = decoderblock_3(skipcon2_lin, num_filters=num_filters1,kernel_size=ksize,nonlinear=nonlinear1, padding=padding) 
        decoder2_lin = multi_block_quadraticandlinear(decoder2_lin,decoder2_nonlin,channels_out=num_filters1)

        encoder1_lin_crop = layers.Cropping2D(cropping=self.get_cropdim(encoder1_lin,decoder2_lin))(encoder1_lin)
        skipcon1_lin = layers.concatenate([encoder1_lin_crop, decoder2_lin], axis=3) 
        decoder1_lin = decoderblock_3(skipcon1_lin, num_filters=num_filters1,kernel_size=ksize_small,nonlinear=nonlinear1, padding=padding)
        decoder1_lin = multi_block_quadraticandlinear(decoder1_lin,decoder1_nonlin,channels_out=num_filters1)

        encoder0_lin_crop = layers.Cropping2D(cropping=self.get_cropdim(encoder0_lin,decoder1_lin))(encoder0_lin)
        skipcon0_lin = layers.concatenate([encoder0_lin_crop, decoder1_lin], axis=3) 
        decoder0_lin = layers.Conv2D(filters=channels_out, kernel_size=(ksize_small, ksize_small), padding=padding, strides=1)(skipcon0_lin) #needs stride 1

        #decoder0_lin = layers.Cropping2D(cropping=((2,2),(2,2)))(decoder0_lin) #optional

        outputs = decoder0_lin
        return inputs, outputs






    def net_linear_nonlinear_doubleU_periodic_k9_4encoders(self,img_shape,channels_out):
        padding = 'valid'
        inputs = layers.Input(shape=img_shape)

        if self.params.wf_mode == "T":
            mapdata = layers.Lambda(lambda x: x[:,:,:,0:1])(inputs)
            maskdata = layers.Lambda(lambda x: x[:,:,:,1:2])(inputs)
        if self.params.wf_mode == "QU":
            mapdata = layers.Lambda(lambda x: x[:,:,:,0:2])(inputs)
            maskdata = layers.Lambda(lambda x: x[:,:,:,2:3])(inputs)

        nonlinear1 = False
        nonlinear2 = True
        num_filters1 = 10 #10 1
        num_filters2 = 10 #10 1
        densebridge1 = False #False
        densebridge2 = False #False
        ksize = 9 #ksize 9 -> npad 62

        actifunc = self.params.network_actifunc

        ######### non-linear U

        encoder0_nonlin = layers.Conv2D(filters=num_filters2, kernel_size=(ksize, ksize), padding=padding, strides=1)(maskdata) #needs stride 1
        #print ("E0", encoder0_nonlin)
        if nonlinear2:
            encoder0_nonlin = layers.Activation(activation=actifunc)(encoder0_nonlin) 

        encoder1_nonlin = encoderblock_1(encoder0_nonlin, num_filters=num_filters2,kernel_size=ksize,nonlinear=nonlinear2, padding=padding)
        #print ("E1", encoder1_nonlin)   
        encoder2_nonlin = encoderblock_1(encoder1_nonlin, num_filters=num_filters2,kernel_size=ksize,nonlinear=nonlinear2, padding=padding)
        #print ("E2", encoder2_nonlin)    
        encoder3_nonlin = encoderblock_1(encoder2_nonlin, num_filters=num_filters2,kernel_size=ksize,nonlinear=nonlinear2, padding=padding)
        #print ("E3", encoder3_nonlin)    
        if densebridge2:
            bridge_nonlin = bridge_sc_1(encoder3_nonlin,num_filters=num_filters2) #with linear layer

            if nonlinear2:
                bridge_nonlin = layers.Activation(activation=actifunc)(bridge_nonlin)    
        else:
            bridge_nonlin = encoder3_nonlin

        decoder3_nonlin = decoderblock_3(bridge_nonlin, num_filters=num_filters2,kernel_size=ksize,nonlinear=nonlinear2, padding=padding)    
        #print ("D3", decoder3_nonlin)
        encoder2_nonlin_crop = layers.Cropping2D(cropping=((7,8), (7,8)))(encoder2_nonlin) #((top_pad, bottom_pad), (left_pad, right_pad))
        skipcon2_nonlin = layers.concatenate([encoder2_nonlin_crop, decoder3_nonlin], axis=3) 
        decoder2_nonlin = decoderblock_3(skipcon2_nonlin, num_filters=num_filters2,kernel_size=ksize,nonlinear=nonlinear2, padding=padding) 
        #print ("D2", decoder2_nonlin)    
        encoder1_nonlin_crop = layers.Cropping2D(cropping=((23,23), (23,23)))(encoder1_nonlin)
        skipcon1_nonlin = layers.concatenate([encoder1_nonlin_crop, decoder2_nonlin], axis=3) 
        decoder1_nonlin = decoderblock_3(skipcon1_nonlin, num_filters=num_filters2,kernel_size=ksize,nonlinear=nonlinear2, padding=padding)
        #print ("D1", decoder1_nonlin)   



        ######### linear U

        encoder0_lin = layers.Conv2D(filters=num_filters1, kernel_size=(ksize, ksize), padding=padding, strides=1)(mapdata) #needs stride 1
        encoder0_lin = multi_block_quadraticandlinear(encoder0_lin,encoder0_nonlin,channels_out=num_filters1)

        if nonlinear1:
            encoder0_lin = layers.Activation(activation=actifunc)(encoder0_lin) 

        encoder1_lin = encoderblock_1(encoder0_lin, num_filters=num_filters1,kernel_size=ksize,nonlinear=nonlinear1, padding=padding)
        encoder1_lin = multi_block_quadraticandlinear(encoder1_lin,encoder1_nonlin,channels_out=num_filters1)

        encoder2_lin = encoderblock_1(encoder1_lin, num_filters=num_filters1,kernel_size=ksize,nonlinear=nonlinear1, padding=padding)
        encoder2_lin = multi_block_quadraticandlinear(encoder2_lin,encoder2_nonlin,channels_out=num_filters1)

        encoder3_lin = encoderblock_1(encoder2_lin, num_filters=num_filters1,kernel_size=ksize,nonlinear=nonlinear1, padding=padding)

        if densebridge1:
            bridge_lin = bridge_sc_1(encoder3_lin,num_filters=num_filters1) #with linear layer

            if nonlinear1:
                bridge_lin = layers.Activation(activation=actifunc)(bridge_lin)    
        else:
            bridge_lin = encoder3_lin

        decoder3_lin = decoderblock_3(bridge_lin, num_filters=num_filters1,kernel_size=ksize,nonlinear=nonlinear1, padding=padding)  
        decoder3_lin = multi_block_quadraticandlinear(decoder3_lin,decoder3_nonlin,channels_out=num_filters1)

        encoder2_lin_crop = layers.Cropping2D(cropping=((7,8), (7,8)))(encoder2_lin) #((top_pad, bottom_pad), (left_pad, right_pad))
        skipcon2_lin = layers.concatenate([encoder2_lin_crop, decoder3_lin], axis=3) 
        decoder2_lin = decoderblock_3(skipcon2_lin, num_filters=num_filters1,kernel_size=ksize,nonlinear=nonlinear1, padding=padding) 
        decoder2_lin = multi_block_quadraticandlinear(decoder2_lin,decoder2_nonlin,channels_out=num_filters1)

        encoder1_lin_crop = layers.Cropping2D(cropping=((23,23), (23,23)))(encoder1_lin)
        skipcon1_lin = layers.concatenate([encoder1_lin_crop, decoder2_lin], axis=3) 
        decoder1_lin = decoderblock_3(skipcon1_lin, num_filters=num_filters1,kernel_size=ksize,nonlinear=nonlinear1, padding=padding)
        decoder1_lin = multi_block_quadraticandlinear(decoder1_lin,decoder1_nonlin,channels_out=num_filters1)

        encoder0_lin_crop = layers.Cropping2D(cropping=((54,54), (54,54)))(encoder0_lin)
        skipcon0_lin = layers.concatenate([encoder0_lin_crop, decoder1_lin], axis=3) 
        decoder0_lin = layers.Conv2D(filters=channels_out, kernel_size=(ksize, ksize), padding=padding, strides=1)(skipcon0_lin) #needs stride 1
        outputs = decoder0_lin


        return inputs, outputs




    def net_linear_nonlinear_doubleU_periodic_k5_4encoders(self,img_shape,channels_out):
        padding = 'valid'
        inputs = layers.Input(shape=img_shape)

        if self.params.wf_mode == "T":
            mapdata = layers.Lambda(lambda x: x[:,:,:,0:1])(inputs)
            maskdata = layers.Lambda(lambda x: x[:,:,:,1:2])(inputs)
        if self.params.wf_mode == "QU":
            mapdata = layers.Lambda(lambda x: x[:,:,:,0:2])(inputs)
            maskdata = layers.Lambda(lambda x: x[:,:,:,2:3])(inputs)

        nonlinear1 = False
        nonlinear2 = True
        num_filters1 = 10 #10 1
        num_filters2 = 10 #10 1
        densebridge1 = False #False
        densebridge2 = False #False
        ksize = 5 #ksize 5 -> npad 30

        actifunc = self.params.network_actifunc

        ######### non-linear U

        encoder0_nonlin = layers.Conv2D(filters=num_filters2, kernel_size=(ksize, ksize), padding=padding, strides=1)(maskdata) #needs stride 1
        #print ("E0", encoder0_nonlin)
        if nonlinear2:
            encoder0_nonlin = layers.Activation(activation=actifunc)(encoder0_nonlin) 

        encoder1_nonlin = encoderblock_1(encoder0_nonlin, num_filters=num_filters2,kernel_size=ksize,nonlinear=nonlinear2, padding=padding)
        #print ("E1", encoder1_nonlin)   
        encoder2_nonlin = encoderblock_1(encoder1_nonlin, num_filters=num_filters2,kernel_size=ksize,nonlinear=nonlinear2, padding=padding)
        #print ("E2", encoder2_nonlin)    
        encoder3_nonlin = encoderblock_1(encoder2_nonlin, num_filters=num_filters2,kernel_size=ksize,nonlinear=nonlinear2, padding=padding)
        #print ("E3", encoder3_nonlin)    
        if densebridge2:
            bridge_nonlin = bridge_sc_1(encoder3_nonlin,num_filters=num_filters2) #with linear layer

            if nonlinear2:
                bridge_nonlin = layers.Activation(activation=actifunc)(bridge_nonlin)    
        else:
            bridge_nonlin = encoder3_nonlin

        decoder3_nonlin = decoderblock_3(bridge_nonlin, num_filters=num_filters2,kernel_size=ksize,nonlinear=nonlinear2, padding=padding)    
        #print ("D3", decoder3_nonlin)
        encoder2_nonlin_crop = layers.Cropping2D(cropping=((3,4), (3,4)))(encoder2_nonlin) #((top_pad, bottom_pad), (left_pad, right_pad))
        skipcon2_nonlin = layers.concatenate([encoder2_nonlin_crop, decoder3_nonlin], axis=3) 
        decoder2_nonlin = decoderblock_3(skipcon2_nonlin, num_filters=num_filters2,kernel_size=ksize,nonlinear=nonlinear2, padding=padding) 
        #print ("D2", decoder2_nonlin)    
        encoder1_nonlin_crop = layers.Cropping2D(cropping=((11,11), (11,11)))(encoder1_nonlin)
        skipcon1_nonlin = layers.concatenate([encoder1_nonlin_crop, decoder2_nonlin], axis=3) 
        decoder1_nonlin = decoderblock_3(skipcon1_nonlin, num_filters=num_filters2,kernel_size=ksize,nonlinear=nonlinear2, padding=padding)
        #print ("D1", decoder1_nonlin)   



        ######### linear U

        encoder0_lin = layers.Conv2D(filters=num_filters1, kernel_size=(ksize, ksize), padding=padding, strides=1)(mapdata) #needs stride 1
        encoder0_lin = multi_block_quadraticandlinear(encoder0_lin,encoder0_nonlin,channels_out=num_filters1)

        if nonlinear1:
            encoder0_lin = layers.Activation(activation=actifunc)(encoder0_lin) 

        encoder1_lin = encoderblock_1(encoder0_lin, num_filters=num_filters1,kernel_size=ksize,nonlinear=nonlinear1, padding=padding)
        encoder1_lin = multi_block_quadraticandlinear(encoder1_lin,encoder1_nonlin,channels_out=num_filters1)

        encoder2_lin = encoderblock_1(encoder1_lin, num_filters=num_filters1,kernel_size=ksize,nonlinear=nonlinear1, padding=padding)
        encoder2_lin = multi_block_quadraticandlinear(encoder2_lin,encoder2_nonlin,channels_out=num_filters1)

        encoder3_lin = encoderblock_1(encoder2_lin, num_filters=num_filters1,kernel_size=ksize,nonlinear=nonlinear1, padding=padding)

        if densebridge1:
            bridge_lin = bridge_sc_1(encoder3_lin,num_filters=num_filters1) #with linear layer

            if nonlinear1:
                bridge_lin = layers.Activation(activation=actifunc)(bridge_lin)    
        else:
            bridge_lin = encoder3_lin

        decoder3_lin = decoderblock_3(bridge_lin, num_filters=num_filters1,kernel_size=ksize,nonlinear=nonlinear1, padding=padding)  
        decoder3_lin = multi_block_quadraticandlinear(decoder3_lin,decoder3_nonlin,channels_out=num_filters1)

        encoder2_lin_crop = layers.Cropping2D(cropping=((3,4), (3,4)))(encoder2_lin) #((top_pad, bottom_pad), (left_pad, right_pad))
        skipcon2_lin = layers.concatenate([encoder2_lin_crop, decoder3_lin], axis=3) 
        decoder2_lin = decoderblock_3(skipcon2_lin, num_filters=num_filters1,kernel_size=ksize,nonlinear=nonlinear1, padding=padding) 
        decoder2_lin = multi_block_quadraticandlinear(decoder2_lin,decoder2_nonlin,channels_out=num_filters1)

        encoder1_lin_crop = layers.Cropping2D(cropping=((11,11), (11,11)))(encoder1_lin)
        skipcon1_lin = layers.concatenate([encoder1_lin_crop, decoder2_lin], axis=3) 
        decoder1_lin = decoderblock_3(skipcon1_lin, num_filters=num_filters1,kernel_size=ksize,nonlinear=nonlinear1, padding=padding)
        decoder1_lin = multi_block_quadraticandlinear(decoder1_lin,decoder1_nonlin,channels_out=num_filters1)

        encoder0_lin_crop = layers.Cropping2D(cropping=((26,26), (26,26)))(encoder0_lin)
        skipcon0_lin = layers.concatenate([encoder0_lin_crop, decoder1_lin], axis=3) 
        decoder0_lin = layers.Conv2D(filters=channels_out, kernel_size=(ksize, ksize), padding=padding, strides=1)(skipcon0_lin) #needs stride 1
        outputs = decoder0_lin


        return inputs, outputs






    def net_linear_nonlinear_doubleU_convtrans(self,img_shape,channels_out):
        padding = 'valid'
        inputs = layers.Input(shape=img_shape)

        if self.params.wf_mode == "T":
            mapdata = layers.Lambda(lambda x: x[:,:,:,0:1])(inputs)
            maskdata = layers.Lambda(lambda x: x[:,:,:,1:2])(inputs)
        if self.params.wf_mode == "QU":
            mapdata = layers.Lambda(lambda x: x[:,:,:,0:2])(inputs)
            maskdata = layers.Lambda(lambda x: x[:,:,:,2:3])(inputs)

        nonlinear1 = False
        nonlinear2 = True
        num_filters1 = 10 #10 1
        num_filters2 = 10 #10 1
        densebridge1 = False #False
        densebridge2 = False #False

        actifunc = self.params.network_actifunc

        ######### non-linear U

        encoder0_nonlin = layers.Conv2D(filters=num_filters2, kernel_size=(5, 5), padding=padding, strides=1)(maskdata) #needs stride 1

        if nonlinear2:
            encoder0_nonlin = layers.Activation(activation=actifunc)(encoder0_nonlin) 

        encoder1_nonlin = encoderblock_1(encoder0_nonlin, num_filters=num_filters2,kernel_size=5,nonlinear=nonlinear2, padding=padding)

        encoder2_nonlin = encoderblock_1(encoder1_nonlin, num_filters=num_filters2,kernel_size=5,nonlinear=nonlinear2, padding=padding)

        encoder3_nonlin = encoderblock_1(encoder2_nonlin, num_filters=num_filters2,kernel_size=5,nonlinear=nonlinear2, padding=padding)

        if densebridge2:
            bridge_nonlin = bridge_sc_1(encoder3_nonlin,num_filters=num_filters2) #with linear layer

            if nonlinear2:
                bridge_nonlin = layers.Activation(activation=actifunc)(bridge_nonlin)    
        else:
            bridge_nonlin = encoder3_nonlin

        decoder3_nonlin = decoderblock_2(bridge_nonlin, num_filters=num_filters2,kernel_size=6,nonlinear=nonlinear2, padding=padding)    

        skipcon2_nonlin = layers.concatenate([encoder2_nonlin, decoder3_nonlin], axis=3) 
        decoder2_nonlin = decoderblock_2(skipcon2_nonlin, num_filters=num_filters2,kernel_size=6,nonlinear=nonlinear2, padding=padding) 

        skipcon1_nonlin = layers.concatenate([encoder1_nonlin, decoder2_nonlin], axis=3) 
        decoder1_nonlin = decoderblock_2(skipcon1_nonlin, num_filters=num_filters2,kernel_size=6,nonlinear=nonlinear2, padding=padding)

        #skipcon0_nonlin = layers.concatenate([encoder0_nonlin, decoder1_nonlin], axis=3) 
        #decoder0_nonlin = layers.Conv2D(filters=channels_out2, kernel_size=(16, 16), padding=padding, strides=1)(skipcon0_nonlin) #needs stride 1



        ######### linear U

        encoder0_lin = layers.Conv2D(filters=num_filters1, kernel_size=(5, 5), padding=padding, strides=1)(mapdata) #needs stride 1
        encoder0_lin = multi_block_quadraticandlinear(encoder0_lin,encoder0_nonlin,channels_out=num_filters1)

        if nonlinear1:
            encoder0_lin = layers.Activation(activation=actifunc)(encoder0_lin) 

        encoder1_lin = encoderblock_1(encoder0_lin, num_filters=num_filters1,kernel_size=5,nonlinear=nonlinear1, padding=padding)
        encoder1_lin = multi_block_quadraticandlinear(encoder1_lin,encoder1_nonlin,channels_out=num_filters1)

        encoder2_lin = encoderblock_1(encoder1_lin, num_filters=num_filters1,kernel_size=5,nonlinear=nonlinear1, padding=padding)
        encoder2_lin = multi_block_quadraticandlinear(encoder2_lin,encoder2_nonlin,channels_out=num_filters1)

        encoder3_lin = encoderblock_1(encoder2_lin, num_filters=num_filters1,kernel_size=5,nonlinear=nonlinear1, padding=padding)

        if densebridge1:
            bridge_lin = bridge_sc_1(encoder3_lin,num_filters=num_filters1) #with linear layer

            if nonlinear1:
                bridge_lin = layers.Activation(activation=actifunc)(bridge_lin)    
        else:
            bridge_lin = encoder3_lin

        decoder3_lin = decoderblock_2(bridge_lin, num_filters=num_filters1,kernel_size=6,nonlinear=nonlinear1, padding=padding)  
        decoder3_lin = multi_block_quadraticandlinear(decoder3_lin,decoder3_nonlin,channels_out=num_filters1)

        skipcon2_lin = layers.concatenate([encoder2_lin, decoder3_lin], axis=3) 
        decoder2_lin = decoderblock_2(skipcon2_lin, num_filters=num_filters1,kernel_size=6,nonlinear=nonlinear1, padding=padding) 
        decoder2_lin = multi_block_quadraticandlinear(decoder2_lin,decoder2_nonlin,channels_out=num_filters1)

        skipcon1_lin = layers.concatenate([encoder1_lin, decoder2_lin], axis=3) 
        decoder1_lin = decoderblock_2(skipcon1_lin, num_filters=num_filters1,kernel_size=6,nonlinear=nonlinear1, padding=padding)
        decoder1_lin = multi_block_quadraticandlinear(decoder1_lin,decoder1_nonlin,channels_out=num_filters1)

        skipcon0_lin = layers.concatenate([encoder0_lin, decoder1_lin], axis=3) 
        decoder0_lin = layers.Conv2D(filters=channels_out, kernel_size=(5, 5), padding=padding, strides=1)(skipcon0_lin) #needs stride 1
        outputs = decoder0_lin


        return inputs, outputs





    def net_linear_nonlinear_doubleU(self,img_shape,channels_out):
        padding = 'same'
        inputs = layers.Input(shape=img_shape)

        if self.params.wf_mode == "T":
            mapdata = layers.Lambda(lambda x: x[:,:,:,0:1])(inputs)
            maskdata = layers.Lambda(lambda x: x[:,:,:,1:2])(inputs)
        if self.params.wf_mode == "QU":
            mapdata = layers.Lambda(lambda x: x[:,:,:,0:2])(inputs)
            maskdata = layers.Lambda(lambda x: x[:,:,:,2:3])(inputs)

        nonlinear1 = False
        nonlinear2 = True
        num_filters1 = 10 #10 1
        num_filters2 = 10 #10 1
        densebridge1 = False #False
        densebridge2 = False #False

        klarge = 5 #16
        ksmall = 5 #8

        actifunc = self.params.network_actifunc

        ######### non-linear U

        encoder0_nonlin = layers.Conv2D(filters=num_filters2, kernel_size=(klarge, klarge), padding=padding, strides=1)(maskdata) #needs stride 1

        if nonlinear2:
            encoder0_nonlin = layers.Activation(activation=actifunc)(encoder0_nonlin) 

        encoder1_nonlin = encoderblock_1(encoder0_nonlin, num_filters=num_filters2,kernel_size=klarge,nonlinear=nonlinear2,padding=padding)

        encoder2_nonlin = encoderblock_1(encoder1_nonlin, num_filters=num_filters2,kernel_size=ksmall,nonlinear=nonlinear2,padding=padding)

        encoder3_nonlin = encoderblock_1(encoder2_nonlin, num_filters=num_filters2,kernel_size=ksmall,nonlinear=nonlinear2,padding=padding)

        if densebridge2:
            bridge_nonlin = bridge_sc_1(encoder3_nonlin,num_filters=num_filters2) #with linear layer

            if nonlinear2:
                bridge_nonlin = layers.Activation(activation=actifunc)(bridge_nonlin)    
        else:
            bridge_nonlin = encoder3_nonlin

        decoder3_nonlin = decoderblock_1(bridge_nonlin, num_filters=num_filters2,kernel_size=ksmall,nonlinear=nonlinear2,padding=padding)    

        skipcon2_nonlin = layers.concatenate([encoder2_nonlin, decoder3_nonlin], axis=3) 
        decoder2_nonlin = decoderblock_1(skipcon2_nonlin, num_filters=num_filters2,kernel_size=ksmall,nonlinear=nonlinear2,padding=padding) 

        skipcon1_nonlin = layers.concatenate([encoder1_nonlin, decoder2_nonlin], axis=3) 
        decoder1_nonlin = decoderblock_1(skipcon1_nonlin, num_filters=num_filters2,kernel_size=klarge,nonlinear=nonlinear2,padding=padding)

        #skipcon0_nonlin = layers.concatenate([encoder0_nonlin, decoder1_nonlin], axis=3) 
        #decoder0_nonlin = layers.Conv2D(filters=channels_out2, kernel_size=(16, 16), padding=padding, strides=1)(skipcon0_nonlin) #needs stride 1



        ######### linear U

        encoder0_lin = layers.Conv2D(filters=num_filters1, kernel_size=(klarge, klarge), padding=padding, strides=1)(mapdata) #needs stride 1
        encoder0_lin = multi_block_quadraticandlinear(encoder0_lin,encoder0_nonlin,channels_out=num_filters1)

        if nonlinear1:
            encoder0_lin = layers.Activation(activation=actifunc)(encoder0_lin) 

        encoder1_lin = encoderblock_1(encoder0_lin, num_filters=num_filters1,kernel_size=klarge,nonlinear=nonlinear1,padding=padding)
        encoder1_lin = multi_block_quadraticandlinear(encoder1_lin,encoder1_nonlin,channels_out=num_filters1)

        encoder2_lin = encoderblock_1(encoder1_lin, num_filters=num_filters1,kernel_size=ksmall,nonlinear=nonlinear1,padding=padding)
        encoder2_lin = multi_block_quadraticandlinear(encoder2_lin,encoder2_nonlin,channels_out=num_filters1)

        encoder3_lin = encoderblock_1(encoder2_lin, num_filters=num_filters1,kernel_size=ksmall,nonlinear=nonlinear1,padding=padding)

        if densebridge1:
            bridge_lin = bridge_sc_1(encoder3_lin,num_filters=num_filters1) #with linear layer

            if nonlinear1:
                bridge_lin = layers.Activation(activation=actifunc)(bridge_lin)    
        else:
            bridge_lin = encoder3_lin

        decoder3_lin = decoderblock_1(bridge_lin, num_filters=num_filters1,kernel_size=ksmall,nonlinear=nonlinear1,padding=padding)  
        decoder3_lin = multi_block_quadraticandlinear(decoder3_lin,decoder3_nonlin,channels_out=num_filters1)

        skipcon2_lin = layers.concatenate([encoder2_lin, decoder3_lin], axis=3) 
        decoder2_lin = decoderblock_1(skipcon2_lin, num_filters=num_filters1,kernel_size=ksmall,nonlinear=nonlinear1,padding=padding) 
        decoder2_lin = multi_block_quadraticandlinear(decoder2_lin,decoder2_nonlin,channels_out=num_filters1)

        skipcon1_lin = layers.concatenate([encoder1_lin, decoder2_lin], axis=3) 
        decoder1_lin = decoderblock_1(skipcon1_lin, num_filters=num_filters1,kernel_size=klarge,nonlinear=nonlinear1,padding=padding)
        decoder1_lin = multi_block_quadraticandlinear(decoder1_lin,decoder1_nonlin,channels_out=num_filters1)

        skipcon0_lin = layers.concatenate([encoder0_lin, decoder1_lin], axis=3) 
        decoder0_lin = layers.Conv2D(filters=channels_out, kernel_size=(klarge, klarge), padding=padding, strides=1)(skipcon0_lin) #needs stride 1
        outputs = decoder0_lin


        return inputs, outputs





    def net_linear_nonlinear_simple(self,img_shape,channels_out):
        num_filters = 10 #1

        inputs = layers.Input(shape=img_shape)

        if self.params.wf_mode == "T":
            mapdata = layers.Lambda(lambda x: x[:,:,:,0:1])(inputs)
            maskdata = layers.Lambda(lambda x: x[:,:,:,1:2])(inputs)
        if self.params.wf_mode == "QU":
            mapdata = layers.Lambda(lambda x: x[:,:,:,0:2])(inputs)
            maskdata = layers.Lambda(lambda x: x[:,:,:,2:3])(inputs)

        term1 = linear_block_1(mapdata,channels_out=1,num_filters=num_filters,nonlinear=False)

        term2 = linear_block_1(maskdata,channels_out=1,num_filters=num_filters,nonlinear=True)

        #multi = layers.multiply([term1, term2])
        #multi = multi_block_quadratic(term1,term2,channels_out=channels_out)
        multi = multi_block_quadraticandlinear(term1,term2,channels_out=channels_out)  

        #outputs = layers.Conv2D(filters=channels_out, kernel_size=(8, 8), padding=padding, strides=1)(multi)
        outputs = multi

        return inputs,outputs 














###################### Multiplication blocks





#make quadratic local product. y_i = M^i_jk input1_j input2_k
def multi_block_quadratic(input1,input2,channels_out):    
    #implementation:
    #- split both inputs in channels
    #- multiply each pair and concat -> N*M layers
    #(could also concat the linear terms -> N*M+N+M layers, but we want it explicitly quadratic here)
    #now use a 1x1 convo on them to scale down to some wanted channel number

    #print ("TEST1",input1)
    
    channelnr_1 = input1.get_shape().as_list()[-1]
    channelnr_2 = input2.get_shape().as_list()[-1]
    
    #split channels
    #https://github.com/keras-team/keras/issues/5474
    channels_1 = []
    for i in range(channelnr_1):
        chan = layers.Lambda(lambda x: x[:,:,:,i:i+1])(input1)
        channels_1.append(chan)    
    channels_2 = []
    for i in range(channelnr_2):
        chan = layers.Lambda(lambda x: x[:,:,:,i:i+1])(input2)
        channels_2.append(chan)     
   
    #multiply and concat
    channels_multi = []
    for chan1 in channels_1:
        for chan2 in channels_2:
            multi = layers.Multiply()([chan1,chan2])
            channels_multi.append(multi)
    #print ("TEST2",multi) 
    if (len(channels_multi)>1):
        multilayer = layers.Concatenate()(channels_multi)    
    else:
        multilayer = channels_multi[0]  
        
    #print ("TEST3",multilayer)   
    
    #now 1x1 convo this
    outputs = layers.Conv2D(filters=channels_out, kernel_size=(1, 1), padding='same', strides=1)(multilayer) 
    
    return outputs




#make quadratic local product. y_i = M^i_jk input1_j input2_k + a_n input1_n + a_m input1_m
def multi_block_quadraticandlinear(input1,input2,channels_out):    
    #implementation:
    #- split both inputs in channels
    #- multiply each pair and concat -> N*M layers
    #- also concat the linear terms -> N*M+N+M layers
    #now use a 1x1 convo on them to scale down to some wanted channel number
    
    channelnr_1 = input1.get_shape().as_list()[-1]
    channelnr_2 = input2.get_shape().as_list()[-1]
    
    #split channels
    #https://github.com/keras-team/keras/issues/5474
    channels_1 = []
    for i in range(channelnr_1):
        chan = layers.Lambda(lambda x: x[:,:,:,i:i+1])(input1)
        channels_1.append(chan)    
    channels_2 = []
    for i in range(channelnr_2):
        chan = layers.Lambda(lambda x: x[:,:,:,i:i+1])(input2)
        channels_2.append(chan)     
   
    #multiply and concat
    channels_multi = []
    for chan1 in channels_1:
        for chan2 in channels_2:
            multi = layers.Multiply()([chan1,chan2])
            channels_multi.append(multi)
    
    #also add linear channels
    for chan1 in channels_1:
        channels_multi.append(chan1)
    for chan2 in channels_2:
        channels_multi.append(chan2)
    
    if (len(channels_multi)>1):
        multilayer = layers.Concatenate()(channels_multi)    
    else:
        multilayer = channels_multi[0] 
    
    #now 1x1 convo this
    outputs = layers.Conv2D(filters=channels_out, kernel_size=(1, 1), padding='same', strides=1)(multilayer) 
    
    return outputs








#make quadratic local product. y_i = M^i_jk input1_j input2_k + a_n input1_n 
def multi_block_quadraticandsemilinear(input1,input2,channels_out):    
    #implementation:
    #- split both inputs in channels
    #- multiply each pair and concat -> N*M layers
    #- also concat the linear terms -> N*M+N+M layers
    #now use a 1x1 convo on them to scale down to some wanted channel number
    
    channelnr_1 = input1.get_shape().as_list()[-1]
    channelnr_2 = input2.get_shape().as_list()[-1]
    
    #split channels
    #https://github.com/keras-team/keras/issues/5474
    channels_1 = []
    for i in range(channelnr_1):
        chan = layers.Lambda(lambda x: x[:,:,:,i:i+1])(input1)
        channels_1.append(chan)    
    channels_2 = []
    for i in range(channelnr_2):
        chan = layers.Lambda(lambda x: x[:,:,:,i:i+1])(input2)
        channels_2.append(chan)     
   
    #multiply and concat
    channels_multi = []
    for chan1 in channels_1:
        for chan2 in channels_2:
            multi = layers.Multiply()([chan1,chan2])
            channels_multi.append(multi)
    
    #also add linear channels
    for chan1 in channels_1:
        channels_multi.append(chan1)
    
    if (len(channels_multi)>1):
        multilayer = layers.Concatenate()(channels_multi)    
    else:
        multilayer = channels_multi[0] 
    
    #now 1x1 convo this
    outputs = layers.Conv2D(filters=channels_out, kernel_size=(1, 1), padding='same', strides=1)(multilayer) 
    
    return outputs






###################### Quadradic network



def net_quadratic_1(img_shape,channels_out):

    num_filters=10
    densebridge=False
    
    inputs = layers.Input(shape=img_shape)
    
    term1 = linear_block_2(inputs,channels_out=num_filters,num_filters=num_filters,densebridge=densebridge)
    
    term2 = linear_block_2(inputs,channels_out=num_filters,num_filters=num_filters,densebridge=densebridge)
    
    #term3 = linear_block_1(inputs,channels_out,num_filters=num_filters,densebridge=densebridge)
    
    #multi = multi_block(term1,term2) #allow tensor multiplication coefficients.
    multi = layers.multiply([term1, term2]) #just multiply per channel
    
    #concat = layers.concatenate([multi, term3], axis=3)
    
    #outputs = linear_block_2(multi,channels_out=channels_out,num_filters=num_filters,densebridge=densebridge) #does not seem to work?

    outputs = layers.Conv2D(filters=channels_out, kernel_size=(16, 16), padding=padding, strides=1)(multi) #GOOD
    #outputs = layers.Conv2D(filters=channels_out, kernel_size=(5, 5), padding=padding, strides=1)(multi) #ok
    #outputs = layers.Conv2D(filters=channels_out, kernel_size=(1, 1), padding=padding, strides=1)(multi) #not good
    
    return inputs,outputs




    
    
    

###################### experimental networks in Fourier space


def net_ftonly_quadratic_1(img_shape,channels_out):
    
    inputs = layers.Input(shape=img_shape)
    
    term1 = ft_block_1(inputs,channels_out,img_shape)
    
    term2 = ft_block_1(inputs,channels_out,img_shape)
    
    multi = layers.multiply([term1, term2])
    
    outputs = layers.Conv2D(filters=channels_out, kernel_size=(16, 16), padding=padding, strides=1)(multi)
   
    return inputs,outputs


def net_ft_then_real_quadratic_1(img_shape,channels_out):
    print ("IMG", img_shape)
    inputs = layers.Input(shape=img_shape)
    
    term1 = ft_block_1(inputs,channels_out=2,img_shape=img_shape)
    print ("FT TEST", term1)
    term1 = unet_block_2(term1,channels_out=2)
    
    term2 = ft_block_1(inputs,channels_out=2,img_shape=img_shape)
    term2 = unet_block_2(term2,channels_out=2)
    
    multi = layers.multiply([term1, term2])
    
    outputs = layers.Conv2D(filters=channels_out, kernel_size=(16, 16), padding=padding, strides=1)(multi)
   
    return inputs,outputs


def net_real_then_ft_quadratic_1(img_shape,channels_out):
    
    inputs = layers.Input(shape=img_shape)
    
    term1 = unet_block_2(inputs,channels_out=2)
    term1 = ft_block_1(term1,channels_out,img_shape=img_shape)
    
    term2 = unet_block_2(inputs,channels_out=2)
    term2 = ft_block_1(term2,channels_out,img_shape=img_shape)
    
    multi = layers.multiply([term1, term2])
    
    outputs = layers.Conv2D(filters=channels_out, kernel_size=(16, 16), padding=padding, strides=1)(multi)
   
    return inputs,outputs


#FT and de-FT layer



def ft_block_1(inputs,channels_out,img_shape):
    densebridge = True
    
    print ("IMG", img_shape)
    print ("TEST0",inputs)
    
    ################ fft channel wise
    
    #https://stackoverflow.com/questions/50701913/how-to-split-the-input-into-different-channels-in-keras
    fft_channels = []
    for i in range(img_shape[-1]):
        #https://stackoverflow.com/questions/49616081/use-tensorflow-operation-in-tf-keras-model
        #https://www.tensorflow.org/api_docs/python/tf/keras/layers/Lambda
        #https://www.tensorflow.org/api_docs/python/tf/spectral/rfft2d
        #fft_chan = layers.Lambda(lambda x: tf.spectral.rfft2d)(inputs[:,:,i])
        fft_chan = layers.Lambda(lambda x: tf.expand_dims(tf.spectral.rfft2d(x[:,:,:,i]) ,axis=-1))(inputs)
        #https://github.com/titu1994/Keras-DualPathNetworks/issues/3 it seems one needs to give output shape to Lambda here if it is not the same as input shape.
        print ("TEST a",fft_chan)
        fft_channels.append(fft_chan)

    # Concatenating together the per-channel results:
    if (len(fft_channels)>1):
        fftout = layers.Concatenate()(fft_channels) #seems one cannot concat lambda layers
        #https://stackoverflow.com/questions/53376996/cannot-concatenate-keras-lambda-layers
        #fftout =  layers.Lambda(lambda x: K.stack(x))(fft_channels)
    else:
        fftout = fft_channels[0]
    
    print ("TEST1",fftout)
    
    ################ operate on FFT
    
    #dealing with the comples numbers
    #https://github.com/tensorflow/tensorflow/issues/2255
    #what would be a good rep? real and img independent? or something with phase vs amp, eg keep amp the same?
    
    #first version: split in real and img and double the channel number
    fftout_real =  layers.Lambda(lambda x: tf.real(x))(fftout)
    fftout_img =  layers.Lambda(lambda x: tf.imag(x))(fftout)
    
    #now do something on these spectra
    channels = fftout_real.get_shape().as_list()[-1]
    fftout_real = ft_subblock_1(fftout_real,channels,num_filters=channels,densebridge=densebridge) 
    fftout_img = ft_subblock_1(fftout_img,channels,num_filters=channels,densebridge=densebridge) 
    
    #fftout_real = layers.Conv2D(filters=img_shape[-1], kernel_size=(16, 16), padding='same', strides=1)(fftout_real)
    #fftout_img = layers.Conv2D(filters=img_shape[-1], kernel_size=(16, 16), padding='same', strides=1)(fftout_img)
    
    #now make a complex number again
    fftout_complex =  layers.Lambda(lambda x: tf.complex(x[0],x[1]))([fftout_real,fftout_img])
    
    print ("TEST 2",fftout)
    
    ################ undo the FFT
    
    #ifft channel wise
    ifft_channels = []
    for i in range(img_shape[-1]):
        ifft_chan = layers.Lambda(lambda x: tf.expand_dims(tf.spectral.irfft2d(x[:,:,:,i]), axis=-1))(fftout_complex)
        print ("TEST b",ifft_chan)
        ifft_channels.append(ifft_chan)

    # Concatenating together the per-channel results:
    if (len(ifft_channels)>1):
        ifftout = layers.Concatenate()(ifft_channels)
        #ifftout =  layers.Lambda(lambda x: K.stack(x))(ifft_channels)
    else:
        ifftout = ifft_channels[0]    
    print ("TEST3",ifftout)
    
    #we need to compress to the number of output channels. use 1x1 to not introduce real space dofs.
    outputs = layers.Conv2D(filters=channels_out, kernel_size=(1, 1), padding=padding, strides=1)(ifftout)
   
    return outputs



def ft_subblock_1(inputs,channels_out,num_filters=2,densebridge=False):
    
    encoder0 = layers.Conv2D(filters=num_filters, kernel_size=(16, 16), padding='same', strides=1)(inputs) #needs stride 1
    
    encoder1 = encoderblock_1(encoder0, num_filters=num_filters,kernel_size=16)
   
    encoder2 = encoderblock_1(encoder1, num_filters=num_filters,kernel_size=8)
    
    encoder3 = encoderblock_1(encoder2, num_filters=num_filters,kernel_size=8)
    
    if densebridge:
        bridge = bridge_sc_1(encoder3,num_filters=num_filters) #with linear layer
    else:
        bridge = encoder3   

    decoder3 = decoderblock_1(bridge, num_filters=num_filters)    
     
    crop2 = layers.Cropping2D(((0,0),(0,1)))(decoder3)
    skipcon2 = layers.concatenate([encoder2, crop2], axis=3) 
    decoder2 = decoderblock_1(skipcon2, num_filters=num_filters,kernel_size=8) 
    
    crop1 = layers.Cropping2D(((0,0),(0,1)))(decoder2)
    skipcon1 = layers.concatenate([encoder1, crop1], axis=3) 
    decoder1 = decoderblock_1(skipcon1, num_filters=num_filters,kernel_size=16)
    
    crop0 = layers.Cropping2D(((0,0),(0,1)))(decoder1)
    skipcon0 = layers.concatenate([encoder0, crop0], axis=3) 
    decoder0 = layers.Conv2D(filters=channels_out, kernel_size=(16, 16), padding='same', strides=1)(skipcon0) #needs stride 1
    
    return decoder0




def net_ft_linear_1(img_shape,channels_out):
    
    inputs = layers.Input(shape=img_shape)
    
    outputs = ft_block_1(inputs,channels_out,img_shape)
   
    return inputs,outputs









###################### Scale filter


#single conv
#reduces dimention by factor 2
def encoderblock_1(input_downstream, num_filters, kernel_size=3,nonlinear=False,padding='same',activation='relu'):
    #input_downstream = layers.BatchNormalization()(input_downstream) #BN
    
    conv = layers.Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), padding=padding, strides=2)(input_downstream)
    
    if nonlinear:
        conv = layers.Activation(activation=activation)(conv)
        
    #conv = layers.BatchNormalization()(conv) #BN
    
    #conv = layers.Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), padding=padding, strides=1)(conv) #TEST
    return conv
    

def encoderblock_2(input_downstream, num_filters, kernel_size=3,nonlinear=False,padding='same',activation='relu'):
    #input_downstream = layers.BatchNormalization()(input_downstream) #BN
    
    conv = layers.Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), padding=padding, strides=2)(input_downstream)
    
    if nonlinear:
        conv = layers.Activation(activation=activation)(conv)
        
    #conv = layers.BatchNormalization()(conv) #BN
    
    conv = layers.Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), padding="same", strides=1)(conv) 

    if nonlinear:
        conv = layers.Activation(activation=activation)(conv)
    
    return conv    
    
    
#single conv  
#increases dimension by factor 2    
def decoderblock_1(input_upstream, num_filters, kernel_size=3,nonlinear=False,padding='same',activation='relu'): 
    #input_upstream = layers.BatchNormalization()(input_upstream) #BN
    
    conv = layers.Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), padding=padding, strides=1)(input_upstream)
    #conv = layers.Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), padding=padding, strides=1)(conv) #TEST
    
    if nonlinear:
        conv = layers.Activation(activation=activation)(conv)    
    
    upsamp = layers.UpSampling2D(size=(2, 2))(conv)
    
    #upsamp = layers.BatchNormalization()(upsamp) #BN
    return upsamp


#single conv  
#increases dimension by factor 2    
def decoderblock_2(input_upstream, num_filters, kernel_size=3,nonlinear=False,padding='same',activation='relu'): 
    
    conv = layers.Conv2DTranspose(filters=num_filters, kernel_size=(kernel_size, kernel_size), padding=padding, strides=2)(input_upstream)
    
    if nonlinear:
        conv = layers.Activation(activation=activation)(conv)    
    
    return conv



def decoderblock_3(input_upstream, num_filters, kernel_size=3,nonlinear=False,padding='same',activation='relu'): 
    upsamp = layers.UpSampling2D(size=(2, 2))(input_upstream)
    
    conv = layers.Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), padding=padding, strides=1)(upsamp)
    
    if nonlinear:
        conv = layers.Activation(activation=activation)(conv)    
        
    #upsamp = layers.BatchNormalization()(upsamp) #BN
    return conv


def decoderblock_4(input_upstream, num_filters, kernel_size=3,nonlinear=False,padding='same',activation='relu'): 
    upsamp = layers.UpSampling2D(size=(2, 2))(input_upstream)
    
    conv = layers.Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), padding=padding, strides=1)(upsamp)
    
    if nonlinear:
        conv = layers.Activation(activation=activation)(conv)    
        
    conv = layers.Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), padding="same", strides=1)(conv)
    
    if nonlinear:
        conv = layers.Activation(activation=activation)(conv)     
    return conv



#fc layer
#16x16
def bridge_sc_1(input_bridge,num_filters):
    flat = layers.Flatten()(input_bridge)
            
    flat = layers.Dropout(rate=0.5)(flat) #Dropout
    
    shapex = input_bridge.get_shape().as_list()[1]
    shapey = input_bridge.get_shape().as_list()[2]
    #print ("TEST",shapex,shapey)
    dense = layers.Dense(shapex*shapey*num_filters)(flat)
    reshape = layers.Reshape((shapex, shapey,num_filters))(dense)
    return reshape


def unet_block_1(inputs,channels_out,num_filters=2,densebridge=False,nonlinear=False,activation='relu'):
    
    encoder0 = layers.Conv2D(filters=num_filters, kernel_size=(16, 16), padding=padding, strides=1)(inputs) #needs stride 1
    
    if nonlinear:
        encoder0 = layers.Activation(activation=activation)(encoder0) 
    
    encoder1 = encoderblock_1(encoder0, num_filters=num_filters,kernel_size=16,nonlinear=nonlinear)
   
    encoder2 = encoderblock_1(encoder1, num_filters=num_filters,kernel_size=8,nonlinear=nonlinear)
    
    encoder3 = encoderblock_1(encoder2, num_filters=num_filters,kernel_size=8,nonlinear=nonlinear)
    
    if densebridge:
        bridge = bridge_sc_1(encoder3,num_filters=num_filters) #with linear layer
        
        if nonlinear:
            bridge = layers.Activation(activation=activation)(bridge)    
    else:
        bridge = encoder3

    decoder3 = decoderblock_1(bridge, num_filters=num_filters,nonlinear=nonlinear)    

    skipcon2 = layers.concatenate([encoder2, decoder3], axis=3) 
    decoder2 = decoderblock_1(skipcon2, num_filters=num_filters,kernel_size=8,nonlinear=nonlinear) 
    
    skipcon1 = layers.concatenate([encoder1, decoder2], axis=3) 
    decoder1 = decoderblock_1(skipcon1, num_filters=num_filters,kernel_size=16,nonlinear=nonlinear)
    
    skipcon0 = layers.concatenate([encoder0, decoder1], axis=3) 
    decoder0 = layers.Conv2D(filters=channels_out, kernel_size=(16, 16), padding=padding, strides=1)(skipcon0) #needs stride 1
    
    return decoder0


#like one but samples one size further down
def unet_block_2(inputs,channels_out,num_filters=2,densebridge=False,nonlinear=False,activation='relu'):
    
    encoder0 = layers.Conv2D(filters=num_filters, kernel_size=(16, 16), padding=padding, strides=1)(inputs) #needs stride 1
    
    if nonlinear:
        encoder0 = layers.Activation(activation=activation)(encoder0)     
    
    encoder1 = encoderblock_1(encoder0, num_filters=num_filters,kernel_size=16,nonlinear=nonlinear) 
    #out: 64   
   
    encoder2 = encoderblock_1(encoder1, num_filters=num_filters,kernel_size=8,nonlinear=nonlinear)
    #out: 32
    
    encoder3 = encoderblock_1(encoder2, num_filters=num_filters,kernel_size=8,nonlinear=nonlinear) 
    #out: 16
    
    encoder4 = encoderblock_1(encoder3, num_filters=num_filters,kernel_size=4,nonlinear=nonlinear)
    #out: 8
    
    if densebridge:
        bridge = bridge_sc_1(encoder4,num_filters=num_filters) #with linear layer
        
        if nonlinear:
            bridge = layers.Activation(activation=activation)(bridge)  
    else:
        bridge = encoder4

    decoder4 = decoderblock_1(bridge, num_filters=4,nonlinear=nonlinear) 
    #out: 16    
       
    skipcon3 = layers.concatenate([encoder3, decoder4], axis=3) 
    decoder3 = decoderblock_1(skipcon3, num_filters=num_filters,kernel_size=8,nonlinear=nonlinear)    

    skipcon2 = layers.concatenate([encoder2, decoder3], axis=3) 
    decoder2 = decoderblock_1(skipcon2, num_filters=num_filters,kernel_size=8,nonlinear=nonlinear) 
    
    skipcon1 = layers.concatenate([encoder1, decoder2], axis=3) 
    decoder1 = decoderblock_1(skipcon1, num_filters=num_filters,kernel_size=16,nonlinear=nonlinear)
    
    skipcon0 = layers.concatenate([encoder0, decoder1], axis=3) 
    decoder0 = layers.Conv2D(filters=channels_out, kernel_size=(16, 16), padding=padding, strides=1)(skipcon0) #needs stride 1
    
    return decoder0

    
    
#no residual connections, but with skip connections    
#write skip conns one by one to be flexible.
def net_sc_1(img_shape,channels_out):
    
    inputs = layers.Input(shape=img_shape)
    
    outputs = unet_block_1(inputs,channels_out,densebridge=True,nonlinear=False)
   
    return inputs,outputs
    
    
    
    
    
    
    
    
    
    
