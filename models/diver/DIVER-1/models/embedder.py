import torch.nn as nn
from models.layers import *
from models.model_builders import AddEmbedding, ConcateEmbedding, CombinedEmbedding, MakeModelIgnoreDataInfoList, DataInfoAwareSequential

class Embedder(DataInfoAwareSequential):
    def __init__(self, d_model, e_layers, mup, patch_size):

        self.d_model = d_model
        self.e_layers = e_layers
        self.mup = mup
        self.patch_size = patch_size
        modules = []
        if self.patch_size == 500:
            modules.append(Test_CNN_KernelSize_Effect_500(self.d_model, 64, 8, 3))
        elif self.patch_size == 50:
            modules.append(Test_CNN_KernelSize_Effect_50(self.d_model))
        else:
            pass

        rfft_size = self.d_model // 2 +1 
        modules.append(AddEmbedding(CBraModSpectralEmbedding(d_model=self.d_model, rfft_size=rfft_size, dropout=0.1)))             
        
        d_chtype = int(self.d_model/4) 
        available_channel_types = ['EEG','iEEG']
        available_channel_subtypes = ['grid','strip','depth']
        
        modules.append(
                AddEmbedding(
                    ConcateEmbedding(
                        PositionalEncoding3D(d_model=self.d_model-d_chtype),
                        CombinedEmbedding(
                            ChannelTypeEmbedding(available_channel_types, d_chtype, True),
                            ChannelSubTypeEmbedding(available_channel_subtypes,d_chtype, True))                         
                    )
                )
            )      
        stcpe_d_model = self.d_model // 8
        stcpe_backbone = OriginalMoiraiEncoder(d_model=stcpe_d_model, e_layers=1, mup=self.mup)
        modules.append(
                AddEmbedding(
                    DataInfoAwareSequential(
                        MakeModelIgnoreDataInfoList(nn.Linear(self.d_model, stcpe_d_model)), 
                        TimeWiseSTCPE( 
                            model=stcpe_backbone,
                            stride = 1, 
                            timewindow_size=7,
                            normalize=True
                        ),
                        MakeModelIgnoreDataInfoList(nn.Linear(stcpe_d_model, self.d_model))
                    )
                )
            )
        super().__init__(*modules)