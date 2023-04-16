import pytorch_lightning as pl
import torch
import numpy as np
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torch.optim import lr_scheduler, optimizer
# import utils
import torchvision.transforms as T
import clip
import faiss

# from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule
# from dataloaders.HPointLocDataloader import HPointLocDataModule
from MixVPR.models import helper

import os 
tf_vpr = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
class VPRModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,
                #---- Backbone
                backbone_arch='resnet50',
                pretrained=True,
                layers_to_freeze=1,
                layers_to_crop=[],
                
                #---- Aggregator
                agg_arch='ConvAP', #CosPlace, NetVLAD, GeM
                agg_config={},
                
                #---- Train hyperparameters
                lr=0.03, 
                optimizer='sgd',
                weight_decay=1e-3,
                momentum=0.9,
                warmpup_steps=500,
                milestones=[5, 10, 15],
                lr_mult=0.3,
                
                #----- Loss
                loss_name='MultiSimilarityLoss', 
                miner_name='MultiSimilarityMiner', 
                miner_margin=0.1,
                faiss_gpu=False
                 ):
        super().__init__()
        self.encoder_arch = backbone_arch
        self.pretrained = pretrained
        self.layers_to_freeze = layers_to_freeze
        self.layers_to_crop = layers_to_crop

        self.agg_arch = agg_arch
        self.agg_config = agg_config

        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.warmpup_steps = warmpup_steps
        self.milestones = milestones
        self.lr_mult = lr_mult

        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin    

        self.tf_vpr = T.Compose([
            T.Resize((320, 320), interpolation=T.InterpolationMode.BILINEAR),
            #add a transform convert_img_type to float from uint8
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        self.save_hyperparameters() # write hyperparams into a file
        
        # self.loss_fn = utils.get_loss(loss_name)
        # self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = [] # we will keep track of the % of trivial pairs/triplets at the loss level 

        # self.faiss_gpu = faiss_gpu
        # res = faiss.StandardGpuResources()
        # flat_config = faiss.GpuIndexFlatConfig()
        # flat_config.useFloat16 = True
        # flat_config.device = 0
        self.embed_size = 4096
        self.faiss_index = faiss.IndexFlatL2(self.embed_size)

        # perfect inekf
        # adjoint purpose
        
        
        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = helper.get_backbone(backbone_arch, pretrained, layers_to_freeze, layers_to_crop)

        self.activation = {}
        

        self.llm, self.llm_preprocess = clip.load("RN50", device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.llm.visual.layer1.register_forward_hook(self.get_activation('layer1'))
        self.llm.visual.layer2.register_forward_hook(self.get_activation('layer2'))
        self.llm.visual.layer3.register_forward_hook(self.get_activation('layer3'))
        self.llm.visual.layer4.register_forward_hook(self.get_activation('layer4'))
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)
        
    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook
    
    # the forward pass of the lightning model
    def forward(self, x, llm_x):
        #create a for loop going through the batch dimension 
        #in the for loop convert each image to a PIL image 
        #then pass it through the llm model
        #then concatenate the llm output with the backbone output
        #then pass it through the aggregator
        #return the output
        
        # llm_in  = torch.cat(llm_in )
        with torch.no_grad():
            llm_feat = self.llm.encode_image(llm_x)
        llm_feat = llm_feat.detach()
        x = self.backbone(x)

        #resize pytorch tensor to BxCx20x20
        llm_feat = torch.nn.functional.interpolate(self.activation["layer3"], size=(20, 20), mode='bilinear', align_corners=False)
        x = torch.cat([x, llm_feat], dim=1)
        x = self.aggregator(x)
        return x
    
#     # configure the optimizer 
#     def configure_optimizers(self):
#         if self.optimizer.lower() == 'sgd':
#             optimizer = torch.optim.SGD(self.parameters(), 
#                                         lr=self.lr, 
#                                         weight_decay=self.weight_decay, 
#                                         momentum=self.momentum)
#         elif self.optimizer.lower() == 'adamw':
#             optimizer = torch.optim.AdamW(self.parameters(), 
#                                         lr=self.lr, 
#                                         weight_decay=self.weight_decay)
#         elif self.optimizer.lower() == 'adam':
#             optimizer = torch.optim.AdamW(self.parameters(), 
#                                         lr=self.lr, 
#                                         weight_decay=self.weight_decay)
#         else:
#             raise ValueError(f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"')
#         scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.lr_mult)
#         return [optimizer], [scheduler]
    
#     # configure the optizer step, takes into account the warmup stage
#     def optimizer_step(self,  epoch, batch_idx,
#                         optimizer, optimizer_idx, optimizer_closure,
#                         on_tpu, using_native_amp, using_lbfgs):
#         # warm up lr
#         if self.trainer.global_step < self.warmpup_steps:
#             lr_scale = min(1., float(self.trainer.global_step + 1) / self.warmpup_steps)
#             for pg in optimizer.param_groups:
#                 pg['lr'] = lr_scale * self.lr
#         optimizer.step(closure=optimizer_closure)
        
#     #  The loss function call (this method will be called at each training iteration)
#     def loss_function(self, descriptors, labels):
#         # we mine the pairs/triplets if there is an online mining strategy
#         if self.miner is not None:
#             miner_outputs = self.miner(descriptors, labels)
#             loss = self.loss_fn(descriptors, labels, miner_outputs)
            
#             # calculate the % of trivial pairs/triplets 
#             # which do not contribute in the loss value
#             nb_samples = descriptors.shape[0]
#             nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
#             batch_acc = 1.0 - (nb_mined/nb_samples)

#         else: # no online mining
#             loss = self.loss_fn(descriptors, labels)
#             batch_acc = 0.0
#             if type(loss) == tuple: 
#                 # somes losses do the online mining inside (they don't need a miner objet), 
#                 # so they return the loss and the batch accuracy
#                 # for example, if you are developping a new loss function, you might be better
#                 # doing the online mining strategy inside the forward function of the loss class, 
#                 # and return a tuple containing the loss value and the batch_accuracy (the % of valid pairs or triplets)
#                 loss, batch_acc = loss

#         # keep accuracy of every batch and later reset it at epoch start
#         self.batch_acc.append(batch_acc)
#         # log it
#         self.log('b_acc', sum(self.batch_acc) /
#                 len(self.batch_acc), prog_bar=True, logger=True)
#         return loss
    
#     # This is the training step that's executed at each iteration
#     def training_step(self, batch, batch_idx):
#         places, llm_places, labels = batch
        
#         # Note that GSVCities yields places (each containing N images)
#         # which means the dataloader will return a batch containing BS places
#         BS, N, ch, h, w = places.shape
        
#         # reshape places and labels
#         images = places.view(BS*N, ch, h, w)
#         llm_images = llm_places.flatten(0,1)
#         labels = labels.view(-1)

#         # Feed forward the batch to the model
#         descriptors = self(images, llm_images) # Here we are calling the method forward that we defined above
#         loss = self.loss_function(descriptors, labels) # Call the loss_function we defined above
        
#         self.log('loss', loss.item(), logger=True)
#         return {'loss': loss}
    
#     # This is called at the end of eatch training epoch
#     def training_epoch_end(self, training_step_outputs):
#         # we empty the batch_acc list for next epoch
#         self.batch_acc = []

#     # For validation, we will also iterate step by step over the validation set
#     # this is the way Pytorch Lghtning is made. All about modularity, folks.
#     def validation_step(self, batch, batch_idx, dataloader_idx=None):
#         places, llm_places, _ = batch
#         # calculate descriptors
#         descriptors = self(places, llm_places)
#         return descriptors.detach().cpu()
    
#     def validation_epoch_end(self, val_step_outputs):
#         """this return descriptors in their order
#         depending on how the validation dataset is implemented 
#         for this project (MSLS val, Pittburg val), it is always references then queries
#         [R1, R2, ..., Rn, Q1, Q2, ...]
#         """
#         dm = self.trainer.datamodule
#         # The following line is a hack: if we have only one validation set, then
#         # we need to put the outputs in a list (Pytorch Lightning does not do it presently)
#         if len(dm.val_datasets)==1: # we need to put the outputs in a list
#             val_step_outputs = [val_step_outputs]
        
#         for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
#             feats = torch.concat(val_step_outputs[i], dim=0)
            
#             if 'pitts' in val_set_name:
#                 # split to ref and queries
#                 num_references = val_dataset.dbStruct.numDb
#                 num_queries = len(val_dataset)-num_references
#                 positives = val_dataset.getPositives() #length of positives is num_queries
#                 r_list = feats[ : num_references]
#                 q_list = feats[num_references : ]
#             elif 'msls' in val_set_name:
#                 # split to ref and queries
#                 num_references = val_dataset.num_references
#                 num_queries = len(val_dataset)-num_references
#                 positives = val_dataset.pIdx
#                 r_list = feats[ : num_references]
#                 q_list = feats[num_references : ]
#             elif 'hloc' in val_set_name:
#                 #blah blah blah
#                 num_references = val_dataset.references.shape[0]
#                 reference_indices = val_dataset.references 
#                 query_indices = val_dataset.queries 
#                 positives = val_dataset.positives #should be np array of arrays 
#                 # [[...] * num_q]
#                 offset = 0
#                 new_positives = []
#                 for l in positives:
#                     new_positives.append(np.array(range(len(l))) + offset)
#                     offset += len(l)
#                 r_list = feats[reference_indices]
#                 q_list = feats[query_indices]
#                 # new_positives = np.concatenate(new_positives)

#             else:
#                 print(f'Please implement validation_epoch_end for {val_set_name}')
#                 raise NotImplemented

            
#             pitts_dict = utils.get_validation_recalls(r_list=r_list, 
#                                                 q_list=q_list,
#                                                 k_values=[1, 5, 10, 15, 20, 50, 100],
#                                                 gt=new_positives,
#                                                 print_results=True,
#                                                 dataset_name=val_set_name,
#                                                 faiss_gpu=self.faiss_gpu
#                                                 )
#             del r_list, q_list, feats, num_references, positives

#             self.log(f'{val_set_name}/R1', pitts_dict[1], prog_bar=False, logger=True)
#             self.log(f'{val_set_name}/R5', pitts_dict[5], prog_bar=False, logger=True)
#             self.log(f'{val_set_name}/R10', pitts_dict[10], prog_bar=False, logger=True)
#         print('\n\n')
            
            
# if __name__ == '__main__':
#     pl.utilities.seed.seed_everything(seed=190223, workers=True)
        
#     # datamodule = GSVCitiesDataModule(
#     #     batch_size=32,
#     #     img_per_place=2,
#     #     min_img_per_place=2,
#     #     shuffle_all=False, # shuffle all images or keep shuffling in-city only
#     #     random_sample_from_each_place=True,
#     #     image_size=(320, 320),
#     #     num_workers=28,
#     #     show_data_stats=True,
#     #     val_set_names=['pitts30k_val'], # pitts30k_val, pitts30k_test, msls_val
#     # )

    
    
#     # examples of backbones
#     # resnet18, resnet50, resnet101, resnet152,
#     # resnext50_32x4d, resnext50_32x4d_swsl , resnext101_32x4d_swsl, resnext101_32x8d_swsl
#     # efficientnet_b0, efficientnet_b1, efficientnet_b2
#     # swinv2_base_window12to16_192to256_22kft1k
#     model = VPRModel(
#         #---- Encoder
#         backbone_arch='resnet50',
#         pretrained=True,
#         layers_to_freeze=2,
#         layers_to_crop=[4], # 4 crops the last resnet layer, 3 crops the 3rd, ...etc
        
#         #---- Aggregator
#         # agg_arch='CosPlace',
#         # agg_config={'in_dim': 2048,
#         #             'out_dim': 2048},
#         # agg_arch='GeM',
#         # agg_config={'p': 3},
        
#         # agg_arch='ConvAP',
#         # agg_config={'in_channels': 2048,
#         #             'out_channels': 2048},

#         agg_arch='MixVPR',
#         agg_config={'in_channels' : 2048, #change this to 1024 if no clip, but 2048 with clip 
#                 'in_h' : 20,
#                 'in_w' : 20,
#                 'out_channels' : 1024,
#                 'mix_depth' : 4,
#                 'mlp_ratio' : 1,
#                 'out_rows' : 4}, # the output dim will be (out_rows * out_channels)
        
#         #---- Train hyperparameters
#         lr=0.05, # 0.0002 for adam, 0.05 or sgd (needs to change according to batch size)
#         optimizer='sgd', # sgd, adamw
#         weight_decay=0.001, # 0.001 for sgd and 0 for adam,
#         momentum=0.9,
#         warmpup_steps=650,
#         milestones=[5, 10, 15, 25, 45],
#         lr_mult=0.3,

#         #----- Loss functions
#         # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss,
#         # FastAPLoss, CircleLoss, SupConLoss,
#         loss_name='MultiSimilarityLoss',
#         miner_name='MultiSimilarityMiner', # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
#         miner_margin=0.1,
#         faiss_gpu=False
#     )

#     val_set = 'hloc'
#     datamodule = HPointLocDataModule(
#         batch_size=32,
#         img_per_place=2,
#         min_img_per_place=2,
#         shuffle_all=True, # shuffle all images or keep shuffling in-city only
#         random_sample_from_each_place=True,
#         image_size=(320, 320),
#         num_workers=28,
#         show_data_stats=True,
#         val_set_names=[val_set], # pitts30k_val, pitts30k_test, msls_val
#         llm_transform = model.llm_preprocess
#     )
    
#     # model params saving using Pytorch Lightning
#     # we save the best 3 models accoring to Recall@1 on pittsburg val
#     checkpoint_cb = ModelCheckpoint(
#         monitor=f'{val_set}/R1',
#         filename=f'{model.encoder_arch}' +
#         '_epoch({epoch:02d})_step({step:04d})_R1[{hloc/R1:.4f}]_R5[{hloc/R5:.4f}]',
#         auto_insert_metric_name=False,
#         save_weights_only=True,
#         save_top_k=3,
#         mode='max',)

#     #------------------
#     # we instanciate a trainer
#     trainer = pl.Trainer(
#         accelerator='cuda', gpus=1,
#         default_root_dir=f'./LOGS/{model.encoder_arch}', # Tensorflow can be used to viz 

#         num_sanity_val_steps=0, # runs a validation step before stating training
#         precision=16, # we use half precision to reduce  memory usage
#         max_epochs=80,
#         check_val_every_n_epoch=1, # run validation every epoch
#         callbacks=[checkpoint_cb],# we only run the checkpointing callback (you can add more)
#         reload_dataloaders_every_n_epochs=1, # we reload the dataset to shuffle the order
#         log_every_n_steps=20,
#         # fast_dev_run=True # uncomment or dev mode (only runs a one iteration train and validation, no checkpointing).
#     )

#     # we call the trainer, we give it the model and the datamodule
#     trainer.fit(model=model, datamodule=datamodule)
