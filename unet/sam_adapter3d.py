""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_model_3d import *
from .unet_parts import *
from torch.nn.functional import threshold, normalize

from segment_anything import sam_model_registry, SamPredictor


class SAMApter3d(nn.Module):
    def __init__(self):
        super(SAMApter3d, self).__init__()

        sam_checkpoint = "/hy-tmp/74e556d0ef5f48c2af338d376fdc054c/CT/notebooks/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device).train().half()
        # predictor = SamPredictor(sam)
        self.samApter3d_0 = SAMApter3d_0()
        self.samApter3d_0.to(device=device).train().half()

    def forward(self, x, embed):
        """
        x.size()     = (Batch * numSlice, C, H, W) = (Batch * 64,   1, 512, 512)
        embed.size() = (Batch * numSlice, C, H, W) = (Batch * 64, 256,  64,  64)
        """
        prompt_1, embed3 = self.samApter3d_0(x, embed)
        # prompt_2 = torch.ones_like(prompt_1)
        
        with torch.no_grad():
            #### 送入prompt_encoder
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=None,
                boxes=None,
                masks=prompt_1)
            sparse_embeddings = torch.empty((1,0,256), dtype=torch.float16, device = 'cuda')
        #### 送入mask_decoder
        low_res_masks, iou_predictions = self.sam.mask_decoder(
        image_embeddings=embed3,
        image_pe=self.sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
        )
        upscaled_masks = self.sam.postprocess_masks(low_res_masks, (512,512), (512,512)).to("cuda")
        # binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))
        return upscaled_masks #, iou_predictions
        

class SAMApter3d_0(nn.Module):
    def __init__(self):
        super(SAMApter3d_0, self).__init__()        
        self.adapter3d = UNet3d(256, 256)
        self.down2d = Down(1, 16)
        self.prompt3d = UNet3d(16, 1)
        self.numSlice = 64

    def forward(self, x, embed):
        """
        x.size()     = (Batch * numSlice, C, H, W) = (Batch * 64,   1, 512, 512)
        embed.size() = (Batch * numSlice, C, H, W) = (Batch * 64, 256,  64,  64)
        """
        
        #### image_encoder输出的embed，送入adapter3d，得到embed3后面进入mask_decoder
        embed_transpose = embed.view(-1, self.numSlice, 256, 64, 64)
        embed_transpose = embed_transpose.permute(0,2,1,3,4)
        embed2_transpose = self.adapter3d(embed_transpose)
        embed2_transpose = embed2_transpose.permute(0,2,1,3,4)
        embed2 = embed2_transpose.reshape(-1, 256, 64, 64)
        embed3 = embed + embed2
        
        #### x进入prompt3d得到(Batch * numSlice, C, H, W), 即(Batch * 64, 1, 256, 256), 再送入prompt_encoder
        x_1 = self.down2d(x)
        x_1_transpose = x_1.view(-1, self.numSlice, 16, 256, 256)
        x_1_transpose = x_1_transpose.permute(0,2,1,3,4)
        prompt_1 = self.prompt3d(x_1_transpose)
        prompt_1 = prompt_1.permute(0,2,1,3,4)
        prompt_1 = prompt_1.reshape(-1, 1, 256, 256)            
        
        
        return prompt_1, embed3

if __name__=="__main__":
    import numpy
    import torch
    unet3d = SAMApter3d()
    x = torch.rand(2*64,1,512,512).cuda().half()
    embed = torch.rand(2*64,256,64,64).cuda().half()
    o = unet3d(x, embed)
    print(o[0].size())
    print(o[0].min(), o[0].max())
    print(o[1].size())
    print(o[1][0])
    
