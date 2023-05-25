
import os
import glob
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy
import cv2
import torch

from PIL import Image
from torchvision import transforms
from unet import UNet

class SegModel(object):
    def __init__(self, model_path=r"model.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = UNet(n_channels=1, n_classes=2).to(device=self.device)
        net.load_state_dict(torch.load(model_path, map_location=self.device))
        net.eval()
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.track_running_stats = False
        self.net = net
    def infer(self, images):
        # images : NCHW, uint8
        assert len(images.shape)==4
        assert images.dtype=="uint8"
        assert images.shape[1]==1
        img = [cv2.resize(elm[0], (800,800), interpolation=cv2.INTER_NEAREST)[None] for elm in images]
        img = numpy.asarray(img) / 255.0
        img = torch.from_numpy(img)
        img = img.to(device=self.device, dtype=torch.float32)
        with torch.no_grad():
            output = self.net(img)
        probs = torch.argmax(output, 1)
        # print(probs.max())
        masks = probs.cpu().numpy().astype("uint8")
        masks = [cv2.resize(elm, (images.shape[3],images.shape[2]), interpolation=cv2.INTER_NEAREST) for elm in masks]
        
        masks = numpy.asarray(masks)
        return masks
    
    def process(self, mask):
        mask = mask*250
        res = cv2.Canny(mask, 50, 150)
        return res
        

            
            
            
def infer_testset_images(segModel):
    root = "./testset/"
    image_files = glob.glob(os.path.join(root, "x", "*.*"))
    image_files = [elm for elm in image_files if elm.endswith(".jpg") 
                                                 or elm.endswith(".jpeg") 
                                                 or elm.endswith(".JPG") 
                                                 or elm.endswith(".JPEG") 
                                                 or elm.endswith(".png") 
                                                 or elm.endswith(".PNG") 
                                                 or elm.endswith(".bmp") 
                                                 or elm.endswith(".BMP")]
    for image_file_i in image_files:
        image = cv2.imread(image_file_i, 0)
        images = image[None, None]
        mask = segModel.infer(images)[0]
        mask_file_i = image_file_i.replace("/x/", "/y/")
        cv2.imwrite(mask_file_i, (mask*250).astype("uint8"))



import SimpleITK as sitk
def save_mhd(array_img, output_path):
    sitk_img = sitk.GetImageFromArray(array_img, isVector=False)
    sitk.WriteImage(sitk_img, output_path)

def images2niigz():
    root = "./testset/"
    image_files = glob.glob(os.path.join(root, "x", "*.*"))
    image_files = [elm for elm in image_files if elm.endswith(".jpg") 
                                                 or elm.endswith(".jpeg") 
                                                 or elm.endswith(".JPG") 
                                                 or elm.endswith(".JPEG") 
                                                 or elm.endswith(".png") 
                                                 or elm.endswith(".PNG") 
                                                 or elm.endswith(".bmp") 
                                                 or elm.endswith(".BMP")]
    mask_files = [elm.replace("/x/", "/y/") for elm in image_files]
    images = [cv2.imread(image_file_i, 0) for image_file_i in image_files]
    masks = [cv2.imread(mask_file_i, 0) for mask_file_i in mask_files]
    images_array = numpy.asarray(images)
    masks_array = numpy.asarray(masks)
    save_mhd(images_array, os.path.join(root, "x.nii.gz"))
    save_mhd(masks_array, os.path.join(root, "y.nii.gz"))



def infer_testset_videos(segModel):
    root = "./testset/"
    video_files = glob.glob(os.path.join(root, "x", "*.*"))
    video_files = [elm for elm in video_files if elm.endswith(".avi") 
                                                 or elm.endswith(".mp4")]
    for video_file_i in video_files:
        mask_file_i = video_file_i.replace("/x/", "/y/")
        avi2res(video_file_i, mask_file_i, segModel)


        

def avi2res(avi_file, res_file, segModel):
    cap = cv2.VideoCapture(avi_file)
    ret, frame = cap.read()
    print(frame.shape)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 15
    count = 0
    vout = cv2.VideoWriter(res_file, fourcc, fps, (frame.shape[1], frame.shape[0]), False)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            count += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            images = frame[None, None]
            mask = segModel.infer(images)[0]
            mask = segModel.process(mask)
            frame = numpy.where(mask>0, 255, frame)
            vout.write(frame)
        else:
            break
    vout.release()
    cap.release()

        

if __name__ == "__main__":
    segModel = SegModel(r"./checkpoints/CP_epoch17.pth")
    infer_testset_images(segModel)
    # images2niigz()
    #infer_testset_videos(segModel)







