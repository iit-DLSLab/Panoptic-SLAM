#======================================
#Title: panoptic_python
#Date: 9/7/2022
#Author: Gabriel Fischer
#Usage Debug: python3 panoptic_python.py --config [PATH.yaml][OPTIONAL] --img [IMG_PATH]
#======================================

import cv2
import multiprocessing as mp
import numpy as np
from easydict import EasyDict
import yaml
import argparse


from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import Visualizer

from predictor import VisualizationDemo

class Panoptic_FPN_Net:
    def __init__(self,config):

        #print(f"config file: {config}")

        with open(config,'r') as file:
            yaml_content = yaml.safe_load(file)

        model_path = yaml_content['MODEL_PATH']
        cfg_file = yaml_content['CONFIG']

        args = self.setup_args(model_path,cfg_file)
        cfg =self.setup_cfg(args)
        self.demo = VisualizationDemo(cfg)

        self.output_img = None
        self.union_instance_mask = None
        self.image = None
       
        #print("Panoptic Python class initialized")
    #----------------------------------------------
    def setup_args(self,model_path,cfg_file):
        args_cfg = EasyDict()

        args_cfg.config_file = cfg_file
        args_cfg.confidence_threshold = 0.5
        args_cfg.opts = []
        args_cfg.opts.append("MODEL.WEIGHTS")
        args_cfg.opts.append(model_path)

        return args_cfg
    #-------------------------------------------------
    def setup_cfg(self,args):
        # load config from file and command-line arguments
        cfg = get_cfg()
        # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
        # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
        # add_panoptic_deeplab_config(cfg)
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        # Set score_threshold for builtin models
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
        cfg.freeze()
        return cfg
    #---------------------------------------------
    def panoptic_run(self,image_cv):
        predictions, visualized_output = self.demo.run_on_image(image_cv)
        return predictions,visualized_output
    #---------------------------------------------
    def panoptic_run_cpp(self,image_cv):
        self.image = image_cv
        
        results = []
        self.all_masks = []
        predictions, visualized_output = self.demo.run_on_image(image_cv)

        semantic_mask = predictions['sem_seg'].argmax(0).cpu()
        panoptic_result =predictions['panoptic_seg'][1]

        instances = predictions['instances']
        fields = instances.get_fields() 
        instance_mask = instances.get('pred_masks').cpu().numpy() # collect all instance pred masks
        bboxes =  fields['pred_boxes'].tensor.cpu().numpy() # collect all instances bounding boxes
       
        self.output_img = visualized_output.get_image()[:, :, ::-1]
        
        #fill results with "Things" 
        #[id,isThing,score,category_id,instance_id,area,mask,bbox]
        num_instances = len(instance_mask)
        semantic_labels = [] # store semantic label 

        for info,pred_mask,boxes in zip(panoptic_result,instance_mask,bboxes):
            tmp=[]
            
            info_items =list(info.items()) # get each item from dict panoptic_result and convert into a list

            pred_mask = np.array(pred_mask*255).astype('uint8')         

            #if isThing is False
            if(info_items[1][1] is False): continue # ignore Stuff objects 
            
            tmp.append(info_items[0][1]) # id
            tmp.append(info_items[1][1]) # isThing
            tmp.append(info_items[2][1]) #score
            tmp.append(info_items[3][1]) #category_id
            tmp.append(info_items[4][1]) #instance_id
            tmp.append(info_items[5][1]) #area
            tmp.append(self.binary_mask_2_bytearray(pred_mask)) # numpy mask
            tmp.append(list(boxes)) # numpy bbox
            results.append(tmp)
            self.all_masks.append(pred_mask)
     
        #fill results with "stuff" 
        semantic_results = panoptic_result[num_instances:]
        
        for info in semantic_results:
            info_items =list(info.items())
            semantic_labels.append(info_items[2][1])
   
        for x in np.unique(semantic_mask):
            if(x==0): 
                binary_mask = np.array(semantic_mask)==x
                self.union_instance_mask = binary_mask
            else:
                if(x in semantic_labels):
                    idx = semantic_labels.index(x)
                    binary_mask = np.array(semantic_mask)==x
                    semantic_result = semantic_results[idx]

                    info_items =list(semantic_result.items())
                    tmp = []

                    binary_mask = np.array(binary_mask*255).astype('uint8')

                    tmp.append(info_items[0][1]) # id
                    tmp.append(info_items[1][1]) # isThing
                    tmp.append(0.0) #score
                    tmp.append(info_items[2][1]) #category_id
                    tmp.append(0) #instance_id
                    tmp.append(info_items[3][1]) #area
                    tmp.append(self.binary_mask_2_bytearray(binary_mask)) # numpy mask
                    tmp.append([0,0,1,1]) # numpy bbox
                    results.append(tmp)
                    self.all_masks.append(binary_mask)
        
        return results
    #---------------------------------------------
    def get_output_img(self):
        return bytearray(self.output_img)
    #---------------------------------------------
    def get_all_instance_mask(self):
        temp = np.zeros_like(self.image)
        v = Visualizer(temp, None, scale=1,instance_mode=2)   
        mask = v.draw_binary_mask(self.union_instance_mask,color='white',alpha=1)
        return bytearray(mask.get_image())
    #---------------------------------------------
    def get_all_masks(self,w=640,h=480):
        masks = np.zeros([h,w,3],dtype=np.uint8)
        for mask in self.all_masks:
            dst = cv2.merge((mask,mask,mask))
            masks = cv2.addWeighted(masks,1,dst,1,0)
        return bytearray(masks)
    #---------------------------------------------

    def binary_mask_2_bytearray(self,binary_mask):
        dst = cv2.merge((binary_mask,binary_mask,binary_mask))
        return bytearray(dst)
#=============================DEBUG====================================
def main():
    #print("Debug Panoptic_FPN_Net")

    #parsing arguments--------------------------
    parser=argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/Panoptic_SLAM.yaml", help="Path to config(.yaml) file")
    parser.add_argument("--img",type=str, required=True, help="image input path")
    args=parser.parse_args()
    #-------------------------------------------

    net = Panoptic_FPN_Net(args.config)
    
    img = cv2.imread(args.img)
    results = net.panoptic_run_cpp(img)
    output_img = net.get_output_img()
    instance_mask = net.get_all_instance_mask()

    #print(predictions)
    #print(type(predictions)

    #WINDOW_NAME = "Panoptic visualization"
    #cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    #cv2.imshow(WINDOW_NAME,  output_img)
    #cv2.imshow("instance mask",instance_mask)
    #cv2.waitKey(0)
    #cv2.imwrite("panoptic_output.png",visualized_output.get_image()[:, :, ::-1])
    
    #np.savetxt("panoptic_mask.txt",predictions['panoptic_seg'][0].cpu().numpy())

    #print("done")
#===================================================
if __name__=="__main__": main()
