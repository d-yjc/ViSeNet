# ViSeNet: Visual Semantic Network for Salient Object Detection with Language using Tranformers
<p align="center">
<img src="https://github.com/d-yjc/ViSeNet/blob/main/repo_images/showcase.png" alt="Heatmap Fixation Showcase Image" style="width:90%;">
<p/>
ViSeNet is a novel, deep-learning SOD (Salient Object Detection) network which utilises both visual and semantic annotation information in order to power its saliency predictions on images.
  
Our [Relationship Module](https://github.com/d-yjc/ViSeNet/blob/main/mask2former/modeling/transformer_decoder/relation_module.py) is what provides ViSeNet the capability of using relation-aware features in order to enhance its predictions. 

To this end, we implement the Relationship Module using an edge-featured **Graph Attention Network** to model the semantic relationships between objects in the scene.
Objects are represented as the nodes, with the relationship between them defining edge and edge features therefore. 
<p align="center">
<img src="https://github.com/d-yjc/ViSeNet/blob/main/repo_images/vise_3.jpg" alt="Model Diagram Image" style="width:85%;">
<p/>

## Notability
 Our model achieves performance comparative to state-of-the-arts methods, despite being trained on *significantly* less data.
<p align="center">
<img src="https://github.com/d-yjc/ViSeNet/blob/main/repo_images/compare_stat.png" alt="Compare Stat Image" style="width:80%;">
<p/>

Furthermore, our model's object relation-aware features allows it to accurately predict small objects that are salient yet often undetected
by state-of-the-art to their lack of size (i.e., low-level feature)
<p align="center">
<img src="https://github.com/d-yjc/ViSeNet/blob/main/repo_images/comparisons.png" alt="Visual Comparisons" style="width:80%;">
<p/>

  
##
## Install 
ViSeNet is built upon [Mask2Former](https://github.com/facebookresearch/Mask2Former) as our backbone. 
Please refer to [Mask2Former instructions](https://github.com/facebookresearch/Mask2Former/blob/main/INSTALL.md).  

We recommend using **Python 3.7** and **PyTorch 1.8-1.10**.

### HiCoST Dataset
Our image dataset, HiCoST (High-Context Saliency Transfer), can be found at this Google Drive link:  
[Download HiCoST](https://drive.google.com/drive/folders/1Yv4PglXflvrZHCwdM3iyPI2_Ysc8RjDJ?usp=sharing)
