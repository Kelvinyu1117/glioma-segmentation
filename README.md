# Glioma Segmentation
EE4211 Computer Vision Project - Brain Tumor Segmentation with BraTS 2018

## Data preprocessing
1. Download and extract the dataset under the subdirectory ```data```

2. Run the pre-preprocessing.py
   
## Train and Prediction

1. Upload all the generated pre-processed data to google drive in task1-full/task1-full, your directory structure should be the following:

    task1-full  
    |---resnet_like_unet    
    |---task1-full   
    |---unet_batchNorm   
    |---unet_original   



2. Run the following colab notebooks:

    * <a href="https://colab.research.google.com/drive/1V5StoBPhNDS_s5xllhMVIdv3GzQQ1PyW?usp=sharing">model1-original unet</a>

    * <a href="https://colab.research.google.com/drive/1XSxHLyD3MEw5Y-gCcFipMHnNj7j1b_qC?usp=sharing">model2-batchnorm unet</a>

    * <a href="https://colab.research.google.com/drive/1VQO6AvSYtDeprZsKyRDt1aehOSlEEBWl?usp=sharing">model3-resnet-like unet</a>

## Acknowledgements and Citation 
This project is developed with reference from the repo - <a href="https://github.com/polo8214/Brain-tumor-segmentation-using-deep-learning">Brain-tumor-segmentation-using-deep-learning</a> authored by  <a href="https://github.com/polo8214">polo8214</a>.

