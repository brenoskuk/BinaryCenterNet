# BinaryCenterNet

Binary neural networks, Object detection using center point detection:

> **Efficient Object Detection Using Binary Neural Networks**,            
> Breno Skuk, Dimitris Milioris, Joachim Wabnig         
> ([technical report](https://github.com/brenoskuk/BinaryCenterNet/blob/main/Technical_Object_Detection_BNN.pdf)) 

 

 


Contact: [breno.skuk@gmail.com](mailto:breno.skuk@gmail.com) 


## Abstract 

Recent advances in the field of neural networks with quantized weights and activations down to single bit precision have allowed the development of models that can be deployed in resource constrained settings, where a trade-off between task performance and efficiency is accepted.
Here we design an efficient single stage object detector based upon the CenterNet architecture containing a combination of full precision and binarized layers. Our model is easy to train and  achieves comparable results with a full precision network trained from scratch while requiring an order of magnitude less FLOP. 
Even in cases where the efficient models show lower task performance than their full precision counterparts, their use is still justified by the speedup of up to 58X that comes with performing convolutions of binary weights and binary activations. This opens the possibility of deploying an object detector in applications where time is of the essence and a GPU is absent.  
We study the impact of increasing the ratio of binarized MAC operations without significant increases in the total amount of MAC operations. Our study offers an insight into the the design process of binary neural networks for object detection and the involved trade-offs. 

## About BinaryCenterNet

- **A decent and efficient object detector:** Our implementation is straightforward and simple to understand. Although the state of the art of BNN's on object detection has almost managed to catch up with its full precision counterparts, this implementation does not involve any complicated training tricks and should be straightforward to implement. 

- **Modular architecture:** We modularize our network into an input module, an encoder, a decoder and a head. Each module can have it's feature maps and types of layers easily changed, allowing for experimentation with the number of network parameters and operations. 

- **Create and train your own implementetion:**  You can test your own implementations and profit of the training pipeline on the PascalVOC dataset. The architecture folder contains an easy to modify template of a Binary CenterNet architecture that can be customized as needed. 


## Experiments results

### Object Detection on PascalVOC2007 test set

The related paper to this implementation contains a convention thas specifies architectures modularly. The table below showcases some of the results that can be obtained by modifying the custom_qn_centernet file in the architectures folder using the convention established in the paper:

| Input | Encoder | Decoder | Heads |  mAP | eq FLOP |
| -----|---------|---------|-------|------|-----------------|
|qn(512)    |qn([4 × 128, 4 × 256, 4 × 324, 4 × 512])        | tp(2)([64, 64, 32])       |   h(32)   |    54.6  |       1.60          |
| qn(512)    |qn([2 × 128, 4 × 256, 8 × 324, 10 × 512])         | tp(4)([128, 64, 16])       |   h(64)    |   56.4  |       3.19          |
| qn(512)    |qn([4 × 64, 4 × 128, 4 × 162, 4 × 256])       | tp(2)([64, 64, 64])       |  h(32)   |   40.0  |       1.53         |


## Installation

Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions.

## Design BinaryCenterNet models

You can design your own custom backbone and add it to the architectures folders. 


## Train BinaryCenterNet models

### Train using PascalVOC Object detection images and bounding boxes


### Train using architectures and PascalVOC


To train using an architecture from the architectures folder:     

~~~
python train.py --architecture <<architecture name>>
~~~

### Training hyperparameters and args

Here is a list of the arguments that can be given to train.py:

    --architecture : Architecture that will be used to train the network.
    
    --batch_size : Size of the batches. default=32

    --num_classes : Number of classes (20 for PascalVOC). default=20

    --input_size : Size of the input (eg.: 256). default=512
   
    --gpu : Id of the GPU to use (as reported by nvidia-smi).
   
    --epochs : Number of epochs to train. default=200
   
    --no-snapshots : Disable saving snapshots.
   
    --epoch-evaluation : Disable per epoch evaluation.

    --final-evaluation : Disable final model evaluation.

    --data-augmentation : Enable/disable data augmentation.

    --model-summary : Disable model summary.

    --dataset-path : Set the dataset path. default='datasets\PascalVOC-OD-2007-2012

    --compute-val-loss : Disable validation loss during training.

    --val-sampling : Set the fraction relative to all data to be sampled at each epoch when calculating the validation mAP during training (1 uses all val set). default=0.25

    --train-sampling : Set the fraction relative to all data to be sampled at each epoch when calculating the train mAP during training (1 uses all train set). default=0.05

    --train-verbose: Tensorflow .fit verbose: 0 = silent, 1 = progress bar, 2 = only epochs. default=2

## Inference with BinaryCenterNet models

For infering objects using a trained model, run:


**New feature:** Current version supports inference on .mp4 files. Functions can be found at utils/video_utils.py 
~~~
python infer.py  --architecture <<path to architecture>> --model-weights-path <<path to model weights>>
~~~

## Larq resources

This project uses [**Larq**](https://docs.larq.dev/larq/), an Open-Source Library for Training Binarized Neural Networks.

We also make use of the QuickNet architecture. 

> [**Larq: An Open-Source Library for Training Binarized Neural Networks**](http://arxiv.org/abs/1904.07850),            
> Lukas Geiger and Plumerai Team,        
> *Journal of Open Source Software ([https://doi.org/10.21105/joss.01746](https://doi.org/10.21105/joss.01746))* 

## Original CenterNet resources

> [**Objects as Points**](http://arxiv.org/abs/1904.07850),            
> Xingyi Zhou, Dequan Wang, Philipp Kr&auml;henb&uuml;hl,        
> *arXiv technical report ([arXiv 1904.07850](http://arxiv.org/abs/1904.07850))* 

- CenterNet + embedding learning based tracking: [FairMOT](https://github.com/ifzhang/FairMOT) from [Yifu Zhang](https://github.com/ifzhang).
- Detectron2 based implementation: [CenterNet-better](https://github.com/FateScript/CenterNet-better) from [Feng Wang](https://github.com/FateScript).
- Keras Implementation: [keras-centernet](https://github.com/see--/keras-centernet) from [see--](https://github.com/see--) and [keras-CenterNet](https://github.com/xuannianz/keras-CenterNet) from [xuannianz](https://github.com/xuannianz).
- MXnet implementation: [mxnet-centernet](https://github.com/Guanghan/mxnet-centernet) from [Guanghan Ning](https://github.com/Guanghan).
- Stronger human open estimation models: [centerpose](https://github.com/tensorboy/centerpose) from [tensorboy](https://github.com/tensorboy).
- TensorRT extension with ONNX models: [TensorRT-CenterNet](https://github.com/CaoWGG/TensorRT-CenterNet) from [Wengang Cao](https://github.com/CaoWGG).
- CenterNet + DeepSORT tracking implementation: [centerNet-deep-sort](https://github.com/kimyoon-young/centerNet-deep-sort) from [kimyoon-young](https://github.com/kimyoon-young).
- Blogs on training CenterNet on custom datasets (in Chinese): [ships](https://blog.csdn.net/weixin_42634342/article/details/97756458) from [Rhett Chen](https://blog.csdn.net/weixin_42634342) and [faces](https://blog.csdn.net/weixin_41765699/article/details/100118353) from [linbior](https://me.csdn.net/weixin_41765699).


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Citation

If you find this project useful for your research, please use the following BibTeX entry.

@techreport{skuk2022bnn,
  title={Efficient Object Detection Using Binary Neural Networks},
  author={Skuk, Breno and Milioris, Dimitris and Wabnig, Joachim},
  institution={Nokia Bell Labs},  % or your affiliation
  year={2022},
  number={Technical Report},  % Optional
  url={https://github.com/brenoskuk/BinaryCenterNet/blob/main/Technical_Object_Detection_BNN.pdf}
}

