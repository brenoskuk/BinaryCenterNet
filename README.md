# BinaryCenterNet

Binary nural networks, Object detection using center point detection:

> [**Efficient Object Detection Using Binary Neural Networks**](http://arxiv.org/abs/1904.07850),            
> Breno Skuk, Dimitris Milioris, Joachim Wabnig         
> *technical report ([None](None))*  

 


Contact: [breno.skuk@gmail.com](mailto:breno.skuk@gmail.com) 


## Abstract 

Recent advances in the field of neural networks with quantized weights and activations down to single bit precision have allowed the development of models that can be deployed in resource constrained settings, where a trade-off between task performance and efficiency is accepted.
Here we design an efficient single stage object detector based upon the CenterNet architecture containing a combination of full precision and binarized layers. Our model is easy to train and  achieves comparable results with a full precision network trained from scratch while requiring an order of magnitude less FLOP. 
Even in cases where the efficient models show lower task performance than their full precision counterparts, their use is still justified by the speedup of up to 58$\times$ that comes with performing convolutions of binary weights and binary activations. This opens the possibility of deploying an object detector in applications where time is of the essence and a GPU is absent.  
We study the impact of increasing the ratio of binarized MAC operations without significant increases in the total amount of MAC operations. Our study offers an insight into the the design process of binary neural networks for object detection and the involved trade-offs. 

## About BinaryCenterNet

- **XX:** 

- **XX:** 

- **XX:** 

- **XX**: 

- **XX:** 

## Experiments results

### Object Detection on PascalVOC2007 test

| Stem | Encoder | Decoder | Heads |  mAP | eq FLOP |
| -----|---------|---------|-------|------|-----------------|
| x    |x        | x       |   x   |   x  |      x          |
| x    |x        | x       |   x   |   x  |      x          |
| x    |x        | x       |   x   |   x  |      x          |


## Installation

Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions.

## Design BinaryCenterNet models

For creating and architecture, run     

~~~
python demo.py ctdet --demo webcam --load_model ../models/ctdet_coco_dla_2x.pth
~~~

## Train BinaryCenterNet models


For training a given architecture, run:

~~~
python demo.py ctdet --demo /path/to/image/or/folder/or/video --load_model ../models/ctdet_coco_dla_2x.pth
~~~

## Inference with BinaryCenterNet models

For infering objects using a trained model, run:

~~~
python demo.py ctdet --demo /path/to/image/or/folder/or/video --load_model ../models/ctdet_coco_dla_2x.pth
~~~

## Original CenterNet resources

- CenterNet + embedding learning based tracking: [FairMOT](https://github.com/ifzhang/FairMOT) from [Yifu Zhang](https://github.com/ifzhang).
- Detectron2 based implementation: [CenterNet-better](https://github.com/FateScript/CenterNet-better) from [Feng Wang](https://github.com/FateScript).
- Keras Implementation: [keras-centernet](https://github.com/see--/keras-centernet) from [see--](https://github.com/see--) and [keras-CenterNet](https://github.com/xuannianz/keras-CenterNet) from [xuannianz](https://github.com/xuannianz).
- MXnet implementation: [mxnet-centernet](https://github.com/Guanghan/mxnet-centernet) from [Guanghan Ning](https://github.com/Guanghan).
- Stronger human open estimation models: [centerpose](https://github.com/tensorboy/centerpose) from [tensorboy](https://github.com/tensorboy).
- TensorRT extension with ONNX models: [TensorRT-CenterNet](https://github.com/CaoWGG/TensorRT-CenterNet) from [Wengang Cao](https://github.com/CaoWGG).
- CenterNet + DeepSORT tracking implementation: [centerNet-deep-sort](https://github.com/kimyoon-young/centerNet-deep-sort) from [kimyoon-young](https://github.com/kimyoon-young).
- Blogs on training CenterNet on custom datasets (in Chinese): [ships](https://blog.csdn.net/weixin_42634342/article/details/97756458) from [Rhett Chen](https://blog.csdn.net/weixin_42634342) and [faces](https://blog.csdn.net/weixin_41765699/article/details/100118353) from [linbior](https://me.csdn.net/weixin_41765699).


## License

X

## Citation

If you find this project useful for your research, please use the following BibTeX entry.

    @inproceedings{XXX,
      title={Efficient Object Detection using Binary Networks},
      author={Skuk, Breno and Milioris, Dimitris and Wabnig, Joachim},
      booktitle={XX},
      year={2022}
    }