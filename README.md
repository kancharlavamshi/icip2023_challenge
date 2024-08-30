This work focuses on improving object detection performance by addressing the issue of image distortions, commonly encountered in uncontrolled acquisition environments. High-level computer vision tasks such as object detection, recognition, and segmentation are particularly sensitive to image distortion. To address this issue, we propose a novel approach employing an image defilter to rectify image distortion prior to object detection. This method enhances object detection accuracy, as models perform optimally when trained on non distorted images. Our experiments demonstrate that utilizing defiltered images significantly improves mean average precision compared to training object detection models on distorted images. Consequently, our proposed method offers considerable benefits for real-world applications plagued by image distortion. To our knowledge, the contribution lies in employing distortion-removal paradigm for object detection on images captured in natural settings. We achieved an improvement of 0.562 and 0.564 of mean Average precision on validation and test data.


To improve object detection accuracy, it's essential to provide non-distorted images. Instead of training models with distorted images, our proposed method removes distortion from the given image using defilters and passes the resulting images through an object detection model.
Our proposed method follows these steps: 

1. Machine learning classifiers are used to determine the type of distortion affecting the image. 
2. The corresponding defilters are applied to remove the distortion from the image.
3. The resulting images are then processed through an object detection model to detect objects.
![block_diagram](https://github.com/user-attachments/assets/80064d9c-9d43-490b-ad66-cfac47fe4c38)


Launch the ICIP_submission_notebook.ipynb notebook; all of the instructions are outlined in the notebook step by step.

sample images are located at files/not_defiltered folder.

### Our solution secured second place on the leaderboard.
![leader_board_codalab](https://github.com/user-attachments/assets/67178c80-10e2-4819-a143-a953857094f5)


Please find our [Paper] (https://arxiv.org/pdf/2404.08293) attached and feel free to reach out to us if you encounter any issues.
