Computer vision tasks, such as object detection, recognition, and segmentation, are highly sensitive to image capture conditions. In uncontrolled acquisition environments, images may be affected by local and global noise, making object recognition challenging in real-world applications. To improve object detection accuracy, it's essential to provide non-distorted images. Instead of training models with distorted images, our proposed method removes distortion from the given image using defilters and passes the resulting images through an object detection model.
Our proposed method follows these steps: 

1. Machine learning classifiers are used to determine the type of distortion affecting the image. 
2. The corresponding defilters are applied to remove the distortion from the image.
3. The resulting images are then processed through an object detection model to detect objects.

Launch the ICIP_submission_notebook.ipynb notebook; all of the instructions are outlined in the notebook step by step.

sample images are located at not_defiltered folder.

Please get in touch with us if you run into any problems.
