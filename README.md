# Pneumonia-Detection

### Pneumonia

> Pneumonia is a form of acute respiratory infection that affects the lungs. The lungs are made up of small sacs called alveoli, which fill with air when a healthy person breathes. When an individual has pneumonia, the alveoli are filled with pus and fluid, which makes breathing painful and limits oxygen intake.

### Diagnosis 

> An X-ray helps your doctor look for signs of inflammation in your chest. If inflammation is present, the X-ray can also inform your doctor about its location and extent.

### Pneumonia Detection

> Pneumonia-Detector attempts to automate methods to detect and classify pneumonia from medical x-ray images using a [Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network). It is able to detect correctly 88% of pneumonia cases but it is NOT in any way a substitute for consulting a professional medical examiner. 

# Model Architecture Plot

[Click to view full architecture](images/model_architecture.png)

![Yolo Driving Environment Model Architecture](images/model_architecture_short.png "Pneumonia Detection Model Architecture")

# Built Using
 - [Python](https://python.org)
 - [PyTorch](https://pytorch.org)
 - [OpenCV](https://opencv.org)
 - [Kaggle Notebooks](https://www.kaggle.com)
 - [Scikit-Learn](https://scikit-learn.org)
 - [ipywidgets](https://ipywidgets.readthedocs.io/)
 - Others

# Prerequisite and Installation
* [Python](https://python.org)
    ```
        python detect.py
    ```     
* [Voila](https://voila.readthedocs.io/en/stable/using.html)
    
# Project Structure

```
│   detect-voila.ipynb
│   detect.py
│   pneumonia-detection.ipynb 
│
├───pneumonia-detector-utils
│   ├──constants.py
│   ├──pneumonia_model.py
│   └───utils.py
│
└───weights
   └─── pneumonia_detector_model.pth
```

# Usage

> For coders: Use the 'diagnose' method in detect.py either by importing or editing the script file itself. Pass an x-ray image (either a PIL.Image, torch.tensor or numpy.array) as argument to the function. 

> For non-coders: Visit this [Binder](https://mybinder.org/v2/gh/Daheer/Pneumonia-Detection/HEAD?urlpath=%2Fvoila%2Frender%2Fdetect-voila.ipynb) link, wait for it to render, sip some coffee as you wait :). 


# Demo

Placeholder               |  Prediction
:-------------------------:|:-------------------------:
![](images/placeholder-voila.jpeg) |  ![](images/sample-prediction-voila.jpeg)

# References

- [Kaggle Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- [World Health Organization Pneumonia Details](https://www.who.int/news-room/fact-sheets/detail/pneumonia)

# Contact

Dahir Ibrahim (Deedax Inc) - http://instagram.com/deedax_inc <br>
Email - suhayrid@gmail.com <br>
YouTube - https://www.youtube.com/channel/UCqvDiAJr2gRREn2tVtXFhvQ <br>
Project Link - https://github.com/Daheer/Pneumonia-Detection <br>
Twitter - https://twitter.com/DeedaxInc