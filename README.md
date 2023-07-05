# Concrete-cracks-segmentation
Safety grants important role in buildings and roads made of concrete. Cracks in concrete can lead to dangerous situations and may put human lives in danger. 
Premature detection of this cracks can prevent such situations. The Concrete Cracks Segmentation project aims to develop a model that can accurately identify and segment cracks in concrete images.
This project utilizes computer vision techniques and deep learning algorithms to automate the process of crack detection, enabling efficient assessment and maintenance of concrete structures.

<h3 align="left">Languages and Tools:</h3>
<p align="left"> 
  <a href="https://www.python.org" target="_blank" rel="noreferrer"> 
    <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> 
  </a> 
  <a href="https://pytorch.org/" target="_blank" rel="noreferrer"> 
    <img src="https://www.vectorlogo.zone/logos/pytorch/pytorch-icon.svg" alt="pytorch" width="40" height="40"/> 
  </a> 

  <a href="https://jupyter.org" target="_blank" rel="noreferrer"> 
    <img src="https://upload.wikimedia.org/wikipedia/commons/3/38/Jupyter_logo.svg" alt="jupyter" width="40" height="40"/> 
  </a> 
  <a href="https://comet.com" target="_blank" rel="noreferrer"> 
    <img src="https://www.comet.com/site/wp-content/uploads/2022/05/comet-logo.svg" alt="comet" width="40" height="40"/> 
  </a> 
</p>

## Features 
 - **Crack Segmentation:** The system can accurately segment cracks in concrete images.</br>
 - **Preprocessing:** The project includes preprocessing steps to enhance image quality and remove corrupted images, which improve the accuracy of crack detection.</br>
 - **Deep Learning Model:** It utilizes a deep learning model trained on a large dataset of annotated concrete images to perform crack segmentation.</br>
 
## Downloading data
I'm using dataset from kaggle, folder "Concrete"
<a href="https://www.kaggle.com/datasets/yatata1/crack-dataset">https://www.kaggle.com/datasets/yatata1/crack-dataset</a>.

Before using this dataset in our model you have to perform some steps:
1. Change folder name at path ./Concrete/Negative/Mask to Masks
1. Check if every image has corrensponding mask
1. Remove 3-channel masks #ToDo

In order to do that you can use function called ```fix_dataset()``` in ```concrete_scars_dataset.py``` file
If you skip these steps it can lead to errors during training process. 

# Experiments results
The code is integrated with comet. All results of my experiments you can find under this link

<a href="https://www.comet.com/my-projects/concrete-cracks-segmentation/view/new/panels">https://www.comet.com/my-projects/concrete-cracks-segmentation</a>

# Next steps
This project still requires a lot of work to be done in order to make it usable.

## Clean dataset
As author of dataset claims, some of the datasets present some isolated cases of incorrect image-mask pairs or low precision cases, which can significantly affect the quality and accuracy of the model.
These issues have to be eliminated or taken with extra care by manually optimizing them when necessary.

## Improve the model architecture
We will test different common known architectures for our task like ENet or U-net

## License

<a href="https://en.wikipedia.org/wiki/MIT_License">MIT license</a>



