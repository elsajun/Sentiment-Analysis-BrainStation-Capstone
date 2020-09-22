# Sentiment Analysis BrainStation Capstone
Predicting the sentiment (positive or negative) of a product review using machine learning tools

README

Title: Sentiment Analysis of Amazon's Product Reviews

Files Includes With This Project:
   Sentiment Analysis - EDA, Modelling, Analysis, & Result.ipynb 	: analysis of 8 modellings
   Sentiment Analysis - Prediction with Linear SVC			: make predictions with Linear SVC model
   my_util.py								: functions that I made for this project
   data/Amazon Reviews.csv						: the original data
   data/clean_data.csv							: consists of reviews.text and reviews.rating column from Amazon Reviews.csv


INTRODUCTION
------------
Customer Experience (CX) is the key to business success. Now, more than ever, it is the key for companies to pay close attention to improve the customer experience. 
By analyzing and getting insights from customer feedback, companies have better information to make strategic decisions, an accurate understanding of what the customer
actually wants and, as a result, a better experience for everyone.
But what are customers saying about your product? How can you provide a better experience? With these questions in mind, businesses are using tools that collect public 
reviews about their products. However, they just end up with an overload of puzzling feedback that still does not answer their questions, unless they devote hours of 
manual labor to analyzing this unstructured data. Those days are over thanks to sentiment analysis. But what is it? 
Sentiment analysis is machine learning technique to detect polarity within a given text. 
This machine learning tool can provide insights by automatically analyzing product reviews and separating them into tags: Positive or Negative.

REQUIREMENTS
------------
This project was completed using Jupyter Notebook and Python with Numpy, Pandas, Matplotlib, Seaborn, Sys, NLTK, re, String, and Tensorflow


INSTALLATIONS
-------------

For Jupyter Notebook and Python, you can download it here: https://www.anaconda.com/download/ 
Installing Anaconda: Windows Note 
- Adding Anaconda to the systmen PATH environment variable is necessary to use Anaconda in its full capabilites

Before working with TensorFlow, Keras or PyTorch, we need to install the packages. 
There is nothing that explicitly stops us from installing these packages in our base environment, however, this is generally considered bad practice for two reasons:
- We can quickly loose track of what packages we have installed.
- Some packages do not play well together, so we can potentially corrupt our base environment.

Instead, lets create a new environment to install our deep learning libraries to.
Step 1. Create the new empty environment named 'sentimentanalysis'.
        conda create -n sentimentanalysis
Step 2. Activate the new environment.
        conda activate deeplearning
Step 3. Install all the basic packages we'll need (including jupyter notebook and lab).
        conda install numpy pandas matplotlib jupyter jupyterlab pydot pillow seaborn
Note: You might get an initial frozen solve, but wait it out and it should get installed using a flexible solve
Step 4. Install TensorFlow in this environment. pip install --upgrade pip
        pip install tensorflow==2.2
Note: The TensorFlow version available from Conda is a few versions behind. So we will use pip to install TensorFlow so we can use the latest version.
Step 5. Install some more packages that we'll need in the TensorFlow Lecture.
        conda install scikit-learn nltk
        conda install -c conda-forge gensim
Step 6. Open a Jupyter Notebook or Lab from this environment.


Troubleshooting: Microsoft Windows
If you receive an error message when attempting to import a TensorFlow module, you will likely need to download the latest C++ library for Microsoft Windows. 
You can download required installer here: https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads
Select the version required for your computer processor (likely the x64 version).


CONTRIBUTING
------------
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate.


PROJECT STATUS
--------------
Will move on to multi classifications and try other algorithms with better prediction outcomes









 

