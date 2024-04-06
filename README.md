# CS6910_DeepLearning_Assignment_2

# Deep Learning Assignment 1: CNN

##### ANIKET KESHRI CS23M013
This is assignment number 2 in the course, Fundamentals of Deep Learning CS6910 by Prof. Mitesh Khapra  at IIT madras. This assignment is based on the CNN with Nature_12k dataset containing 12,000 images of 10 different classes.

I run upto 200 different configurations and track them all using wandb, we then find correlations with the best features and tune further searches to attempt to reach as high an accuracy as possible:-

Report can be accessed here:- https://wandb.ai/cs23m013/Deep_Learning_A2/reports/CS6910-Assignment-2--Vmlldzo3NDE5MTQ4



#Libraries used :
The code is written in Python and using notebook using following libraries:
- torch
- numpy
- os
- wandb
- torchvision
- matplotlib
- ob

### Dataset
This assignment is based on the CNN with Nature_12k dataset containing 12,000 images of 10 different classes.

For the hyper parameter optimisation stage, 20% of the randomly shuffled training data set are kept aside for validation for each hyperparameter configuration while the model is trained on the remaining images from the randomly shuffled training data set.

Once the best configuration is identified with the help of wandb using Random search or Bayesian optimisation, the full training dataset is used to train the best model configuration and the test accuracy is calculated. 

### Convolutional Neural network
A convolutional neural network (CNN) is a category of machine learning model, namely a type of deep learning algorithm well suited to analyzing visual data. CNNs -- sometimes referred to as convnets -- use principles from linear algebra, particularly convolution operations, to extract features and identify patterns within images. Although CNNs are predominantly used to process images, they can also be adapted to work with audio and other signal data.

### classes involved
- class initialize_lenet_CNN : initiliaze number of conv layers and max pooling layers.

### Activities
1. In PART A i tuned the hyperparameters and visual filters on training CNN model from scratch .
2. I Fine tuned a pre-trained model .
3. In PART B i used a pre existing model and then validated the val dataset.
  

### Training 
The trainA.py and trainB.py script is used to train the CNN using the optimizer class with various options that can be specified using command-line arguments. Here is a description of the available options. The deafult values are set according to what worked best in the wandb sweeps.


# PART A : CNN model Training from Scratch
Here I , built a small CNN model consisting of 5 convolution layers. Each convolution layer would be followed by an activation and a max-pooling layer. 
Then after 5 such conv-activation-maxpool blocks, i have kept one dense layer followed by the output layer containing 10  neurons 

i used following hyper parameters configuration for this:

    "parameters": {

         "image_size" : {
         
            "values": [224, 256]
            
        },
        
         "epochs": {
         
            "values": [5,10]
            
        },
        
        "activation_function":
        
        {
        
            "values" : ["LeakyReLU" ,"Mish", "SiLU", "GELU", "ReLU"]
            
        },
        
        "dropout_factor":
        
        {
        
            "values": [0 ,0.1 , 0.2 , 0.3, 0.4]
            
        },
        
        "num_of_filters": {
        
            "values": [32, 64]
            
        },
        
        "filter_multiplier": {
        
            "values": [0.3 , 0.4 , 0.5]
            
        },
        
        "learning_rate" : {
        
            "values" : [0.0001, 0.0003]
            
            },
            
        "apply_data_augmentation" : {
        
            "values" : [True]
            
            },

    }

### Training A
Training is done for this model using Nature_12k Dataset with 20% data as val data. 
I used Wandb for visualizing runs and then selecting best parameters for the best model after analysis of the runs.

### Functions used 
- train_load_ : loading data batch wise after transform.
- transform_datasets_ : will transform data to add noise and all.
- data_fetch : will fetch data after all the transformation and batch .
- train : for training using various parameters.
- main : for calling from wandb to the model.

### Plot
I plotted 3 x 10 images with their actiual and predicted labels

  
# PART B : CNN model Training from Scratch



"""Train"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter) 
    
    # # Implemented Arg parse to take input of the hyperparameters from the command.
    
    parser = argparse.ArgumentParser(description="Stores all the hyperpamaters for the model.")
    
    parser.add_argument("-wp" , '--wandb_project', help='Project name used to track experiments in Weights & Biases dashboard' , type=str, 
    default='Deep_Learning_A2')
    
    parser.add_argument("-we", "--wandb_entity",type=str, help="Wandb Entity used to track experiments in the Weights & Biases dashboard." , default="cs23m013")
    
    parser.add_argument("-lg","--logger",type=bool,default=False,choices=[True,False] , help="Log to wandb or not" )
    
    parser.add_argument('-lr', '--learning_rate', help = 'Learning rate used to optimize model parameters', type=float, default=0.001)
    
    parser.add_argument("-tsbs","--test_batch_size",type=int,default=16)
    
    parser.add_argument('-ac', '--activation', help='choices: ["LeakyReLU" ,"Mish", "SiLU", "GELU", "ReLU"]', choices = ["LeakyReLU" ,"Mish", "SiLU", "GELU", 
    "ReLU"],type=str, default='ReLU')
    
    parser.add_argument("-df", "--dropout_factor", help="Dropout factor" , default=0.3, type=float)
    
    parser.add_argument("-ff", "--filter_multiplier",default="1", type=float, choices=[1, 0.5, 2])
    
    parser.add_argument("-nc","--number_of_classes",type=int,default=10)
    
    parser.add_argument("-ks","--kernel_size",type=int,default=3)
    
    parser.add_argument("-nf", "--num_filters",default="3", type=int, help="Number of filters in the convolutianal neural network.")
    
    parser.add_argument("-trbs","--train_batch_size",type=int,default=64)
    
    parser.add_argument("-vbs","--val_batch_size",type=int,default=16)
    
    parser.add_argument("-imgs","--image_size",type=int,default=256 , choices=[246,256])
    
    parser.add_argument("-aug","--apply_data_augmentation",type=bool,default=True , choices=[True,False])
    
    parser.add_argument('-ep', '--epochs', help="Number of epochs to train neural network.", type=int, default=3)
    
    parser.add_argument('-trd', '--train_data_directory', help="Dataset", type=str, default='/content/drive/MyDrive/Deeplearning/inaturalist_12K/train/')
    
    parser.add_argument('-tsd', '--test_data_directory', help="Dataset", type=str, default='/content/drive/MyDrive/Deeplearning/inaturalist_12K/val/')




