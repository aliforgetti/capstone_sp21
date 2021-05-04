## Predicting Suicidal Intent in Reddit Users Experiencing Mental Illnesses
This is the repository for my final capstone project during my Masters of Science in Data Science at Vanderbilt University.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           
    ├── README.md          
    │
    ├── models                                              <- 4 trained models ready to be used as detailed in the final_evaluation
    ├── notebooks          
    │   ├── 01_target_scraping.ipynb                        <- notebook detailing how the target cohort was scraped
    │   ├── 02_target_metadata_analysis.ipynb               <- notebook detailing metadata analysis performed on the target cohort.
    │   ├── 03_control_scraping_metadata_analysis.ipynb     <- notebook detailing metadata analysis and scraping the control group
    │   ├── 04_dataset_development.ipynb                    <- notebook detailing development of the dataset as well filtering required for modeling
    │   ├── 05_user_activity_groups.ipynb                   <- notebook detailing development of different user groups
    │   ├── 06_baseline models.ipynb                        <- notebook detailing development of baseline text classification models. 
    │   ├── 07_NN_models.ipynb                              <- notebook detailing training of neural network models
    │   ├── 08_SUIBERT_classifier.ipynb                     <- notebook detailing training of BERT text classification model to detect suicidal language
    │   └── 09_HITOPBERT_classifier.ipynb                   <- notebook detailing training of BERT text classification model to diagnose psychological illness.
    |                                            
    ├── final_model_evaluation.ipynb                        <- Final evaluation of all models on the testing dataset.   
    ├── dataset.py                                          <- Torch dataset development.         
    ├── train_eval.py                                       <- training and evaluation functions along with loss function defined
    └── model.py                                            <- Neural Network modeling architecture used for this project.  

--------

