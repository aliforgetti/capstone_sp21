## Predicting Suicidal Intent in Reddit Users Experiencing Mental Illnesses

### About
This is the repository for my final capstone project during my Masters of Science in Data Science at Vanderbilt University.

The aim of this project is to create a novel predictive modeling framework that models the transition of Reddit users experiencing mental health problems into experiencing suicidal ideation. Through this, a more holistic understanding can be gleaned of the underlying factors governing the rise of suicidal ideation, and how the occurrence of such factors could be reduced effectively. This will be done by developing a sequential representation of the user’s text activity on the platform combined with where they are posting. This is done through a sequential neural network model known as a Time-LSTM. 

### Data Description

A list of 57 subreddits were identified based on commonly attributed causes or risk factors of sucide. These risk factors could be mental health related (internal) or linked to external events or circumstances. These will be referred to as MH subreddits (because a majority of them are linked to mental health conditions). 

| **Internal (N=40)**             | **External (N=17)**               |
|-----------------------------|-------------------------------|
| Depression (N=6)            | Turbulent Relationships (N=7) |
| Bipolar Disorder (N=2)      | Turbulent Upbringing (N=5)    |
| Personality Disorders (N=1) | Traumatic Events (N=5)        |
| PTSD (N=2)                  |                               |
| OCD (N=1)                   |                               |
| Addiction (N=12)            |                               |
| Schizophrenia (N=1)         |                               |
| Eating Disorders (N=5)      |                               |
| Self Harm (N=2)             |                               |
| Loneliness (N=2)            |                               |

To extract the target users (suicidal users), the following steps were taken.
1. Get all users that posted on r/SuicideWatch from 2018 to 2021
2. Remove moderator posts
3. Remove deleted posts
4. Retrieve all posts and comments of remaining users and only keep them if:
    - Posted on MH subreddits within the last 365 days on more than two occasions
    - Account age was more than 3 months (no new accounts)

All posts before their first post on r/SuicideWatch were deleted since the predictions are to be made on activity leading up to them eventually becoming suicidal.

The control group was extracted from users that posted in the MH subreddits in 2017 but never posted in r/SuicideWatch after that. This was done to ensure they had three years to potentially post in r/SuicideWatch. To maintain homogeneity, avoid bias and make sure that we model what causes those at risk to actually become suicidal,  the same distribution of people in MH subreddits is maintained in the control group. That means if we had 1000 target users with activity in /r/depression, the same number of users must exist in our control group. 

A total of 10,056 users and their posting histories were scraped off Reddit with 5,075 suicidal users and 4,981 non-suicidal users. Two target subgroups were identified, where the target was given a value of 1, and one where the target was 0. The former is the ‘suicidal’ sub-group, which includes users having a history of posting in MH subreddits and posted in the years 2018 - 2021. The latter is the ‘non-suicidal’ sub-group, which includes users having a history of posting in MH subreddits in 2017, but who never posted in /r/SuicideWatch following that. For each user, all of their posts and comments were extracted, along with other metadata such as where those posts and comments were made (subreddit), the number of upvotes, downvotes as well as the number of responses.

### Model Architecture
![image](https://user-images.githubusercontent.com/17579534/128061977-761798aa-62e2-4aa8-86e7-3108e07b2935.png)

You can reach out to me on ayaqoobing@gmail.com to request access to data and project details.

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

