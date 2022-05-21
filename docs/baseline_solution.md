The Baseline Implementation
=========
This starter pack contains a baseline solution to the challenge. Its accuracy is not high enough and it contains a very large potential 
for improving the computational performance and energy efficiency. Here, we give a very short description, that should serve as a guideline 
that describes what a minimum degree of documentation should be.

# Description
## Algorithm
The baseline solution performs a random forest regression on hand-creafted features of the dataset. In order to do that, the algorithm first iterates
through the dataset and computes a number of features from the data. In a second step, these features are used to train a random forest 
for the prediction. Hyperparameters are optimized with a grid search.  Finally, it applies the random forest predictor to the test data. 

The constructed features include min, max, mean, median and standard deviation for Sentinel-2 imagery, mean and max for digital elevation model and nightlights, total area covered by each class in land use classifications, majority class for local climate zone, and osm-based features such as street density, presence of highways, railways, etc., extracted from osm patches. Using this process, we calculated 125 features for each patch. We used the absolute population count as the response variable.  
We used grid search to automatically fine-tune the number of trees to grow and the maximum number of features to consider splitting a node and assess the performance through a k-fold cross-validation. 

The trained model has been validated on the 20 unseen test cities. We calculated the root-mean-square error (RMSE), the mean absolute error (MAE) and, the square of the correlation coefficient (R2). Currently, we achieved the RMSE of 1327.284. We believe that with more sophisticated features and machine learning methods, a powerful model could be developed for the task.

## Implemenentation
The feature calculation is implemented as an aggregation of data from different source based on standard python libraries such as numpy. 
No parallelization is applied. The resulting features are stored as csv. The random forest regression is performed with the scikit-learn implementation. 

# How to run
## Description
On JUWELS Booster, the baseline run can be performed by cloning simply cloning the repository into a directory where the user has write access and submitting a computational job. Two options are available, the job script `run_train_on_jb.sh` will execute the run using the software modules installed on the system. The job script `run_train_singularity_jb.sh` uses a singularity image that is based on the Dockerfile [docker/scikit-learn/Dockerfile](docker/scikit-learn/Dockerfile). It was created with the workflow described [https://gitlab.jsc.fz-juelich.de/AI_Recipe_Book/recipes/singularity_docker_jupyter](here). 
Accordingly, on HoreKa, the 

## Step by step
0. Obtaining the code
Clone the repo:
```bash
cd /path/to/repo
git clone https://gitlab.hzdr.de/count-me-in-challenge/orga-team/starter-pack.git
```
1. Data preparation
Nothing needs to be done, the data is used in the originally prepared form, accessible on the respective computers. 
2. Execution
```bash
cd /path/to/repo
sbatch <submission_script>
```
where `<submission_script>` is one of `run_train_on_jb.sh` `run_train_singularity_jb.sh` and `run_horeka.sh`. 
3. Results retrieval
The results are written into the path `/path/to/repo/` into a file named by the execution time. The stdout can be found 
in a file named `slurm-<jobid>.out`. On HoreKa, this file contains the consumed energy. 
