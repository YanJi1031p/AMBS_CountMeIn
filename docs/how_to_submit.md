How To Submit
============
# The YAML File
For a valid submission, you will need to provide a `zip` file that contains: (i) the predicted population counts (a `csv` file, please check our [example](../rf_regression.py), line 133), and a `yaml` file with the fields describing your job. The fields are inspired by the slurm job properties but contain additional fields. Note that the performance (RMSE) will be computed in the submission system. Therefore, you need to check whether the performance threshold is met.

[Here](submission_template.yaml) is a preliminary example submission. Please check before submitting, if this format is still applicable.

```yaml
haicore_center: kit # Can be kit or fzj
JobId: 1646588
StartTime: 2022-04-05T23:13:26 
EndTime: 2022-04-05T23:14:46
energy: 40362 # Replace by NaN for fzj
repository: http://github.com/awesome_dude/my_submission
Commit: aff6768gb
```

# The submission platform
Your submission will be submitted to the Helmholtz Data Challenges platform found following with the following link http://helmholtz-data-challenges.de. The platform will be available after April 30. Please, check this page before preparing your submission, as some details may have changed.

# Reproducibility and Documentation
As members from the scientific community, we require participants to stick to Good Scientific Practice when submitting a solution. In practice, this requires to fulfill the following requirements
- Your code was run from a clean git repository, and the command to execute the run is included in the logs.
- The logs contain the commit that is currently checked out and information about the code history
- With the submission deadline, your code is pushed to a public git repository, where the commit ID of your code and its history are visible. The public repo can contain further commits with documentation, but it must be possible to roll back to the state of your submission.
- The organizing committee must be able to reproduce your time/energy stats within an acceptable accuray.
- If data preprocessing steps are involved, the corresponding preprocessing scripts must be included in the submission commit. Checksums of the processed data files must be added, too.
- The applied method is soundly documented and method attribution suffices in scientific standards.
- Only the provided *train* data can be used to train the model.

Note that you can increase the visibility of your solution by creating a repository in the challenge official repo https://gitlab.hzdr.de/count-me-in-challenge. 

