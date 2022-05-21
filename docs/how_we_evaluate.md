How we evaluate the submissions
=========
The submission deadline is set to Monday, May 23 anywhere on earth. At the deadline we will start taking a look at the leading submission. In order to do that, we will go through the following steps:

1. We sort the leaderboard in the Helmholtz Data Challenges platform by the two metrics, the execution time and the energy consumption. 
2. For the leading submissions, where also the RMSE is below 1111, we will check the slurm logs if the exeuction time and energy consumption have been reported correctly. 
3. Then, we will check the accessibility of the submitted repository links and perform a quick check that a short description of the method and a minimum documentation of the code is available.
4. For the best submissions, we will attempt to reproduce your findings based on the submitted code. Based on these findings we will decide about the winner. If the the results are tight, we will consider performing multiple runs and analyze the statistics. In that case, we also might take a closer look at the documentation. 

Note that it is currently not possible to show all submission fields correctly in the platform. With admin access, we can however completely extract your submissions. 

Furthermore a very important note: Please make our lives for reproducing the runs as easy as possible. We use the same computers, therefor it is possible bring your code into a form where we only need to clone your repositories, and submit a single script to slurm. If you do it right, this is all we need to do and then we'll be very happy! If it will be more difficult, it might be necessary to contact you, and this for sure does not make it more likely to win the challenge :-).
