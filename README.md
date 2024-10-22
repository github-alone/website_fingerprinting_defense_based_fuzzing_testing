# website_fingerprinting_defense_based_fuzzing_testing
website_fingerprinting_defense_based_fuzz_testing

1.Dataset:

We publish the datasets of web traffic traces produced for the closed-world evaluations on non-defended. 

The dataset can be downloaded from this link:  https://www.alipan.com/s/H1mE22FSCeV

2. Source Code's Description

The source codes contain two part:

①generate the defended trace: python main/ClosedWorld_gen_fuzz_traffic.py [0] 0.2 45 1022 df
   
② evaluate the performance of the current strategy: python main/ClosedWorld_eval_add_two [0] 0.2 45 1022 df

The result will be saved in Folder:result/df/ClosedWorld/1022[0]_0.2_45.csv

In each row of the csv file consists of the related performance metrics with respect to different thresholds including

Bandwidth Overhead (BO), Detection Rate (DR)



