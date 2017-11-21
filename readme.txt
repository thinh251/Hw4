Contributors :

Amer Rez            012363767
Kunal Deshmukh      012324741
Thinh Nguyen        011014146


conv_train file takes the arguments as below :
<mode> <network_description> <epsilon> <max_updates> <class_letter> <model_file_name> <data_folder>

This file produces a graph for Question 1 among other statistics like Training
Cost,Validation cost as well as training and validation statistics.

Possible modes are :
cross,cross-l1,cross-l2,test. test mode is can be used to test a already
developed model using test data.
If test mode is used epsilon, max_updates will be required.