# iPVP-DRLF


# 1 Description
Plant vacuoles are the most important organelles for plant growth, development, and defense, and they play an important role in many types of stress responses. An important function of vacuole proteins is the transport of various classes of amino acids, ions, sugars, and other molecules. Accurate identification of autophagy proteins is crucial for revealing their biological functions. Here, we developed a new machine learning-based software that enables the classification of proteins into PVPs or non-PVPs specifically and effectively. This predictive software was designed using a light gradient boosting machine classifier and hybrid features composed of deep representation learning feature and adaptive skip dipeptide composition feature. It has the potential to facilitate future computational work in this field.
Webserver and datasets are also available at:
http://lab.malab.cn/~acy/iPVP-DRLF.


# 2 Requirements
Before running, please make sure the following packages are installed in Python environment:

python==3.7

pytorch==1.4.0

tape_proteins==0.5

rich==9.12.4

lightgbm==3.1.1

numpy==1.20.3

pandas==1.3.5

joblib==1.0.1

scikit-learn==1.0.2




# 3 Running
Down load model [here](http://121.40.214.145:8080/iPVP-DRLP/BiLSTM_embed.model)

Changing working dir to iAVP-DRLF-main, and then running the following command:

python iAVP-DRLF.py -i test.fasta -o prediction_results.csv

-i: input file in fasta format

-o: output file name
