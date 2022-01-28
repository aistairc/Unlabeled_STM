# Unlabeled_STM

This sample scripts are described in the paper "Subject-transfer framework with unlabeled data based on multiple distance measures for surface electromyogram pattern recognition" accepeted to <i>Biomedical Signal Processing and Control</i>.<br />

__\<Description\>__<br />
After changing information about your directories in main_script.m (lines 6 and 8), downloading getxxfeat.m, and installing LIBSVM package, you can use this codes.<br />
This project has three folders:<br />
1. data<br />
    - private<br />
        - 22-class (1-DoF 8-class and 2-DoF 14-class) EMG data from 25 subjects<br />
            - The detail is described in <a href="https://github.com/Suguru55/SS-STM_for_MyoDatasets" target="_blank">here</a><br />
            - We only used 1-DoF 8-class data (i.e., from M1T1.csv to M8T5.csv)<br />
        - csv files (each data has 5-s information: the last 1 s is already cut)<br />
        - M means the motion label (e.g., M1 means resting state and M2 means wrist flexion)<br />
        - T means the number of trials<br />
        - After applying preprocessing_ds1.m, F_c.mat will be added in the folter .../data/private.<br />
    - NinaPro DB5 exerciseA<br />
        - 12-class finger motions from 10 subjects<br />
            - You can get this datasets from <a href="https://zenodo.org/record/1000116#.YNU4m-j7RPY" target="_blank">here</a><br />
        - Put SX_E1_A1.mat in each subject's folder<br />
        - S means the subject<br />
        - After applying preprocessing_ds2.m, F_c.mat will be added in the folter .../data/NinaPro DB5 exerciseA<br />
    - NinaPro DB5 exerciseB<br />
        - 17-class hand and wrist motions from 10 subjects<br />
        - Put SX_E2_A1.mat in each subject's folder<br />
        - S means the subject<br />
        - After applying preprocessing_ds3.m, F_c.mat will be added in the folter .../data/NinaPro DB5 exerciseB<br />
    - NinaPro DB5 exerciseC<br />
        - 23-class hand and wrist motions from 10 subjects<br />
        - Put SX_E3_A1.mat in each subject's folder<br />
        - S means the subject<br />
        - After applying preprocessing_ds4.m, F_c.mat will be added in the folter .../data/NinaPro DB5 exerciseC<br />

2. code<br />
    this folder has one main m.file named main_script that uses:<br />
    - set_config<br />
    - preprocessing_ds1<br />
        - extract_features<br />
        you can get the following m.files from <a href="http://www.sce.carleton.ca/faculty/chan/index.php?page=matlab" target="_blank">here</a><br />
            - getrmsfeat<br />
            - getmavfeat<br />
            - getzcfeat<br />
            - getsscfeat<br />
            - getwlfeat<br />
            - getarfeat<br />
    - preprocessing_ds2<br />
        - extract_features<br />
    - preprocessing_ds3<br />
        - extract_features<br />
    - preprocessing_ds4<br />
        - extract_features<br />
    - evaluate_lda_acc<br />
    - evaluate_svm_acc<br />
        you can make the m scripts, svmtrain.m and svmpredict.m from <a href="https://www.csie.ntu.edu.tw/~cjlin/libsvm/#download" target="_blank">here</a><br />
        - svmtrain (LIBSVM)<br />
        - svmpredict (LIBSVM)<br />
        - supervised_STM<br />
            - find_target<br />
            - calculate_A_b<br />
    - evaluate_lda_mdms<br />
        - svmtrain (LIBSVM)<br />
        - svmpredict (LIBSVM)<br />
    - evaluate_svm_mdms<br />
        - svmtrain (LIBSVM)<br />
        - svmpredict (LIBSVM)<br />
        - supervised_STM<br />
            - find_target<br />
            - calculate_A_b<br />
    - evaluate_lda_random<br />
    - evaluate_svm_random<br />
        - svmtrain (LIBSVM)<br />
        - svmpredict (LIBSVM)<br />
    - visualize_results<br />
        
3. resuts<br />
    this folder will store results_lda/svm_acc/mdms/random_dsx.mat and boxplots.fig.<br />

__\<Environments\>__<br />
Windows 10<br />
MATLAB R2020a<br />
    1. Signal Processing Toolbox<br />
    2. Statics and Machine Learning Toolbox<br />
    3. Parallel Comupting Toolbox<br />
