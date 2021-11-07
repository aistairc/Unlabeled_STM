%% Clear workspace
clear all
close all

%% Set config
main_dir = 'C:\Users\eyed164\Documents\GitHub\Unlabeled_STM'; % change to your directory
config = set_config(main_dir);
addpath('C:\Users\eyed164\Desktop\BSCP_MDM\libsvm-3.23\matlab'); % change to your directory

%% Preprocessing
preprocessing_ds1(config); % private
preprocessing_ds2(config); % NinaPro DB5 exercise A
preprocessing_ds3(config); % NinaPro DB5 exercise B
preprocessing_ds4(config); % NinaPro DB5 exercise C

%% Evaluation
evaluate_lda_acc(config);
evaluate_svm_acc(config);

evaluate_lda_mdms(config)
evaluate_svm_mdms(config);

evaluate_lda_random(config);
evaluate_svm_random(config);

%% Visualization
visualize_results(config);