function visualize_results(config)

for dataset_ind = 1:length(config.dataset_names)
    disp(['DB ', num2str(dataset_ind)]);
    
    %%%%%%%%%%%%%%%%%%%%
    % set local config %
    %%%%%%%%%%%%%%%%%%%%
    save_dir = config.save_dir;
    sub_num = config.sub_num(dataset_ind);
    boxplot_lib = zeros(9, sub_num);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % load and rename results %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 1: random lda, 2; random svm,
    % 3: mdms lda, 4: mdms svm, 5: mdms stm-svm,
    % 6: acc lda, 7: acc svm, 8: acc csa-LDA, 9: acc stm-svm
    cd(save_dir);
    
    % lda_acc
    load(['results_lda_acc_ds', num2str(dataset_ind)]); 
    boxplot_lib(6, :) = acc_lda;
    boxplot_lib(8,:) = acc_lda_transfered;
    disp(['LDA acc: ', num2str(mean(acc_lda)), ', CSA-LDA acc: ', num2str(mean(acc_lda_transfered))]);
    
    % lda_mdms
    load(['results_lda_mdms_ds', num2str(dataset_ind)]); 
    boxplot_lib(3,:) = acc_lda;
    disp(['LDA mdms: ', num2str(mean(acc_lda))]);
    
    % lda_random
    load(['results_lda_random_ds', num2str(dataset_ind)]); 
    boxplot_lib(1,:) = acc_lda;
    disp(['LDA random: ', num2str(mean(acc_lda))]);
    
    % svm_acc
    load(['results_svm_acc_ds', num2str(dataset_ind)]); 
    boxplot_lib(7,:) = acc_svm;
    boxplot_lib(9,:) = acc_svm_transfered;
    disp(['SVM acc: ', num2str(mean(acc_svm)), ', STM-SVM acc: ', num2str(mean(acc_svm_transfered))]);
    
    % svm_mdms
    load(['results_svm_mdms_ds', num2str(dataset_ind)]); 
    boxplot_lib(4,:) = acc_svm;
    boxplot_lib(5,:) = acc_svm_transfered;
    disp(['SVM mdms: ', num2str(mean(acc_svm)), ', STM-SVM mdms: ', num2str(mean(acc_svm_transfered))]);
    
    % svm_random
    load(['results_svm_random_ds', num2str(dataset_ind)]); 
    boxplot_lib(2,:) = acc_svm;
    disp(['SVM random: ', num2str(mean(acc_svm))]); 
    
    figure(1)
    subplot(2,2,dataset_ind)
    boxplot(boxplot_lib')
    xline(2.5, '--k', 'LineWidth', 1.5); xline(5.5, '--k', 'LineWidth', 1.5);
    ylim([0 1.1])
    ylabel('Acc')
    grid on
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 16);
end

savefig('boxplots.fig');
cd(config.code_dir);