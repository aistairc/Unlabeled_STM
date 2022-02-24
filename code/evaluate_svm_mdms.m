function evaluate_svm_mdms(config)

for dataset_ind = 1:length(config.dataset_names)
   
    %%%%%%%%%%%%%%%%
    % set local config %
    %%%%%%%%%%%%%%%%
    data_dir = [config.data_dir, '\', config.dataset_names{dataset_ind}];
    code_dir = config.code_dir;
    save_dir = config.save_dir;
    sub_num = config.sub_num(dataset_ind);
    nb_senator = config.nb_senator(dataset_ind);
    mov_num = config.mov_num(dataset_ind);
    trial_num = config.trial_num(dataset_ind);
    nb_init = 15;  % find center of cluster for each class

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % load features, labels, and optimized parameters %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    filename = ['best_parameters_svm_mdms_ds', num2str(dataset_ind), '.mat'];
    cd(data_dir);
    load(['F_c.mat']);
    load(filename);
    cd(code_dir);
    feat_dim = size(F{1,1,1},2);
    C = best_C; kernel_para = best_kernel_para;
    beta = best_beta; gamma = best_gamma;
    
    %%%%%%%%%%
    % buffer %
    %%%%%%%%%%
    acc_svm = zeros(1, sub_num);
    acc_svm_transfered = zeros(1, sub_num);
    pred_svm = cell(1, sub_num);
    pred_svm_transfered = cell(1, sub_num);
    local_z_mu = zeros(sub_num, feat_dim);
    local_z_sigma = zeros(sub_num, feat_dim);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%
    % train individual SVMs %
    %%%%%%%%%%%%%%%%%%%%%%%%%
    SVMs = [];
    
    for sub_ind = 1:sub_num
        % preparation
        data = []; label = [];
        for trial_ind = 1:trial_num
            for mov_ind = 1:mov_num
                data = [data; F{sub_ind, trial_ind, mov_ind}];
                label = [label; c{sub_ind, trial_ind, mov_ind}];
            end
        end
        
        cmd = ['-q -s 0 -t 2 -b 1 -c ', num2str(C), '-g ', num2str(kernel_para)]; % RBF kernel (with probability estimates)
    
        % normalization
        [ZF, local_z_mu(sub_ind,:), local_z_sigma(sub_ind,:)] = zscore(data);
        SVMs = [SVMs; svmtrain(label, ZF, cmd)];
    end
    
    disp(['svm mdms dataset', num2str(dataset_ind), ': train ', num2str(sub_num), ' SVMs done'])
    
    %%%%%%%%%%%%%%
    % evaluation %
    %%%%%%%%%%%%%%
    for sub_ind = 1:sub_num
    
        % preparation
        sub_ind_seq = 1:1:sub_num;
        sub_ind_seq(sub_ind) = [];       
        S_cal = []; L_cal = [];
        S_tes = []; L_tes = [];      
        
        if dataset_ind == 1
            for mov_ind = 1:mov_num
                S_cal = [S_cal; F{sub_ind, 1, mov_ind}]; % 1st trial
                L_cal = [L_cal; c{sub_ind, 1, mov_ind}];
            end
        
            for trial_ind = 3:5
               for mov_ind = 1:mov_num
                    S_tes = [S_tes; F{sub_ind, trial_ind, mov_ind}]; % 3rd to 5th trial
                    L_tes = [L_tes; c{sub_ind, trial_ind, mov_ind}];
                end
            end
        else
            for trial_ind = 1:2
                for mov_ind = 1:mov_num
                    S_cal = [S_cal; F{sub_ind, trial_ind, mov_ind}]; % 1st and 2nd trials
                    L_cal = [L_cal; c{sub_ind, trial_ind, mov_ind}];
                end
            end
        
            for trial_ind = 5:6
                for mov_ind = 1:mov_num
                    S_tes = [S_tes; F{sub_ind, trial_ind, mov_ind}]; % 5th and 6th trials
                    L_tes = [L_tes; c{sub_ind, trial_ind, mov_ind}];
                end
            end 
        end
               
        [nb_cal, ~] = size(S_cal);
        [nb_tes, ~] = size(S_tes);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % source selection based on multiple distance measures %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        est_XPs = zeros(1, sub_num-1);

        for i = 1:sub_num-1  
            MDMs = zeros(1, config.mdms_num);
        
            % train tranferability predictive model
            MDMs_train = zeros(sub_num-2, config.mdms_num);
            XPs_train = zeros(1, sub_num-2);
        
            sub_source = [];
            for trial_ind = 1:trial_num
                for mov_ind = 1:mov_num
                    sub_source = [sub_source; F{sub_ind_seq(i), trial_ind, mov_ind}];
                end
            end
            sub_source = (sub_source - local_z_mu(sub_ind_seq(i), :)) ./ local_z_sigma(sub_ind_seq(i), :);
            median_sub_source = median(sub_source);
            hist_sub_source = hist(median_sub_source, length(median_sub_source))/size(median_sub_source,2);
            hist_sub_source = hist_sub_source + eps;
            
            sub_ind_seq_dammy = sub_ind_seq;
            sub_ind_seq_dammy(i) = [];
                  
            for j = 1:sub_num-2
                another_source = []; another_source_L = [];
                for trial_ind = 1:trial_num
                    for mov_ind = 1:mov_num
                        another_source = [another_source; F{sub_ind_seq_dammy(j), trial_ind, mov_ind}];
                        another_source_L = [another_source_L; c{sub_ind_seq_dammy(j), trial_ind, mov_ind}];
                    end
                end
                
                another_source = (another_source - local_z_mu(sub_ind_seq(i), :)) ./ local_z_sigma(sub_ind_seq(i), :);
            
                median_another_source = median(another_source);
                hist_another_source = hist(median_another_source, length(median_another_source))/size(median_another_source,2);
                hist_another_source = hist_another_source + eps;
                        
                % Euclidian distance
                MDMs_train(j, 1) = sum((hist_sub_source - hist_another_source).^2);
        
                % Correlation distance
                temp = corrcoef(hist_sub_source, hist_another_source);
                MDMs_train(j, 2) = temp(1,2);
        
                % Chebyshev distance
                MDMs_train(j, 3) = max(abs(hist_sub_source - hist_another_source));
        
                % Cosine distance
                MDMs_train(j, 4) = 1 - (sum(hist_sub_source.*hist_another_source) / (sqrt(sum(hist_sub_source.^2))*(sqrt(sum(hist_another_source.^2)))));
                                         
                % Kullback-Leibler divergence
                KL1 = sum(hist_sub_source .* (log2(hist_sub_source)-log2(hist_another_source)));
                KL2 = sum(hist_another_source .* (log2(hist_another_source)-log2(hist_sub_source)));
                MDMs_train(j, 5) = (KL1+KL2)/2;      
            
                % Cross-subjecct model performance
                [pred, ~, ~] = svmpredict(another_source_L, another_source, SVMs(sub_ind_seq(i)), '-q');
                Acc = sum(pred==another_source_L)/length(another_source_L);             
                XPs_train(j) = Acc;
            end
        
            % linear epcilon SVR
            cmd = ['-q -s 3 -t 0 -c ', num2str(best_C)];
            [Z_MDMs_train, MDMs_mu, MDMs_sigma] = zscore(MDMs_train);
            SVR = svmtrain(XPs_train', Z_MDMs_train, cmd);
        
            % estimate transferability for target data
            S_cal_dammy = (S_cal - local_z_mu(sub_ind_seq(i), :)) ./ local_z_sigma(sub_ind_seq(i), :);
            median_cal = median(S_cal_dammy);                                   % calculate median across windows 
            hist_cal = hist(median_cal, length(median_cal))/size(median_cal,2); % probability density function (sum equal to 1)
            hist_cal = hist_cal + eps;
                
            temp = corrcoef(hist_sub_source, hist_cal);
            KL1 = sum(hist_sub_source .* (log2(hist_cal)-log2(hist_cal)));
            KL2 = sum(hist_cal .* (log2(hist_cal)-log2(hist_sub_source)));
        
            MDMs(1) = sum((hist_sub_source - hist_cal).^2);
            MDMs(2) = temp(1,2);
            MDMs(3) = max(abs(hist_sub_source - hist_cal));
            MDMs(4) = 1 - (sum(hist_sub_source.*hist_cal) / (sqrt(sum(hist_sub_source.^2))*(sqrt(sum(hist_cal.^2)))));                                 
            MDMs(5) = (KL1+KL2)/2;                                         
        
            % estimate transferability
            Z_MDMs = (MDMs- MDMs_mu) ./ MDMs_sigma;
            est_XPs(i) = svmpredict(1, Z_MDMs, SVR,'-q');
        end
   
        % select senator SVMs
        [~, b] = sort(est_XPs,'descend');
        senator_index = b(1:nb_senator);
        senator_index_lib{sub_ind} = senator_index; 
        
        % use selected sources (we already know which data is similar to the target data by optimization process)
        naive_prob = zeros(length(L_tes), mov_num);
        transfered_prob = zeros(length(L_tes), mov_num);
        senator_index = senator_index_lib{sub_ind};
        
        for nb_ind = 1:nb_senator
            % nomalization
            S_cal_dammy = (S_cal - local_z_mu(sub_ind_seq(senator_index(nb_ind)), :)) ./ local_z_sigma(sub_ind_seq(senator_index(nb_ind)), :);
            S_tes_dammy = (S_tes - local_z_mu(sub_ind_seq(senator_index(nb_ind)), :)) ./ local_z_sigma(sub_ind_seq(senator_index(nb_ind)), :);
                    
            % naive performance
            [~, ~, temp_prob] = svmpredict(L_tes, S_tes_dammy, SVMs(sub_ind_seq(senator_index(nb_ind))), '-q -b 1');
            naive_prob = naive_prob + temp_prob;
            
            % style transfer mapping
            source = []; source_L = [];
            
            for trial_ind = 1:trial_num
                for mov_ind = 1:mov_num
                    source = [source; F{sub_ind_seq(senator_index(nb_ind)), trial_ind, mov_ind}];
                    source_L = [source_L; c{sub_ind_seq(senator_index(nb_ind)), trial_ind, mov_ind}];
                end
            end
            
            % transfer to make S_val more familiar to source SVM
            train_x = zscore(source);
            train_y = source_L;
                    
            [S_transfered] = supervised_STM(train_x, train_y, S_cal_dammy, L_cal, S_tes_dammy, nb_init, mov_num, beta, gamma);
            [~, ~, temp_prob] = svmpredict([L_cal; L_tes], S_transfered, SVMs(sub_ind_seq(senator_index(nb_ind))), '-q -b 1');
            temp_prob = temp_prob(nb_cal+1:nb_cal+nb_tes,:);
            transfered_prob = transfered_prob + temp_prob;
            % weighted ensemble strategy
            %transfered_prob = transfered_prob + temp_prob*weights(senator_index(nb_ind));
        end
        
        % recognition
        [~, pred_naive_temp] = max(naive_prob');
        pred_svm{sub_ind} = pred_naive_temp;
        acc_svm(sub_ind) = sum(pred_naive_temp'==L_tes)/length(L_tes);
        
        [~, pred_transfered_temp] = max(transfered_prob');
        pred_svm_transfered{sub_ind} = pred_transfered_temp;
        acc_svm_transfered(sub_ind) = sum(pred_transfered_temp'==L_tes)/length(L_tes); 
        
        disp(['mdms dataset ', num2str(dataset_ind), ', sub ', num2str(sub_ind)])
        disp(['Naive acc (SVM) = ', num2str(acc_svm(sub_ind)), ', Transfered acc = ', num2str(acc_svm_transfered(sub_ind))]);
    end

    %%%%%%%%%%%%%%%%
    % save results %
    %%%%%%%%%%%%%%%%
    cd(save_dir);
    filename = ['results_svm_mdms_ds', num2str(dataset_ind)];
    save(filename,'acc_svm', 'acc_svm_transfered', 'senator_index_lib',...
                  'pred_svm', 'pred_svm_transfered','best_beta','best_gamma',...
                  'best_C', 'best_kernel_para');
    cd(code_dir);
end