function experiment_template_belta(dataInput, formulaLength)

if strcmp('ecg', dataInput)
    num_features = 2;
    t = [1:10];
    train_range = 1:104;
    % train_range = 1:50;
    test_range = 104:133;

    good_data_files = ["ecg/hr_traces_len_10/1_nsr_traces_len_10.csv",
        "ecg/hr_traces_len_10_diff/1_nsr_traces_len_10.csv"];
    bad_data_files = ["ecg/hr_traces_len_10/4_afib_traces_len_10.csv",
        "ecg/hr_traces_len_10_diff/4_afib_traces_len_10.csv"];
    % normalizers = [100.1227, 60.437];
    normalizers = [1, 1];
    Plimit = [[0, 150]; [-100, 100]];

elseif strcmp('hapt', dataInput)
    num_features = 1;
    t = [1:10];
    train_range = 1:80;
    test_range = 81:100;

    good_data_files = ["hapt_data/good_hapt_data_cs.csv"];
    bad_data_files = ["hapt_data/bad_hapt_data_cs.csv"];
    normalizers = [1];
    Plimit = [-2, 2];

elseif strcmp('cruise', dataInput)
    num_features = 1;
    t = [1:101];
    train_range = 1:800;
    %train_range = 1:50;
    test_range = 801:1000;

    good_data_files = ["cruise_control_train/Traces_normal_BIG2.csv"];
    bad_data_files = ["cruise_control_train/Traces_anomaly_BIG2.csv"];
    normalizers = [1];
    Plimit = [5, 50];
    
elseif strcmp('lyft', dataInput)
    num_features = 4;
    % num_features = 2;
    t = [1:50];
    train_range = 1:8080;
    test_range = 8081:10100;
    %train_range = 1:50;

    good_data_files = ["lyft_data/lyft_good_traces_x.csv",
        "lyft_data/lyft_good_traces_x_diff.csv",
        "lyft_data/lyft_good_traces_y.csv",
        "lyft_data/lyft_good_traces_y_diff.csv"];
    bad_data_files = ["lyft_data/bad_traces_x.csv",
        "lyft_data/bad_traces_x_diff.csv",
        "lyft_data/bad_traces_y.csv",
        "lyft_data/bad_traces_y_diff.csv"];
%     good_data_files = ["lyft_data/lyft_good_traces_x_diff.csv",
%         "lyft_data/lyft_good_traces_y_diff.csv"];
%     bad_data_files = ["lyft_data/bad_traces_x_diff.csv",
%         "lyft_data/bad_traces_y_diff.csv"];
    % normalizers = [100.1227, 60.437];
    normalizers = [1, 1, 1, 1];
    % normalizers = [1, 1];
    Plimit = [[-50, 50]; [-50, 50]; [-50, 50]; [-50, 50]];

end

% READ DATA 
good_data = [];
bad_data = [];
for i = 1:length(good_data_files)
    good = csvread(good_data_files(i));
    % good_data = [good_data; good];
    good_data = [good_data; good; good];

    bad = csvread(bad_data_files(i));
    % bad_data = [bad_data; bad];
    bad_data = [bad_data; bad; bad];
end 

% good_data = reshape(good_data, [], num_features, 10);
% bad_data = reshape(bad_data, [], num_features, 10);
good_data = reshape(good_data, [], num_features*2, max(t));
bad_data = reshape(bad_data, [], num_features*2, max(t));
good_data = good_data(train_range,:,:);
bad_data = bad_data(train_range,:,:);
traces = [good_data; bad_data];

V = [1:num_features];
L_max = formulaLength;
P = traces;
s = [ones(size(good_data, 1), 1); zeros(size(good_data, 1), 1)];
% trunc = 1;
trunc = max(t) / 2;
Ns = [10, 1];
delta = size(P, 1) * .2;
J_max = 100;
Tlimit = [trunc, max(t)];
G = [];

% ClassandPred(V,L_max,P, s, t, trunc, Plimit, Ns, delta, J_max);

% % [phi_e,val_e,mn_e] = findformula(V,L_max,P,s,t,[trunc,max(t)],Plimit,Ns,[],delta,J_max);
% [phi_e,val_e,mn_e] = findformula(V,L_max,P,s,t,Tlimit,Plimit,Ns,G,delta,J_max);
% disp(['Classification complete.  Classifying formula is ','$',interpret(phi_e,val_e),'$']);
% disp('Number of misclassified \n');
% disp(mn_e);

% TRY CATCH FOR BUG, RUN 10 TRIALS, RETURN 1 WITH BEST MISCLASSIFICATION
% RATE (FOR THE TRAIN DATA)

mcr = ones(1, 10);
phis = {};
vals = {};
errors = {};
tic
for i = 1:10
% for i = 1:10
%     [phi_e,val_e,mn_e] = findformula(V,L_max,P,s,t,Tlimit,Plimit,Ns,G,delta,J_max);
%     mcr(i) = mn_e;
%     phis{i} = phi_e;
%     vals{i} = val_e;
    try 
        [phi_e,val_e,mn_e,interp] = ClassandPred(V,L_max,P, s, t, trunc, Plimit, Ns, delta, J_max);
        % [phi_e,val_e,mn_e] = findformula(V,L_max,P,s,t,Tlimit,Plimit,Ns,G,delta,J_max);
        mcr(i) = mn_e;
        phis{i} = phi_e;
        vals{i} = val_e;
    catch ERROR
        errors{i} = ERROR;
        mcr(i) = 9999;
        phis{i} = '';
        vals{i} = [];
    end
end

elapsed = toc;

% fileID = fopen(strcat(dataInput, string(formulaLength), '_belta.txt'),'w');
% 
% fprintf(fileID, 'Elapsed time for belta code:\n');
% fprintf(fileID, '%f\n',elapsed);
% 
% [mcr_value, mcr_argmin] = min(mcr);
% best_phi = phis{mcr_argmin};
% best_vals = vals{mcr_argmin};
% fprintf(fileID, ['Classification complete.  Classifying formula is ','$',interpret(best_phi,best_vals),'$\n']);
% fprintf(fileID, 'Number of misclassified %d \n', mcr_value);
% fprintf(fileID, '\n');
% 
% fprintf(fileID, 'Errors it ran into \n');
% for i=1:length(errors)
%     if isa(errors{i}, 'double')
%         continue
%     else
%         fprintf(fileID, errors{i}.message);
%         fprintf(fileID, '\n');
%     end
% %     disp(errors{i})
end
