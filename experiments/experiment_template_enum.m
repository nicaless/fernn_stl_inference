function experiment_template_enum(dataInput, formulaLength, max_num)

MCR_THRESH = 0.2;

traceTimeBegin = 0;
if strcmp('ecg', dataInput)
    num_features = 2;
    numSignatureTraces = 5;
    traceTimeHorizon = 10;
    t = 0:9;
    train_range = 1:104;
    %train_range = 1:50;
    test_range = 104:133;

    good_data_files = ["ecg/hr_traces_len_10/1_nsr_traces_len_10.csv",
        "ecg/hr_traces_len_10_diff/1_nsr_traces_len_10.csv"];
    bad_data_files = ["ecg/hr_traces_len_10/4_afib_traces_len_10.csv",
        "ecg/hr_traces_len_10_diff/4_afib_traces_len_10.csv"];
    % normalizers = [100.1227, 60.437];
    normalizers = [1, 1];

    timeRange = [traceTimeBegin, traceTimeHorizon];
    valHRtestRange = [0, 150];
    s1 = struct('name', 'hr', ...
        'ops', {{'<','>'}}, ...
        'params', {{'val_hr'}}, ...
        'timeRange', timeRange, ...
        'range', valHRtestRange ...
        );
    valHRDifftestRange = [-100, 100];
    s2 = struct('name', 'hr_diff', ...
        'ops', {{'<','>'}}, ...
        'params', {{'val_hr_diff'}}, ...
        'timeRange', timeRange, ...
        'range', valHRDifftestRange ...
        );
    signals = {s1, s2};
elseif strcmp('hapt', dataInput)
    num_features = 1;
    numSignatureTraces = 5;
    traceTimeHorizon = 10;
    t = 0:9;
    train_range = 1:80;
    test_range = 81:100;

    good_data_files = ["hapt_data/good_hapt_data_cs.csv"];
    bad_data_files = ["hapt_data/bad_hapt_data_cs.csv"];
    normalizers = [1];

    timeRange = [traceTimeBegin, traceTimeHorizon];
    haptTestRange = [-1.5, 1.5];
    s1 = struct('name', 'haptx', ...
        'ops', {{'<','>'}}, ...
        'params', {{'val_haptx'}}, ...
        'timeRange', timeRange, ...
        'range', haptTestRange ...
        );
    signals = {s1};
elseif strcmp('cruise', dataInput)
    num_features = 1;
    numSignatureTraces = 5;
    traceTimeHorizon = 101;
    t = 0:100;
    train_range = 1:800;
    test_range = 801:1000;
    %train_range = 1:50;

    %good_data_files = ["cruise_control_train/Traces_normal_BIG2.csv"];
    %bad_data_files = ["cruise_control_train/Traces_anomaly_BIG2.csv"];
    bad_data_files = ["cruise_control_train/Traces_normal_BIG2.csv"];
    good_data_files = ["cruise_control_train/Traces_anomaly_BIG2.csv"];
    normalizers = [1];

    timeRange = [traceTimeBegin, traceTimeHorizon];
    cruiseTestRange = [5, 50];
    s1 = struct('name', 'v', ...
        'ops', {{'<','>'}}, ...
        'params', {{'val_v'}}, ...
        'timeRange', timeRange, ...
        'range', cruiseTestRange ...
        );
    signals = {s1};
elseif strcmp('lyft', dataInput)
    num_features = 4;
    % num_features = 2;
    numSignatureTraces = 5;
    traceTimeHorizon = 50;
    t = 0:49;
    %train_range = 1:8080;
    %test_range = 8081:10100;
    train_range = 1:800;
    test_range = 8081:8281;

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

    timeRange = [traceTimeBegin, traceTimeHorizon];
    valLyftxRange = [-50, 50];
    s1 = struct('name', 'lyftx', ...
        'ops', {{'<','>'}}, ...
        'params', {{'val_lyftx'}}, ...
        'timeRange', timeRange, ...
        'range', valLyftxRange ...
        );
    valLyftxDiffRange = [-50, 50];
    s2 = struct('name', 'lyftx_diff', ...
        'ops', {{'<','>'}}, ...
        'params', {{'val_lyftx_diff'}}, ...
        'timeRange', timeRange, ...
        'range', valLyftxDiffRange ...
        );
    valLyftyRange = [-50, 50];
    s3 = struct('name', 'lyfty', ...
        'ops', {{'<','>'}}, ...
        'params', {{'val_lyfty'}}, ...
        'timeRange', timeRange, ...
        'range', valLyftxRange ...
        );
    valLyftyDiffRange = [-50, 50];
    s4 = struct('name', 'lyfty_diff', ...
        'ops', {{'<','>'}}, ...
        'params', {{'val_lyfty_diff'}}, ...
        'timeRange', timeRange, ...
        'range', valLyftxDiffRange ...
        );
    signals = {s1, s2, s3, s4};
    % signals = {s2, s4};
end

% READ DATA
good_data = {};
bad_data = {};
for i = 1:length(good_data_files)
    good = csvread(good_data_files(i));
    good_data{end+1} = good;

    bad = csvread(bad_data_files(i));
    bad_data{end+1} = bad;
end

Traces0 = makeBreachTraceSystem(signals);%anomalous
Traces1 = makeBreachTraceSystem(signals);%normal

Traces0_test = makeBreachTraceSystem(signals);%anomalous
Traces1_test = makeBreachTraceSystem(signals);%normal

options.numSignatureTraces = numSignatureTraces;
numTraces = 10;
yapp = YAP(signals, numTraces, options);

% ADD TO TRACES FROM DATA
for jj = train_range
    good_trace = [t'];
    bad_trace = [t'];

    for i = 1:length(good_data_files)
        good_data_feature = good_data{i};
        if jj <= length(good_data_feature)
            % x = good_data_feature(jj,:) / normalizers(i);
            x = good_data_feature(jj,:);
            good_trace = [good_trace, x'];
        end

        bad_data_feature = bad_data{i};
        % y = bad_data_feature(jj,:) / normalizers(i);
        y = bad_data_feature(jj,:);
        bad_trace = [bad_trace, y'];
    end

    if jj <= length(good_data_feature)
        Traces1.AddTrace(good_trace);
    end
    yapp.addTrace(jj, good_trace);
    Traces0.AddTrace(bad_trace);
end

for jj = test_range
    good_trace = [t'];
    bad_trace = [t'];

    for i = 1:length(good_data_files)
        good_data_feature = good_data{i};
        if jj <= length(good_data_feature)
            % x = good_data_feature(jj,:) / normalizers(i);
            x = good_data_feature(jj,:);
            good_trace = [good_trace, x'];
        end

        bad_data_feature = bad_data{i};
        % y = bad_data_feature(jj,:) / normalizers(i);
        y = bad_data_feature(jj,:);
        bad_trace = [bad_trace, y'];
    end

    if jj <= length(good_data_feature)
        Traces1_test.AddTrace(good_trace);
    end
    Traces0_test.AddTrace(bad_trace);
end

% RUN ENUM METHOD
% j=1;
% done=0;
% f = formulaIterator(formulaLength, signals);
% max_num_enumerating_formulas = max_num;
% 
% best_mcr = 1;
% best_formula = 0;
% best_params = 0;
% best_param_values = 0;
% tic
% 
% while (j < max_num_enumerating_formulas)
%     formula = f.nextFormula();
%     if isempty(fieldnames(formula))
%         break
%     end
%     yapp.addTimeParams(formula);
% 
%     % check the equivalence of formulas
%     if (yapp.isNew(formula))
%         disp(formula)
% 
%         % SKIP ATOMS ONLY
%         if length(fieldnames(get_params(formula))) == 1
% %         if length(fieldnames(get_params(formula))) < formulaLength
%             j = j + 1;
%             continue
%         end
% 
%         % set parameter ranges
%         params = fieldnames(get_params(formula));
%         numparam=length(params);
%         paramranges = zeros(numparam,2);
% 
%         for i=1:numparam
%             if string(params(i,1)) == "val_hr"
%                 paramranges (i,:)=valHRtestRange;
%             elseif string(params(i,1)) == "val_hr_diff"
%                 paramranges (i,:)=valHRDifftestRange;
%             elseif string(params(i,1)) == "val_haptx"
%                 paramranges (i,:)=haptTestRange;
%             elseif string(params(i,1)) == "val_v"
%                 paramranges (i,:)=cruiseTestRange;
%             elseif string(params(i,1)) == "val_lyftx"
%                 paramranges (i,:)=valLyftxRange;
%             elseif string(params(i,1)) == "val_lyftx_diff"
%                 paramranges (i,:)=valLyftxDiffRange;
%             elseif string(params(i,1)) == "val_lyfty"
%                 paramranges (i,:)=valLyftyRange;
%             elseif string(params(i,1)) == "val_lyfty_diff"
%                 paramranges (i,:)=valLyftyDiffRange;
%             elseif string(params(i,1)) == "tau_1"
%                 % paramranges (i,:)=[100,100];
%                 paramranges (i,:)=[0,traceTimeHorizon];
%             elseif string(params(i,1)) == "tau_2"
%                 % paramranges (i,:)=[100,100];
%                 paramranges (i,:)=[0,traceTimeHorizon];
%             elseif string(params(i,1)) == "tau_3"
%                 % paramranges (i,:)=[100,100];
%                 paramranges (i,:)=[0,traceTimeHorizon];
%             end
%         end
%         % Monotonic_Bipartition function parameters
%         uncertainty=10e-3;
%         num_steps = 3;
%         n = numparam;
% 
%         % < means monoDir should be 0
%         % > means monoDir should be 1
%         formula_str = disp(formula);
%         loc_leq = strfind(formula_str, '<');
%         loc_geq = strfind(formula_str, '>');
%         all_ineq = [loc_leq, loc_geq];
%         monoDir1 = [];
%         for loc = 1:length(all_ineq)
%             ineq = all_ineq(loc);
%             if strcmp('<', formula_str(ineq))
%                 monoDir1 = [monoDir1, 0];
%             else
%                 monoDir1 = [monoDir1, 1];
%             end
%         end
%         count_time_bounds = count(formula_str, 'tau');
%         for taus = 1:count_time_bounds
%             monoDir1 = [monoDir1, 1];
%         end
% 
%         disp(monoDir1)
% 
% 
%         % if (all(monoDir1)>=0)
%         if (all(monoDir1>=0))
% 
%             % obtain validity domain boundary
%             mono=Monotonic_Bipartition(formula,paramranges,num_steps,uncertainty,Traces1,monoDir1);
%             c1=reshape(mono.boundry_points,numparam,size(mono.boundry_points,2)/numparam)';
% 
%             % check points on validity domain boundary and choose
%             % the point with MCR = 0
%             for i = 1:size (c1,1)
%                 formula = set_params(formula,params, c1(i,:));
%                 robustness1(i,:)=Traces1.CheckSpec(formula);
%                 robustness2(i,:)=Traces0.CheckSpec(formula);
% 
%                 if and(all(robustness1(i,:)> 0),all(robustness2(i,:)<0))
%                     fprintf('\n\n');
%                     fprintf('The learned STL formula is:\n');
%                     fprintf('\n');
%                     fprintf(disp(formula));
%                     fprintf('\n\n');
%                     fprintf('The values of parameters are:\n');
%                     for n = 1:size(params,1)
%                         params(n)
%                         c1(i,n)
%                         fprintf('\n');
%                     end
%                     fprintf('train MCR=0\n')
%                     fprintf('Elapsed time with signature based optimization:\n')
%                     toc
%                     done=1;
%                     best_mcr = 0;
%                     best_formula = formula;
%                     best_params = params;
%                     best_param_values = c1(i,:);
%                     break;
%                 else
% 
%                     TruePos=size(find (robustness1(i,:) > 0 == 1),2);
%                     FalsePos=size(robustness1,2)-TruePos;
% 
%                     TrueNeg=size(find (robustness2(i,:) < 0 == 1),2);
%                     FalseNeg=size(robustness2,2)-TrueNeg;
% 
%                     MCR = (FalsePos + FalseNeg)/(size(robustness1,2)+size(robustness2,2));
%                     % fprintf('MCR  = %f\n',MCR);
% 
%                     % check points on validity domain boundary and choose
%                     % the point with MCR < 0.1
%                     if MCR < best_mcr
%                         best_mcr = MCR;
%                         best_formula = formula;
%                         best_params = params;
%                         best_param_values = c1(i,:);
%                         fprintf('Best MCR so far is %f\n', MCR);
%                         fprintf('Best formula so far is\n');
%                         fprintf(disp(formula));
%                         fprintf('The values of best params so far are:\n');
%                         for n = 1:size(params,1)
%                             params(n)
%                             c1(i,n)
%                             fprintf('\n');
%                         end
%                     end
% 
%                     if MCR < MCR_THRESH
%                         best_mcr = MCR;
%                         best_formula = formula;
%                         best_params = params;
%                         best_param_values = c1(i,:);
% 
%                         fprintf('\n\n');
%                         fprintf('The learned STL formula is:\n');
%                         fprintf('\n');
%                         fprintf(disp(formula));
%                         fprintf('\n\n');
%                         fprintf('The values of parameters are:\n');
%                         for n = 1:size(params,1)
%                             params(n)
%                             c1(i,n)
%                             fprintf('\n');
%                         end
%                         fprintf('train MCR = %f\n',MCR);
%                         done=1;
%                         break;
%                     end
%                 end
% 
% 
%             end
%             if done==1
%                 break;
%             end
%         else
%            fprintf('Not monotonic:\n')
%         end
%     else
%         fprintf('Equivalent Formula:\n')
%     end
%     j=j+1;
% end
% 
% % WRITE TO FILE
% 
% elapsed = toc;
% 
% fileID = fopen(strcat(dataInput, string(formulaLength), '.txt'),'w');
% fprintf(fileID, 'Elapsed time with signature based optimization:\n');
% fprintf(fileID, '%f\n',elapsed);
% 
% fprintf(fileID, 'Number formula checked:\n');
% fprintf(fileID, '%d\n',j-1);
% 
% fprintf('\n\n');
% fprintf(fileID, 'The learned STL formula is:\n');
% fprintf(fileID, '\n');
% if strcmp(class(best_formula), 'double')
%     fprintf(fileID, 'No formula found. All templates produce not monotonic');
% else
%     fprintf(fileID, disp(best_formula));
% 
%     fprintf(fileID, '\n\n');
%     fprintf(fileID, 'The values of parameters are:\n');
%     for n = 1:size(best_params,1)
%         param_name = best_params(n);
%         fprintf(fileID, param_name{1});
%         fprintf(fileID, 'value: %f\n', best_param_values(n));
%     end
%     fprintf(fileID, 'train MCR = %f\n', best_mcr);
% 
%     final_formula = set_params(best_formula, best_params, best_param_values);
%     pos=Traces1_test.CheckSpec(final_formula);
%     neg=Traces0_test.CheckSpec(final_formula);
%     mcr_test = (size(find (pos < 0),2) + size(find (neg > 0),2))/(size(pos,2)+size(neg,2));
%     fprintf(fileID, 'test MCR = %f\n',mcr_test);
% end


%% belta ECG 2
% pos = Traces1_test.CheckSpec('not (alw_[1,5] hr[t] > 120.2061) or (alw_[6,10] hr[t] < 86.6774)');
% neg = Traces0_test.CheckSpec('not (alw_[1,5] hr[t] > 120.2061) or (alw_[6,10] hr[t] < 86.6774)');
% ECG 3
% pos = Traces1_test.CheckSpec('not (alw_[1,5] hr[t] > 104.2835) or (alw_[5,10] hr[t] < 83.7916)');
% neg = Traces0_test.CheckSpec('not (alw_[1,5] hr[t] > 104.2835) or (alw_[5,10] hr[t] < 83.7916)');
% ECG 4
% pos = Traces1_test.CheckSpec('not (alw_[1,5] hr[t] > 121.5646) or (alw_[5.3037,9.3037] hr[t] < 88.3734)');
% neg = Traces0_test.CheckSpec('not (alw_[1,5] hr[t] > 121.5646) or (alw_[5.3037,9.3037] hr[t] < 88.3734)');
% ECG 5
% pos = Traces1_test.CheckSpec('not (alw_[1,5] hr[t] > 93.7297) or (alw_[5,9] hr[t] < 91.1654)');
% neg = Traces0_test.CheckSpec('not (alw_[1,5] hr[t] > 93.7297) or (alw_[5,9] hr[t] < 91.1654)');
% ECG 6
% pos = Traces1_test.CheckSpec('not (alw_[1,5] hr[t] > 82.6567) or (alw_[5.6738,10] hr[t] < 84.2203)');
% neg = Traces0_test.CheckSpec('not (alw_[1,5] hr[t] > 82.6567) or (alw_[5.6738,10] hr[t] < 84.2203)');

%% belta hapt 2
% pos = Traces1_test.CheckSpec('not (alw_[1,5] haptx[t] > 0.93511) or (ev_[5,10] haptx[t] > -0.57077)');
% neg = Traces0_test.CheckSpec('not (alw_[1,5] haptx[t] > 0.93511) or (ev_[5,10] haptx[t] > -0.57077)');
% hapt 3
% pos = Traces1_test.CheckSpec('not (alw_[1,5] haptx[t] > 0.33364) or (ev_[6,10] haptx[t] > -0.71986)');
% neg = Traces0_test.CheckSpec('not (alw_[1,5] haptx[t] > 0.33364) or (ev_[6,10] haptx[t] > -0.71986)');
% hapt 4
% pos = Traces1_test.CheckSpec('not (alw_[1,5] haptx[t] > 1.2287) or (ev_[5.3788,10] haptx[t] > -0.61929)');
% neg = Traces0_test.CheckSpec('not (alw_[1,5] haptx[t] > 1.2287) or (ev_[5.3788,10] haptx[t] > -0.61929)');
% hapt 5
% pos = Traces1_test.CheckSpec('not (alw_[1,5] haptx[t] > -0.1119) or (alw_[5.5707,9.5707] haptx[t] > -0.88163)');
% neg = Traces0_test.CheckSpec('not (alw_[1,5] haptx[t] > -0.1119) or (alw_[5.5707,9.5707] haptx[t] > -0.88163)');
% hapt 6
% pos = Traces1_test.CheckSpec('not (alw_[1,5] haptx[t] > -0.63216) or (ev_[6,10] haptx[t] > -0.4666)');
% neg = Traces0_test.CheckSpec('not (alw_[1,5] haptx[t] > -0.63216) or (ev_[6,10] haptx[t] > -0.4666)');

%% belta cruise 2
% pos = Traces1_test.CheckSpec('not (alw_[1,50.5] v[t] > 50) or (alw_[51.5,101] v[t] < 34.8899)');
% neg = Traces0_test.CheckSpec('not (alw_[1,50.5] v[t] > 50) or (alw_[51.5,101] v[t] < 34.8899)');
% cruise 3
% pos = Traces1_test.CheckSpec('not (alw_[1,50.5] v[t] > 42.2692) or (alw_[50.5,100] v[t] < 36.3331)');
% neg = Traces0_test.CheckSpec('not (alw_[1,50.5] v[t] > 42.2692) or (alw_[50.5,100] v[t] < 36.3331)');
% cruise 4
% pos = Traces1_test.CheckSpec('not (alw_[1,50.5] v[t] > 24.7359) or (alw_[50.5,101] v[t] < 34.702)');
% neg = Traces0_test.CheckSpec('not (alw_[1,50.5] v[t] > 24.7359) or (alw_[50.5,101] v[t] < 34.702)');
% cruise 5
% pos = Traces1_test.CheckSpec('not (alw_[1,50.5] v[t] > 30.2328) or (alw_[50.5,101] v[t] < 33.0257)');
% neg = Traces0_test.CheckSpec('not (alw_[1,50.5] v[t] > 30.2328) or (alw_[50.5,101] v[t] < 33.0257)');
% cruise 6
% pos = Traces1_test.CheckSpec('not (alw_[1,50.5] v[t] > 27.0263) or (alw_[50.5,100] v[t] < 42.494)');
% neg = Traces0_test.CheckSpec('not (alw_[1,50.5] v[t] > 27.0263) or (alw_[50.5,100] v[t] < 42.494)');


%% belta lyft 2
% pos = Traces1_test.CheckSpec('not (alw_[1,25] lyftx[t] > -2.1261) or (ev_[26,50] lyftx[t] < -20.2498)');
% neg = Traces0_test.CheckSpec('not (alw_[1,25] lyftx[t] > -2.1261) or (ev_[26,50] lyftx[t] < -20.2498)');
% lyft 3
% pos = Traces1_test.CheckSpec('not (alw_[1,25] lyftx[t] > 50) or (ev_[25,49] lyftx_diff[t] > -10.3421)');
% neg = Traces0_test.CheckSpec('not (alw_[1,25] lyftx[t] > 50) or (ev_[25,49] lyftx_diff[t] > -10.3421)');
% lyft 4
% pos = Traces1_test.CheckSpec('not (alw_[1,25] lyftx[t] > -21.4819) or (alw_[25,50] lyftx_diff[t] < -50)');
% neg = Traces0_test.CheckSpec('not (alw_[1,25] lyftx[t] > -21.4819) or (alw_[25,50] lyftx_diff[t] < -50)');
% lyft 5
% pos = Traces1_test.CheckSpec('not (alw_[1,25] lyftx[t] > 23.4675) or (alw_[26,50] lyftx_diff[t] < 0.94112)');
% neg = Traces0_test.CheckSpec('not (alw_[1,25] lyftx[t] > 23.4675) or (alw_[26,50] lyftx_diff[t] < 0.94112)');
% lyft 6
% pos = Traces1_test.CheckSpec('not (alw_[1,25] lyftx[t] > -2.0946) or (ev_[25,49] lyftx_diff[t] < -46.5557)');
% neg = Traces0_test.CheckSpec('not (alw_[1,25] lyftx[t] > -2.0946) or (ev_[25,49] lyftx_diff[t] < -46.5557)');

% pos = Traces1_test.CheckSpec('alw_[40,60] (v[t] > 34.2579)');
% neg = Traces0_test.CheckSpec('alw_[40,60] (v[t] > 34.2579)'); 
pos = Traces1_test.CheckSpec('alw_[40,60] (v[t] > 40)');
neg = Traces0_test.CheckSpec('alw_[40,60] (v[t] > 40)'); 
disp(size(find (pos < 0),2));
disp(size(find (neg > 0),2));
disp((size(pos,2)+size(neg,2)));
mcr_test = (size(find (pos < 0),2) + size(find (neg > 0),2))/(size(pos,2)+size(neg,2));
fprintf('test MCR = %f\n',mcr_test);

end
