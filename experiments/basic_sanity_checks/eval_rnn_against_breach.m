data1 = readmatrix('data1_trace.csv');
data2 = readmatrix('data2_trace.csv');
data3 = readmatrix('data3_trace.csv');

STL_ReadFile('./reqs.stl');

tests = [
    testUntil, ...
    testEv, ...
    testAlw, ...
    testNext, ...
    testAnd, ...
    testOr, ...
    testEvAlw, ...
    testAlwEv, ...
    testAtomSelection, ...
    testAnd, ...
    testAlwSelect, ...
    testNext];

testNames = [
    "until", ...
    "ev", ...
    "alw", ...
    "next", ...
    "and", ...
    "or", ...
    "ev_alw", ...
    "alw_ev", ...
    "select_atoms", ...
    "select_and", ...
    "select_alw", ...
    "select_nxt"];

SPECIALTESTUNTIL = 1;
SPECIALTESTNEXT = 4;
SPECIALTESTNEXTSELECT = 12;

for i = 1:length(tests)
    testSpec = tests(i);
    disp(testSpec)
    
    %j = 1;
    save_robs = [];
    for j = (1:100)
        if i == SPECIALTESTNEXT || i == SPECIALTESTNEXTSELECT
            times = 0:1:3;
            x = data1(j,1:4);
            y = data2(j,1:4);
            z = data3(j,1:4);
        else
            times = 0:1:4;
            x = data1(j,:);
            y = data2(j,:);
            z = data3(j,:);
        end
        C = [flip(x); flip(y); flip(z)];
        trace = [times', C'];
    
        S = BreachTraceSystem({'x', 'y', 'z'}, trace);
        S.SetTime(times);
    
        all_robs = ones(1, 5);
        if i == SPECIALTESTNEXT || i == SPECIALTESTNEXTSELECT
            for k = (0:3)
                 all_robs(k+1) = S.CheckSpec(testSpec, k);
            end
            all_robs(4) = S.CheckSpec(testSpec, 2.99);
            all_robs(5) = nan;
            
        else
            for k = (0:3)
                all_robs(k+1) = S.CheckSpec(testSpec, k);
            end
            all_robs(5) = S.CheckSpec(testSpec, 3.99);
        end
        save_robs = [save_robs; flip(all_robs)];
    end

    save_file_name = append('breach_predictions_', testNames(i), '.csv');
    writematrix(save_robs, save_file_name, 'Delimiter', ',');
end

