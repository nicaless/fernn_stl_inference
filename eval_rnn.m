InitBreach;
traces = readmatrix('test_data.csv');
trace_size = size(traces);

STL_ReadFile('./reqs.stl');
%%
testSpec = testSpecUntil;
testNext = 0;
testUntil = 1;


for i = (1:trace_size(1))
    if testNext == 0
        C = traces(i,:);
        C = flip(C);
        times = 1:1:trace_size(2);
        trace = [times', C'];
        
        S = BreachTraceSystem({'x'}, trace);
        S.SetTime(times);
        
        all_robs = ones(1, trace_size(2));
        if testUntil == 0
            for j = (1:trace_size(2))
                 all_robs(j) = S.CheckSpec(testSpec, j);
            end
        else
            for j = (0:trace_size(2)-1)
                 all_robs(j+1) = S.CheckSpec(testSpec, j);
            end
        end
    else
        C = traces(i,:);
        C = flip(C);
        C = C(2:trace_size(2)); 
        times = 1:1:trace_size(2)-1;
        trace = [times', C'];

        S = BreachTraceSystem({'x'}, trace);
        S.SetTime(times);
        
        all_robs = ones(1, trace_size(2)-1);
        for j = (1:trace_size(2)-1)
             all_robs(j) = S.CheckSpec(testSpec, j);
        end
    end
    robs = S.CheckSpec(testSpec);
    disp(flip(all_robs));
    disp(robs);
end
