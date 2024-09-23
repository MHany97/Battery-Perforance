e%%
load('batteryDischargeData');
%%

fts_batt = trainData(1)

fts_target = fts_batt.cycle_life


%%

figure 
plot(trainData(3).)
xlabel("test")
ylabel("ss")
title("internal resistance")

%%

fid = fopen('output.json', 'w');  % Open or create a new JSON file
if fid == -1
    error('Cannot create JSON file');
end

% Write the JSON data to the file
fprintf(fid, jsonData);
fclose(fid);

% Display success message (optional)
disp('JSON file has been created successfully.');