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
jsonStr = jsonencode(fts_batt);
%%
fileID = fopen('firstbatt.json', 'w');
fprintf(fileID, '%s', jsonStr);
fclose(fileID);