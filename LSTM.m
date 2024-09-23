%Loading Raw DAta  
load('batteryDischargeData');
%%

%%
%Prepar Data 
%for Linear Regression
[XTrain,yTrain] = helperGetFeatures(testData);
[XTest,yTest] = helperGetFeatures(testData);
[XVal,yVal] = helperGetFeatures(valData);
%For LSTM
%%
[features_train,target_train,mu_train,sigma_train,featuresNorm_train,numObservations_train,featuresSeq_train,targetSeq_train,numOfBatt_train] = GetLSTMFeatures(XTrain,yTrain)
[features_val,target_val,mu_val,sigma_val,featuresNorm_val,numObservations_val,featuresSeq_val,targetSeq_val,numOfBatt_val] = GetLSTMFeatures(XVal,yVal)
[features_test,target_test,mu_test,sigma_test,featuresNorm_test,numObservations_test,featuresSeq_test,targetSeq_test,numOfBatt_test] = GetLSTMFeatures(XTest,yTest)

%%
%Data Visual8ization 
figure, hold on;
for i = 1:size(trainData,2)
    if numel(trainData(i).summary.cycle) == numel(unique(trainData(i).summary.cycle))
      plot(trainData(i).summary.cycle, trainData(i).summary.QDischarge);      
    end
end 
ylim([0.85,1.1]),xlim([0,1000]);
ylabel('Discharge capacity (Ah)');
xlabel('cycle');

figure, hold on;
for i = 1:size(testData,2)
    plot((testData(i).cycles(100).Qdlin - testData(i).cycles(10).Qdlin), testData(i).Vdlin)   
end
ylabel('Voltage (V)'); xlabel('Q_{100} - Q_{10} (Ah)');

figure;
trainDataSize = size(trainData,2);
cycleLife = zeros(trainDataSize,1);
DeltaQ_var = zeros(trainDataSize,1);
for i = 1:trainDataSize
    cycleLife(i) = size(trainData(i).cycles,2) + 1;
    DeltaQ_var(i) = var(trainData(i).cycles(100).Qdlin - trainData(i).cycles(10).Qdlin);
end
loglog(DeltaQ_var,cycleLife, 'o')
ylabel('Cycle life'); xlabel('Var(Q_{100} - Q_{10}) (Ah^2)');


%%
%%Linear Regression 



rng("default")

alphaVec = 0.01:0.1:1;
lambdaVec = 0:0.01:1;
MCReps = 1;
cvFold = 4;

rmseList = zeros(length(alphaVec),1);
minLambdaMSE = zeros(length(alphaVec),1);
wModelList = cell(1,length(alphaVec));
betaVec = cell(1,length(alphaVec));
for i=1:length(alphaVec)
    [coefficients,fitInfo] = ...
        lasso(XTrain.Variables,yTrain,'Alpha',alphaVec(i),'CV',cvFold,'MCReps',MCReps,'Lambda',lambdaVec);
    wModelList{i} = coefficients(:,fitInfo.IndexMinMSE);
    betaVec{i} = fitInfo.Intercept(fitInfo.IndexMinMSE);       
    indexMinMSE = fitInfo.IndexMinMSE;
    rmseList(i) = sqrt(fitInfo.MSE(indexMinMSE));    
    minLambdaMSE(i) = fitInfo.LambdaMinMSE;    
end

numVal = 4;
[out,idx] = sort(rmseList);
val = out(1:numVal);
index = idx(1:numVal);

alpha = alphaVec(index);
lambda = minLambdaMSE(index);
wModel = wModelList(index);
beta = betaVec(index);


rmseValModel = zeros(numVal);
for valList = 1:numVal   
   yPredVal = (XVal.Variables*wModel{valList} + beta{valList});
   rmseValModel(valList) = sqrt(mean((yPredVal-yVal).^2));
end

[rmseMinVal,idx] = min(rmseValModel);
wModelFinal = wModel{idx};
betaFinal = beta{idx};


yPredTest = (XTest.Variables*wModelFinal + betaFinal);

figure;
scatter(yTest,yPredTest)
hold on;
refline(1, 0);
title('Predicted vs Actual Cycle Life')
ylabel('Predicted cycle life');
xlabel('Actual cycle life');

errTest = (yPredTest-yTest);
rmseTestModel = sqrt(mean(errTest.^2))

n = numel(yTest);
nr = abs(yTest - yPredTest);
errVal = (1/n)*sum(nr./yTest)*100


%%
%LSTM 
%%Defining LSTM Network ARCH

inputSize = size(features_train, 2);
numHiddenUnits = 1000;
numResponses = 1;
%%
layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits, 'OutputMode', 'last')
    fullyConnectedLayer(numResponses)
    regressionLayer];

% Specify training options
options = trainingOptions('adam', ...
    'MaxEpochs', 1000, ...
    'GradientThreshold', 1, ...
    'InitialLearnRate', 0.5, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 50, ...
    'LearnRateDropFactor', 0.2, ...
    'Verbose', 0, ...
    'ValidationData', {featuresSeq_val, target_val},...
    'Plots', 'training-progress');
%%
% Train the LSTM network
net = trainNetwork(featuresSeq_train, target_train, layers, options);
predictedRemainingCycleLife = predict(net, featuresSeq_val);
E = rmse(predictedRemainingCycleLife,target_val)


%predictedRemainingCycleLife = cell2mat(predictedRemainingCycleLife);

% Evaluate the model
%rmse_LSTM = sqrt(mean((predictedRemainingCycleLife - TargetTest).^2));
%fprintf('Root Mean Square Error: %.4f\n', rmse_LSTM);


%%
data = load('batteryDischargeData');
jsonStr = jsonencode(data)
fid = fopen('output.json','w');
if fid == -1 
       error('cannot create');
end
fwrite(fid, jsonStr , 'char');
fclose(fid);
%%
function [features,target,mu,sigma,featuresNorm,numObservations,featuresSeq,targetSeq,numOfBatt] = GetLSTMFeatures(X,y)

    features = [X.DeltaQ_var, X.DeltaQ_min, X.CapFadeCycle2Slope, X.CapFadeCycle2Intercept, X.Qd2, X.AvgChargeTime, X.MinIR, X.IRDiff2And100]
    target = [y]
    
    %Normalize Feature
    
    mu = mean(features);
    sigma = std(features);
    featuresNorm = (features - mu) ./ sigma;
    
    % Convert features and target to cell arrays
    numObservations = numel(target);
    featuresSeq = cell(numObservations, 1);
    targetSeq = cell(numObservations, 1);
    
    for numOfBatt = 1:numObservations
        featuresSeq{numOfBatt} = featuresNorm(numOfBatt, :)';
        targetSeq{numOfBatt} = target(numOfBatt);
    end
end

%%
function [xTable, y] = helperGetFeatures(batch)
% HELPERGETFEATURES function accepts the raw measurement data and computes
% the following feature:
%
% Q_{100-10}(V) variance [DeltaQ_var]
% Q_{100-10}(V) minimum [DeltaQ_min]
% Slope of linear fit to the capacity fade curve cycles 2 to 100 [CapFadeCycle2Slope]
% Intercept of linear fit to capacity fade curve, cycles 2 to 100 [CapFadeCycle2Intercept]
% Discharge capacity at cycle 2 [Qd2]
% Average charge time over first 5 cycles [AvgChargeTime]
% Minimum Internal Resistance, cycles 2 to 100 [MinIR]
% Internal Resistance difference between cycles 2 and 100 [IRDiff2And100]

N = size(batch,2); % Number of batteries

% Preallocation of variables
y = zeros(N, 1);
DeltaQ_var = zeros(N,1);
DeltaQ_min = zeros(N,1);
CapFadeCycle2Slope = zeros(N,1);
CapFadeCycle2Intercept = zeros(N,1);
Qd2 = zeros(N,1);
AvgChargeTime = zeros(N,1);
IntegralTemp = zeros(N,1);
MinIR = zeros(N,1);
IRDiff2And100 = zeros(N,1);

for i = 1:N
    % cycle life
    y(i,1) = size(batch(i).cycles,2) + 1;
    
    % Identify cycles with time gaps
    numCycles = size(batch(i).cycles, 2);
    timeGapCycleIdx = [];
    jBadCycle = 0;
    for jCycle = 2: numCycles
        dt = diff(batch(i).cycles(jCycle).t);
        if max(dt) > 5*mean(dt)
            jBadCycle = jBadCycle + 1;
            timeGapCycleIdx(jBadCycle) = jCycle;
        end
    end
    
    % Remove cycles with time gaps
    batch(i).cycles(timeGapCycleIdx) = [];
    batch(i).summary.QDischarge(timeGapCycleIdx) = [];
    batch(i).summary.IR(timeGapCycleIdx) = [];
    batch(i).summary.chargetime(timeGapCycleIdx) = [];
    
    % compute Q_100_10 stats
    DeltaQ = batch(i).cycles(100).Qdlin - batch(i).cycles(10).Qdlin;
    DeltaQ_var(i) = log10(abs(var(DeltaQ)));
    DeltaQ_min(i) = log10(abs(min(DeltaQ)));
    
    % Slope and intercept of linear fit for capacity fade curve from cycle
    % 2 to cycle 100
    coeff2 = polyfit(batch(i).summary.cycle(2:100), batch(i).summary.QDischarge(2:100),1);
    CapFadeCycle2Slope(i) = coeff2(1);
    CapFadeCycle2Intercept(i) = coeff2(2);
    
    %  Discharge capacity at cycle 2
    Qd2(i) = batch(i).summary.QDischarge(2);
    
    % Avg charge time, first 5 cycles (2 to 6)
    AvgChargeTime(i) = mean(batch(i).summary.chargetime(2:6));
    
    % Integral of temperature from cycles 2 to 100
    tempIntT = 0;
    for jCycle = 2:100
        tempIntT = tempIntT + trapz(batch(i).cycles(jCycle).t, batch(i).cycles(jCycle).T);
    end
    IntegralTemp(i) = tempIntT;
    
    % Minimum internal resistance, cycles 2 to 100
    temp = batch(i).summary.IR(2:100);
    MinIR(i) = min(temp(temp~=0));
    IRDiff2And100(i) = batch(i).summary.IR(100) - batch(i).summary.IR(2);
end

xTable = table(DeltaQ_var, DeltaQ_min, ...
    CapFadeCycle2Slope, CapFadeCycle2Intercept,...
    Qd2, AvgChargeTime, MinIR, IRDiff2And100);

end