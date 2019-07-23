%% K-NN Classifer 
% This function classify each image from the testset using euclidean
% between attribute points as the measuring metric. 

function PredictedLabels = KNN_Classifer(testset, trainset, label, K)
    m.x = trainset;
    m.c = label;
    
    x = testset;
    
    for i = 1:size(x,1)
        PredictedLabels(i,1) = Classifer(x(i,:), m, K);
    end
    PredictedLabels = categorical(PredictedLabels);
end

function PredictedLabel = Classifer(X, S, K)
    A = [];
    A = string(A);
    tempCoord = 0;
    tempMax = 0;
    
    for i = 1:size(S.x, 1)
        d = 0;
        for j = 1:size(S.x, 2)

        %Calculate the euclidean distance between sample and input x   
        temp = (S.x(i, j) - X(j))^2;
        d = temp + d;
        end
        
        d = sqrt(d);
        d = string(d);
        if(size(A,1) < K)
            A(size(A,1)+ 1, 2) = S.c(i);
            A(size(A,1), 1) = d;
        elseif (d < tempMax)
            A(tempCoord,1) = d;
            A(tempCoord,2) = S.c(i);
        end
        
        [tempCoord, tempMax] = maxCoord(A);
        
    end
    A = sortDescend(A);
    if(size(A,1) == 1)
        PredictedLabel = A(1,2);
    else
        V = mode(A); 
        PredictedLabel = V(1,2);
    end
end

% This function arrange the elements of the input vector into a descending
% order
function A = sortDescend(A)
    
    for j = 1:size(A,1)-1
        k = size(A,1)-j;
        
        for i = 1:k      
            if (A(i,1) < A(i+1,1))
                
                tempFeature = A(i,1);
                tempLabel = A(i, 2);
                
                A(i,1) = A(i+1,1);
                A(i,2) = A(i+1,2);
                
                A(i+1, 1) = tempFeature;
                A(i+1, 2) = tempLabel;
            end
        end
    end
end

function [x, y] = maxCoord(A)
    
    tempMax = A(1,1);
    coord = 1;
    for j = 1:size(A,1)
        if(tempMax < A(j,1))
            tempMax = A(j,1);
            coord = j;
        end
    end
    x = coord;
    y = tempMax;
end