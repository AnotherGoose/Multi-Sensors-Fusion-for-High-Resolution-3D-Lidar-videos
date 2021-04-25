 function AveragePC = boxfilterPC(I,Ind)

%% locating neighbours using Indices matrix (nxr^2) and averaging
 
AverageX = I(:,1);
AverageX = median(AverageX(Ind),2);

AverageY = I(:,2);
AverageY = median(AverageY(Ind),2);

AverageZ = I(:,3);
AverageZ = median(AverageZ(Ind),2);
 

%% Final output
AveragePC = [AverageX, AverageY, AverageZ]; %final result 
