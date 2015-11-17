function [databaseHistogram] = GetHistogramDatabase(data,clusters)

[~,numberOfClusters] = size(clusters); 
[~,sizeOfData] = size(data);

databaseHistogram = zeros(sizeOfData,numberOfClusters);
for i=1:sizeOfData

    databaseHistogram(i,:) = ComputeHistogram(data{i},clusters);
    
end

end