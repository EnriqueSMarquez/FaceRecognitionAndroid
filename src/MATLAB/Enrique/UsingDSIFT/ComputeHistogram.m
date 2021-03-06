function [histogram] = ComputeHistogram(image,clusters)

    binSize = 8 ;
    magnif = 3 ;
    faceDescriptor = vl_imsmooth(single(image), sqrt((binSize/magnif)^2 - .25)) ;
    [~,faceDescriptor] = vl_dsift(faceDescriptor,'size', binSize);
    
    [~,numberOfClusters] = size(clusters); 
    [~,numberOfDescriptors] = size(faceDescriptor);
    histogram = zeros(1,numberOfClusters);
    for i=1:numberOfDescriptors
        currentDescriptor = faceDescriptor(:,i);
        toSubstract = kron(currentDescriptor,uint8(ones(1,numberOfClusters)));
        substraction = sqrt(sum(int8(toSubstract) - int8(clusters)).^2);
        [~,minIndex] = min(substraction);
        histogram(minIndex) = histogram(minIndex) + 1;
        display(int2str(numberOfDescriptors-i))
    end
end