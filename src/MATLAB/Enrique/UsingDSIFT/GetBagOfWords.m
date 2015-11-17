function [clusters] = GetBagOfWords(data)

[~,numberOfPeople] = size(data);
binSize = 8 ;
magnif = 3 ;
for i=1:numberOfPeople
    faceDescriptor = vl_imsmooth(single(data{i}), sqrt((binSize/magnif)^2 - .25)) ;
    [~,faceDescriptor] = vl_dsift(faceDescriptor,'size', binSize);
    descriptorsDatabase{i} = faceDescriptor';
end

%CONCATENATE ALL THE DESCRIPTORS

for j=1:numberOfPeople
    
    if j==1
    fullDescriptors = descriptorsDatabase{1};
    else
    fullDescriptors = [fullDescriptors ; descriptorsDatabase{j}];
    end
    
end

[clusters,~] = kmeans(single(fullDescriptors'),250);

end