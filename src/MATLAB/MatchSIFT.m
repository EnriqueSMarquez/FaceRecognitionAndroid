function [matches] = MatchSIFT(image1,image2)

[~,siftDescriptors1] = vl_sift(single(image1));
[~,siftDescriptors2] = vl_sift(single(image2));

[matches,scores] = vl_ubcmatch(siftDescriptors1,siftDescriptors2,1.8);

end