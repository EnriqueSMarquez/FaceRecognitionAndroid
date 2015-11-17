cd SelfiesDataSet\
cd person6
image1 = imread('1.jpg');
image2 = imread('2.jpg');
cd ..
cd person3
image3 = imread('2.jpg');
faceImage1 = ExtractFaceFromImage(image1);
faceImage2 = ExtractFaceFromImage(image2);
faceImage3 = ExtractFaceFromImage(image3);

[siftFeatures1,siftDescriptors1] = vl_sift(single(faceImage1));
[siftFeatures2,siftDescriptors2] = vl_sift(single(faceImage2));
[siftFeatures3,siftDescriptors3] = vl_sift(single(faceImage3));

[matches12,scores12] = vl_ubcmatch(siftDescriptors1,siftDescriptors2,2);
imshow(faceImage1)
h1 = vl_plotframe(siftFeatures1(:,matches12(1,:))) ;
set(h1,'color','y','linewidth',3) ;
figure
imshow(faceImage2)
h2 = vl_plotframe(siftFeatures2(:,matches12(2,:)));
set(h2,'color','y','linewidth',2) ;
cd ..
cd ..
pause

