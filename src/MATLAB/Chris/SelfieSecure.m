classdef SelfieSecure < handle
    %SELFIESECURE Object for building and testing face recognition process
    %
    
    properties
        queryImage
        queryFace
        queryFeatures
        queryLabel
        queryIndex
        queryLocation
        faceDatabase
        trainingDatabase
        testingDatabase
        trainingFeatures
        trainingLabels
        trainingClass
        trainingLocation
        faceClassifier
        faceClassifierCV
        faceClassifierOosLoss
        personIndex
        featureMethod = 'hog';
        classifierMethod = 'svm';
    end
    
    methods
        function obj = SelfieSecure(facedatabase)
            %% constructor
            % setup vl_feat toolbox
            addpath('..\..\vlfeat-0.9.20\toolbox')
            vl_setup
            if nargin>0
                obj.faceDatabase = facedatabase;
            end
        end
        function train(obj,varargin)
            % train the chosen recognizer
            if length(varargin)>0
                obj.featureMethod = varargin{1};
            end
            if length(varargin)>1
                obj.classifierMethod = varargin{2};
            end
            
            %% train face recognition algorithm
            obj.trainingFeatures = struct('coords',[],'descrs',[]);
            % extract training features
            featureCount = 1;
            for i = 1:size(obj.trainingDatabase,2)
                for j = 1:obj.trainingDatabase(i).Count
                    obj.trainingFeatures(featureCount) = obj.extractFeatures(read(obj.trainingDatabase(i),j),obj.featureMethod);
                    obj.trainingLabels{featureCount} = obj.trainingDatabase(i).Description;
                    obj.trainingClass(featureCount) = i;
                    obj.trainingLocation(featureCount) = j;
                    featureCount = featureCount + 1;
                end
                obj.personIndex{i} = obj.trainingDatabase(i).Description;
            end
            % create face classifier
            obj.faceClassifier = obj.createFaceClassifier(obj.trainingFeatures,obj.trainingLabels,obj.trainingLocation,obj.personIndex,obj.classifierMethod);
            % cross validation
            %             obj.faceClassifierCV = crossval(obj.faceClassifier);
            % oosLoss
            %             obj.faceClassifierOosLoss = kfoldLoss(obj.faceClassifierCV);
        end
        function [Accuracy,C,order] = test(obj)
            % test the recognizer on the test dataset
            testCount = 1;
            for i = 1:size(obj.testingDatabase,2)
                for j = 1:obj.testingDatabase(i).Count
                    [predictedLabel(testCount),predictedIndex(testCount),predictedLocation(testCount)] = obj.predictFace(read(obj.testingDatabase(i),j));
                    testLabel{testCount} = obj.testingDatabase(i).Description;
                    testIndex(testCount) = i;
                    testLocation(testCount) = j;
                    
                    testCount = testCount + 1;
                end
            end
            % confusion matrix
            [C,order] = confusionmat(testLabel,predictedLabel,'order',obj.personIndex);
            % Accuracy
            Accuracy = sum(diag(C))/sum(sum(C));
            
            % plot up to 5 example predictions
            figure;
            if size(obj.testingDatabase,2)<5
                n = size(obj.testingDatabase,2);
            else
                n=5;
            end
            
            p = randperm(size(obj.testingDatabase,2));
            j=1;
            for i = 1:n
                subplot(n,2,j)
                imshow(read(obj.testingDatabase(testIndex(p(i))),testLocation(p(i))))
                title(['Query person #' num2str(testIndex(p(i)))])
                
                subplot(n,2,j+1)
                imshow(read(obj.trainingDatabase(predictedIndex(p(i))),predictedLocation(p(i))))
                title(['Matched person #' num2str(predictedIndex(p(i)))])
                
                j = j+2;
            end
            
        end
        function [label,index,location] = predictImage(obj,varargin)
            %% predict the person given a query image
            nVarargs = length(varargin);
            if nVarargs>0
                obj.queryImage = varargin{1};
            end
            obj.queryFace = obj.extractFace(obj.queryImage);
            [label,index,location] = obj.predictFace;
        end
        function [label,index,location] = predictFace(obj,varargin)
            %% predict the person given a query facial image
            nVarargs = length(varargin);
            if nVarargs>0
                obj.queryFace = varargin{1};
            end
            obj.queryFeatures = obj.extractFeatures(obj.queryFace,obj.featureMethod);
            [label,index,location] = obj.predictFeatures;
        end
        function [label,index,location] = predictFeatures(obj,varargin)
            %% predict the person given query features
            nVarargs = length(varargin);
            if nVarargs>0
                obj.queryFeatures = varargin{1};
            end
            switch lower(obj.classifierMethod)
                
                case 'svm'
                    obj.queryLabel = predict(obj.faceClassifier,obj.queryFeatures.descrs);
                    booleanIndex = strcmp(obj.queryLabel,obj.personIndex);
                    obj.queryIndex = find(booleanIndex);
                    obj.queryLocation = 1;
                case 'fishervectors'
                    
                    % subtract mean
                    X = bsxfun(@minus, [obj.queryFeatures.descrs]', obj.faceClassifier.mu);
                    % apply PCA
                    score = X*obj.faceClassifier.coeff;
                    data = score(:,1:obj.faceClassifier.num_pc)';
                    % append interest point spatial coords
                    img_descrs_66D = [data;[obj.queryFeatures.coords]];
                    % extract fisher vectors using vlfeat toolbox
                    trainingFeatures = vl_fisher(img_descrs_66D, obj.faceClassifier.means, obj.faceClassifier.covariances, obj.faceClassifier.priors)';
                    % match fisher vectors
                    delta = bsxfun(@minus,[obj.faceClassifier.trainingFeatures],trainingFeatures);
                    dNorm = sqrt(sum(delta.^2,2));
                    % output
                    [minNorm,idx] = min(dNorm);
                    obj.queryLabel = obj.faceClassifier.trainingLabels(idx);
                    booleanIndex = strcmp(obj.queryLabel,obj.personIndex);
                    obj.queryIndex = find(booleanIndex);
                    obj.queryLocation = obj.faceClassifier.trainingLocation(idx);
                    
                case 'vbow'% visual bag of words
                    
                    X = bsxfun(@minus, [obj.queryFeatures.descrs]', obj.faceClassifier.mu);
                    score = X*obj.faceClassifier.coeff;
                    data = score(:,1:obj.faceClassifier.num_pc)';
                    img_descrs_66D = [data;[obj.queryFeatures.coords]];
                    % extract vbow
                    % extract vbow
                    d = vl_alldist2(img_descrs_66D,obj.faceClassifier.centres);
                    [~,counts] = min(d,[],2);
                    trainingFeatures = histcounts(counts,[1:size(obj.faceClassifier.centres,2)]);
                    
                    % match vbow vectors
                    delta = bsxfun(@minus,[obj.faceClassifier.trainingFeatures],trainingFeatures);
                    dNorm = sqrt(sum(delta.^2,2));
                    % output
                    [minNorm,idx] = min(dNorm);
                    obj.queryLabel = obj.faceClassifier.trainingLabels(idx);
                    booleanIndex = strcmp(obj.queryLabel,obj.personIndex);
                    obj.queryIndex = find(booleanIndex);
                    obj.queryLocation = obj.faceClassifier.trainingLocation(idx);
                    
                otherwise
                    
            end
            label = obj.queryLabel;
            index = obj.queryIndex;
            location = obj.queryLocation;
        end
        function partitionFaceDatabase(obj,groupPercentages)
            %% partition facial image database
            % creates the training and testing datasets
            [obj.trainingDatabase,obj.testingDatabase]=partition(obj.faceDatabase,groupPercentages);
        end
    end
    methods(Static)
        function [faceImage,boundingBox] = extractFace(image)
            try
                %% extract facial image from image
                FDetect = vision.CascadeObjectDetector('ClassificationModel','FrontalFaceCART','MinSize',[80 80]);
                %                 currentImage = imresize(image,0.3);
                currentImage = image;
                %                 currentImage = rgb2gray(currentImage);
                %Read the input image
                I = currentImage;
                %                 [rowsImage,columnsImage] = size(I);
                %                                 if columnsImage > rowsImage
                %                                     % first try upright photo
                %                                     I = imrotate(I,90);
                %                                 end
                NDAngles = 6;
                Angles = [90 180 270 linspace(360/(NDAngles+1),360-360/(NDAngles+1),NDAngles)];
                NAngles = NDAngles+3;
                % try rotating the image to account for
                % portrait/landscape/tilted camera
                for j = 1:NAngles
                    %Returns Bounding Box values based on number of objects
                    BB = step(FDetect,I);
                    if isempty(BB)
                        faceImage = [];boundingBox = [];
                    else
                        sizeOfBiggerRec = 0;
                        for i = 1:size(BB,1)
                            currentRectSize = (BB(i,3)*BB(i,4));
                            if sizeOfBiggerRec < currentRectSize
                                sizeOfBiggerRec = currentRectSize;
                                currentRect = i;
                            end
                        end
                        BB = BB(currentRect,:);
                        %%EXTRACT THE FACE
                        faceImage = I(BB(2):BB(2)+BB(4),BB(1):BB(1)+BB(3),:);
                        % resize image to 112 by 92
                        faceImage = imresize(faceImage,[112 92]);
                        boundingBox = BB;
                    end
                    if isempty(faceImage)
                        % try rotating the image if no upright face is detected
                        I = imrotate(I,Angles(j));
                        disp(['failed to find face, rotating image counter-clockwise by ' num2str(Angles(j)) ' degrees and re-trying face detection...'])
                    else
                        disp('face found.')
                        break
                    end
                end
            catch ME
                disp('Error: Face extraction failed.')
                faceImage = [];boundingBox = [];
            end
        end
        function features = extractFeatures(faceImage,method)
            %% extract feature set from facial image
            if iscell(method)
                numMethods = size(method,2);
            else
                numMethods = 1;
                method = {method};
            end
            features = [];
            for k = 1:numMethods
                switch lower(method{k})
                    case 'hog'
                        HOGfeatures = extractHOGFeatures(faceImage);
                        features.coords = [];features.descrs = HOGfeatures;
                    case 'fishervectors'
                        features = extractFVFeatures(faceImage);
                    case 'vbow'
                        features = extractVBOWFeatures(faceImage);
                    otherwise
                        
                end
            end
        end
        function faceDatabase = createFaceDatabase(imgDatabase, faceFolder)
            %% create a face database from an image database
            if ~exist(faceFolder,'dir')
                % create faceFolder directory if it does not exist
                [success,msg,msgId] = mkdir(faceFolder);
                if ~success
                    error(msgId,msg)
                end
            end
            numPeople = size(imgDatabase,2);
            for person = 1:numPeople
                for j = 1:imgDatabase(person).Count
                    faceImage = SelfieSecure.extractFace(read(imgDatabase(person),j));
                    % get image filename
                    [~,name,ext] = fileparts(imgDatabase(person).ImageLocation{j});
                    faceFile = [name ext];
                    if isempty(faceImage)
                        disp(['Failed to extract facial image from image ' faceFile ' in collection ' imgDatabase(person).Description])
                    else
                        [hPix,wPix]=size(faceImage);
                        disp(['Facial image size from image ' faceFile ' in collection ' imgDatabase(person).Description ': ' num2str(wPix) ' by ' num2str(hPix)])
                        facePath = fullfile(faceFolder,imgDatabase(person).Description);
                        if ~exist(facePath,'dir')
                            % create
                            [success,msg,msgId] = mkdir(facePath);
                            if ~success
                                error(msgId,msg)
                            end
                        end
                        faceFullFile = fullfile(facePath,faceFile);
                        % write image file
                        imwrite(faceImage,faceFullFile);
                    end
                end
            end
            faceDatabase = imageSet(faceFolder,'recursive');
        end
        function faceClassifier = createFaceClassifier(trainingFeatures,trainingLabels,trainingLocation,classNames,method)
            %% create a face classifier from a set of labelled training features
            switch lower(method)
                case 'svm'
                    
                    t = templateSVM('Standardize',true);
                    faceClassifier = fitcecoc(cell2mat({trainingFeatures.descrs}'),trainingLabels,'Learners',t,'ClassNames',classNames,'Coding','onevsall');
                    
                case 'fishervectors'
                    
                    % PCA
                    X = [trainingFeatures.descrs]';
                    coords = [trainingFeatures.coords];
                    
                    % limit size of descriptors to 0.5 million
                    n = size(X,1);
                    if n>500000
                        p = randperm(n);
                        X = X(p(1:500000),:);
                        coords = coords(:,p(1:500000));
                    end
                    
                    faceClassifier.mu = mean(X,1);
                    X = bsxfun(@minus, X, faceClassifier.mu);
                    % PCA
                    [faceClassifier.coeff,score,latent,tsquared,explained] = pca(X);
                    if size(faceClassifier.coeff,1)<64
                        faceClassifier.num_pc = size(faceClassifier.coeff,1);
                    else
                        faceClassifier.num_pc = 64;
                    end
                    % reduced data consists of the leading PCs
                    descrs_PCA_RootSIFT = score(:,1:faceClassifier.num_pc)';
                    
                    % append spatial data
                    descrs_66D = [descrs_PCA_RootSIFT;coords];
                    
                    % train GMM
                    numClusters = 256;
                    
                    [faceClassifier.means, faceClassifier.covariances, faceClassifier.priors] = vl_gmm(descrs_66D, numClusters);
                    
                    for i = 1:length(trainingFeatures)
                        X = bsxfun(@minus, [trainingFeatures(i).descrs]', faceClassifier.mu);
                        score = X*faceClassifier.coeff;
                        data = score(:,1:faceClassifier.num_pc)';
                        img_descrs_66D = [data;[trainingFeatures(i).coords]];
                        % extract fisher vectors
                        faceClassifier.trainingFeatures(i,:) = vl_fisher(img_descrs_66D, faceClassifier.means, faceClassifier.covariances, faceClassifier.priors)';
                        
                        faceClassifier.trainingLabels(i) = trainingLabels(i);
                        faceClassifier.trainingLocation(i) = trainingLocation(i);
                    end
                    
                case 'vbow'
                    
                    %
                    X = [trainingFeatures.descrs]';
                    coords = [trainingFeatures.coords];
                    
                    % limit size of descriptors to 0.5 million
                    n = size(X,1);
                    if n>500000
                        p = randperm(n);
                        X = X(p(1:500000),:);
                        coords = coords(:,p(1:500000));
                    end
                    
                    faceClassifier.mu = mean(X,1);
                    X = bsxfun(@minus, X, faceClassifier.mu);
                    [faceClassifier.coeff,score,latent,tsquared,explained] = pca(X);
                    if size(faceClassifier.coeff,1)<64
                        faceClassifier.num_pc = size(faceClassifier.coeff,1);
                    else
                        faceClassifier.num_pc = 64;
                    end
                    % reduced data consists of the leading PCs
                    descrs_PCA_RootSIFT = score(:,1:faceClassifier.num_pc)';
                    
                    % append spatial data
                    descrs_66D = [descrs_PCA_RootSIFT;coords];
                    
                    % train GMM
                    numClusters = 256;
                    
                    [faceClassifier.centres,faceClassifier.assignment] = vl_kmeans(descrs_66D, numClusters);
                    
                    for i = 1:length(trainingFeatures)
                        X = bsxfun(@minus, [trainingFeatures(i).descrs]', faceClassifier.mu);
                        score = X*faceClassifier.coeff;
                        data = score(:,1:faceClassifier.num_pc)';
                        img_descrs_66D = [data;[trainingFeatures(i).coords]];
                        % extract vbow
                        d = vl_alldist2(img_descrs_66D,faceClassifier.centres);
                        [~,counts] = min(d,[],2);
                        faceClassifier.trainingFeatures(i,:) = histcounts(counts,[1:size(faceClassifier.centres,2)]);
                        
                        faceClassifier.trainingLabels(i) = trainingLabels(i);
                        faceClassifier.trainingLocation(i) = trainingLocation(i);
                    end
                    
                otherwise
            end
        end
    end
end

