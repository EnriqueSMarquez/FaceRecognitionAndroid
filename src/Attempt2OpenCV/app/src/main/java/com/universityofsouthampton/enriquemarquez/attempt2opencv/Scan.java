package com.universityofsouthampton.enriquemarquez.attempt2opencv;


import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Environment;
import android.preference.PreferenceManager;
import android.util.Log;
import android.view.Gravity;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.Utils;
import org.opencv.contrib.FaceRecognizer;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.TermCriteria;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.CvKNearest;
import org.opencv.ml.EM;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Vector;

public class Scan extends Activity  {

    private Button back, rescan;
    private TextView nickname;
    private String selectedImagePath;
    private ImageView img;
    private Bitmap faceToRecognise;
    private HashMap<Bitmap, String> facesMap = new HashMap<Bitmap, String>();
    private Map<String, ?> allEntries;
    final static int ORB_THRESHOLD = 200;
    final static int BRISK_THRESHOLD = 600;
    private final static int NUM_CLUSTERS = 1000;
    Mat centres = new Mat();
    Object [] faces;
    List<int[]> histograms = new ArrayList<>();

    FeatureDetector featureDetector;
    DescriptorExtractor descriptorExtractor;
    DescriptorMatcher descriptorMatcher;
    MatOfKeyPoint matOfKeyPointFaceToRecognize = new MatOfKeyPoint();
    Mat descriptorsFaceToRecognize = new Mat();
    Mat matFaceToRecognize;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_scan);

        nickname = (TextView) findViewById(R.id.nickname);
        back = (Button) findViewById(R.id.button_back);
        rescan = (Button) findViewById(R.id.button_rescan);
        img = (ImageView) findViewById(R.id.imageview);

        //Go back to the main screen
        back.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                onBackPressed();
            }
        });

        //Allow them to take a picture again
        rescan.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(Scan.this, TakePicture.class);
                intent.putExtra("New Person", false);
                intent.putExtra("Fisher",getIntent().getBooleanExtra("Fisher", true));
                startActivity(intent);

            }
        });
        //GET FACES FROM DATABASE
        faces = getFacesArrayFromDatabase();
        //INITIALIZE TYPE OF FEATURE DETECTOR
        featureDetector = FeatureDetector.create(FeatureDetector.DENSE);
        //INITIALIZE TYPE OF FEATURE EXTRACTOR
        descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
        //INITIALIZE TYPE OF MATCHES
        descriptorMatcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE);

        //IF FISHER IS GOING TO BE PREFORMED, CALCULATE THE NECESSARY PARAMETERS
        if (getIntent().getBooleanExtra("Fisher", true) == true) {
            //GET ALL THE MATS FROM EACH IMAGE IN THE DATABASE
            List<Mat> facesMat = fromBitmapToMat();
            List<Mat> databaseDescriptors = new ArrayList<>();
            //GET ALL THE DESCRIPOTRS FROM THE IMAGES IN THE DATABASE
            databaseDescriptors.addAll(generateDescriptors(facesMat));
            Mat fullDescriptors = new Mat();
            //CONCATENATE VERTICALLY ALL THE DESCRIPTORS IN A SINGLE MATRIX
            Core.vconcat(databaseDescriptors, fullDescriptors);
            //CONVERTO TO 32 FLOAT TO PERFORM GMM OR KMEANS
            fullDescriptors.convertTo(fullDescriptors, CvType.CV_32F);
            Mat labels = new Mat();
            //ITERATION ALGORITHM PARAMETERS
            TermCriteria criteria = new TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 100, 0.1);
            //COMPUTE CENTROIDS
            Core.kmeans(fullDescriptors, NUM_CLUSTERS, labels, criteria, 1, Core.KMEANS_PP_CENTERS, centres);

            //GO THORUGH ALL THE FACES
            for (int k = 0; k < databaseDescriptors.size(); k++) {
                MatOfDMatch matchesData = new MatOfDMatch();
                List<DMatch> matchesLData;
                Mat currentDescriptor = databaseDescriptors.get(k);
                currentDescriptor.convertTo(currentDescriptor, CvType.CV_32FC1);
                //MATCH ALL THE DESCRIPTORS WITH THE CENTRES
                descriptorMatcher.match(currentDescriptor, centres, matchesData);
                matchesLData = matchesData.toList();
                int[] matchesIndexesTmp = new int[matchesLData.size()];
                for (int j = 0; j < matchesLData.size(); j++) {
                    //COMPUTE VECTOR OF MATCHES
                    DMatch currentMatch = matchesLData.get(j);
                    matchesIndexesTmp[j] = currentMatch.trainIdx;
                }
                //GENERATE HISTOGRAMS OF ALL THE IMAGES
                histograms.add(generateHistogram(matchesIndexesTmp, NUM_CLUSTERS));
            }
        }

        initialise();
        checkFaceIsCorrect();
    }

    //show the image taken
    private void initialise() {
        selectedImagePath = getIntent().getStringExtra("path");
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inScaled = false;
        faceToRecognise = BitmapFactory.decodeFile(selectedImagePath, options);
        img.setImageBitmap(faceToRecognise);
    }

    private void checkFaceIsCorrect() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Face Detection Confirmation");
        builder.setMessage("Has your face been detected correctly?");
        builder.setPositiveButton("Yes", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                //User clicked yes so close the dialog
                dialog.cancel();
                recognise();
            }
        });
        builder.setNegativeButton("No", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                //User clicked no so return to main screen
                tellUser("Please try again! We recommend keeping your face straight and removing your glasses.");
                //end the activity
                onBackPressed();
            }
        });
        AlertDialog dialog = builder.create();
        dialog.setCanceledOnTouchOutside(false);
        //Put the dialog at the top of the screen
        Window window = dialog.getWindow();
        window.clearFlags(WindowManager.LayoutParams.FLAG_DIM_BEHIND);
        WindowManager.LayoutParams wlp = window.getAttributes();
        wlp.gravity = Gravity.TOP;
        window.setAttributes(wlp);
        //Show the dialog
        dialog.show();
    }


    private void tellUser(String message) {
        Toast toast = Toast.makeText(this, message, Toast.LENGTH_LONG);
        toast.show();
    }

    //Recognise the face from the database and then display the appropriate nickname
    private void recognise() {
        //Object[] faces = getFacesArrayFromPreferences();

        Bitmap recognisedFace;
        //Do facial recognition
        if (getIntent().getBooleanExtra("Fisher", true) == true) {
            recognisedFace  = recogniseUsingBOV();
        } else {
            recognisedFace = facialRecognitionUsingORBMatching();
        }
        //If the facial recognition returned an image
        if (recognisedFace != null) {
           //DisplayNicknameFromPreferences(recognisedFace);
            DisplayNicknameFromDatabase(recognisedFace);
        }
        //If the facial recognition didn't return a face because nothing matched
        else {
           //Display that the face is unrecognised
           nickname.setText("Face not recognised");
        }

    }

    //Generate an array of bitmaps containing the faces from shared preferences
    private Object[] getFacesArrayFromPreferences() {
        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(this);
        allEntries = prefs.getAll();

        //get the actual bitmap images to work with
        for (Map.Entry<String, ?> entry : allEntries.entrySet()) {
            String imagePath = entry.getValue().toString();
            Bitmap bmp = BitmapFactory.decodeFile(imagePath);
            facesMap.put(bmp, imagePath);
        }
        Object[] keys = facesMap.keySet().toArray();
        return keys;
    }

    //Match the bitmap found with the nickname stored in the shared preferences
    private void DisplayNicknameFromPreferences(Bitmap recognisedFace) {
        //Get the path from the recognised face
        String imagePath = facesMap.get(recognisedFace);

        //Use the first key as the nickname wanted
        for (Map.Entry<String, ?> entry : allEntries.entrySet()) {
            //If the image path is equal to the image path in the shared preferences
            if (imagePath.equals(entry.getValue().toString())) {
                //Display the nickname
                nickname.setText(nickname.getText().toString() + " " + entry.getKey());
                break;
            }
        }
    }

    //Generate an array of bitmaps containing the faces from Database
    private Object[] getFacesArrayFromDatabase() {
        DatabaseOperations dop = new DatabaseOperations(this);
        Cursor CR = dop.getInformation(dop);
        CR.moveToFirst();
        //Add all the image paths from the database to the array
        do {
            //Add the values from the second column, which is where the paths are stored
            String imagePath = CR.getString(1);
            Bitmap bmp = BitmapFactory.decodeFile(imagePath);
            facesMap.put(bmp, imagePath);
        } while (CR.moveToNext());
        Object[] keys = facesMap.keySet().toArray();
        return keys;
    }

    //Match the bitmap found with the nickname stored in the database
    private void DisplayNicknameFromDatabase(Bitmap recognisedFace) {
        //Get the path from the recognised face
        String imagePath = facesMap.get(recognisedFace);
        DatabaseOperations dop = new DatabaseOperations(this);
        Cursor CR = dop.getNickname(dop, imagePath);
        CR.moveToFirst();
        String name = CR.getString(0);
        //Display the nickname
        nickname.setText(nickname.getText().toString() + " " + name);
    }


    //Actual facial recognition done here
    private Bitmap facialRecognitionUsingORBMatching() {

        featureDetector = FeatureDetector.create(FeatureDetector.ORB);
        matFaceToRecognize = new Mat(faceToRecognise.getHeight(),faceToRecognise.getWidth(), CvType.CV_8UC3);
        //FROM BITMAP TO MAT TEST FACE
        Utils.bitmapToMat(faceToRecognise, matFaceToRecognize);
        //FROM RGB TO GRAY
        Imgproc.cvtColor(matFaceToRecognize, matFaceToRecognize, Imgproc.COLOR_BGR2GRAY);
        //DETEC FEATURES TEST FACE
        featureDetector.detect(matFaceToRecognize, matOfKeyPointFaceToRecognize);
        //EXTRACT DESCRIPTORS TEST FACE
        descriptorExtractor.compute(matFaceToRecognize,matOfKeyPointFaceToRecognize,descriptorsFaceToRecognize);

        //INITIALIZE NUMBER OF MATCHES
        int mostMatches = 0;
        //LIST OF MATCHES
        List<DMatch> goodMatches;
        int currentFaceIndex = 0;
        //LOOP THORUGH ALL THE FACES
        for(int i=0;i < faces.length;i++){
            List<Float> bestMatches = new ArrayList<>();
            //FROM OBJECT TO BITMAP CURRENT FACE
            Bitmap currentBmp = (Bitmap) faces[i];
            //CREATE CURRENT FACE MAT
            Mat currentFace = new Mat(currentBmp.getHeight(),currentBmp.getWidth(),CvType.CV_8UC1);
            MatOfKeyPoint currentFaceKeyPoints = new MatOfKeyPoint();
            Mat descriptorsCurrentFace = new Mat();
            MatOfDMatch matches = new MatOfDMatch();
            //CURRENT FACE FROM BITMAP TO MAT
            Utils.bitmapToMat((Bitmap) faces[i], currentFace);
            //DETECT FEATURES OF THE CURRENT FACE
            featureDetector.detect(currentFace, currentFaceKeyPoints);
            //COMPUTE THE DESCRIPTORS OF THE CURRENT FACE
            descriptorExtractor.compute(currentFace, currentFaceKeyPoints, descriptorsCurrentFace);
            //MATCH TEST FACE WITH CURRENT FACE
            descriptorMatcher.match(descriptorsFaceToRecognize, descriptorsCurrentFace, matches);
            //MATCHES TO LIST
            goodMatches = matches.toList();

            //FILTER MATCHES (IF THE DISTANCE IS TOO BIG, DISCARD CURRENT MATCH)
            for( int j = 0; j < goodMatches.size();j ++){
                if(goodMatches.get(j).distance < ORB_THRESHOLD)
                    bestMatches.add((goodMatches.get(j).distance));
            }
            //IF CURRENT FACE HAS MORE MATCHES, SAVE INDEX
            if(mostMatches < bestMatches.size()) {
                mostMatches = bestMatches.size();
                currentFaceIndex = i;
            }

        }
        //INITIALISE MATCHED FACE
        Bitmap faceToSend = null;
        //IF THERE WAS AT LEAST ONE MATCH, SAVE MATCHED FACE
        if(mostMatches != 0)
            faceToSend = (Bitmap) faces[currentFaceIndex];
        return faceToSend;
    }
    //THIS METHOD RECOGNIZES USING FISHER
    private Bitmap recogniseUsingBOV(){
        //VARIABLE WITH MATCHES INFORMATION
        MatOfDMatch matches = new MatOfDMatch();
        //ARRAY OF MATCHES
        List<DMatch> matchesL;
        //MAT OF FACE TO RECOGNIZE
        matFaceToRecognize = new Mat(faceToRecognise.getHeight(),faceToRecognise.getWidth(), CvType.CV_8UC1);
        //FROM BITMAP TO MAT
        Utils.bitmapToMat(faceToRecognise, matFaceToRecognize);
        //FROM RGB TO GRAY
        Imgproc.cvtColor(matFaceToRecognize, matFaceToRecognize, Imgproc.COLOR_BGR2GRAY);
        //DETECT FEATURES IN FACE TO RECOGNIZE
        featureDetector.detect(matFaceToRecognize, matOfKeyPointFaceToRecognize);
        //COMPUTE DESCRIPTORS IN THE FEATURES PREVIOUSLY DETECTED
        descriptorExtractor.compute(matFaceToRecognize, matOfKeyPointFaceToRecognize, descriptorsFaceToRecognize);
        //HISTOGRAM VARIABLE
        int [] histogramFaceTestFace;
        //CONVERT FROM UNSIGNED INT TO 32 FLOAT
        descriptorsFaceToRecognize.convertTo(descriptorsFaceToRecognize, CvType.CV_32FC1);
        //CALCULATE THE DISTANCES OF THE DESCRIPTORS TO THE CENTRES BETWEEN CENTRES AND THE DESCRIPTORS OF CURRENT FACE
        descriptorMatcher.match(descriptorsFaceToRecognize, centres, matches);
        //TRANSFORM MATCHES INTO A LIST
        matchesL = matches.toList();
        int[] matchesIndexes = new int[matchesL.size()];
        for(int k = 0; k < matchesL.size(); k ++){
            //SAVE ALL THE DISTANCES OF THE DESCRIPTORS TO THE CENTRES AS A SINGLE VECTOR
            DMatch currentMatch = matchesL.get(k);
            matchesIndexes[k] = currentMatch.trainIdx;
        }
        //GENERATE THE HISTOGRAM BASED ON THE DISTANCES TO THE CENTRES
        histogramFaceTestFace = generateHistogram(matchesIndexes,NUM_CLUSTERS);
        //MATCH THE TEST IMAGE WITH THE HISTOGRAMS OF ALL THE IMAGES IN THE DATABASE
        int testImageIndex = matchTestWithTraining(histograms,histogramFaceTestFace);
        //IF THERE IS NO GOOD MATCH, RETURN NULL
        if (testImageIndex == -1) return null;
        //OTHERWISE RETURN THE MATCHED FACE
        else return (Bitmap) faces[testImageIndex];
    }


    @Override
    public void onPause() {
        super.onPause();
    }

    @Override
    public void onResume()
    {
        super.onResume();
    }

    public void onDestroy() {
        //Delete the picture as its no longer needed
        File file = new File(selectedImagePath);
        file.delete();
        super.onDestroy();
    }

    public void onBackPressed() {
        Intent intent = new Intent(Scan.this, MainActivity.class);
        startActivity(intent);
        finish();
    }

    //RETURNS THE DESCRIPTORS GIVEN THE MAT FACES ARRAY
    private List<Mat> generateDescriptors(List<Mat> facesMat){
        List<Mat> descriptorsDatabase = new ArrayList<>();
        //LOOP THROUGH ALL THE FACES
        for(int i=0;i < facesMat.size();i++) {
            //MAT OF INTEREST POINTS
            MatOfKeyPoint currentFaceKeyPoints = new MatOfKeyPoint();
            Mat descriptorsCurrentFace = new Mat();
            //EXTRACT FEATURES FORM CURRENT FACE
            featureDetector.detect(facesMat.get(i), currentFaceKeyPoints);
            //COMPUTE DESCRIPTORS OF CURRENT FACE
            descriptorExtractor.compute(facesMat.get(i), currentFaceKeyPoints, descriptorsCurrentFace);
            //ADD CURRENT DESCRIPTORS TO ARRAY
            descriptorsDatabase.add(descriptorsCurrentFace);
        }
        return descriptorsDatabase;
    }
    //CONVERTS THE FACES ARRAY IN DATABASE INTO A MAT ARRAY
    private List<Mat> fromBitmapToMat(){
        //CREATE VARIABLE
        List<Mat> facesMat = new ArrayList<>();
        //LOOP THROUGH ALL THE FACES
        for(int i = 0; i< faces.length;i ++){
            //TEMPORAL MAT
            Mat tmpMat = new Mat();
            //FROM BITMAP TO MAT
            Utils.bitmapToMat((Bitmap) faces[i], tmpMat);
            //ADD CURRENT MAT TO ARRAY
            facesMat.add(tmpMat);
        }
        return facesMat;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }

    @Override
    protected void onStop() {
        super.onStop();

    }
    //THIS METHOD GENERATES A HISTOGRAM GIVEN THE DISTANCES VECTOR
    private int[] generateHistogram(int[] hist,int k){
        //CREATE HISTOGRAM
        int[] histogram = new int[k];
        //LOOP THROUGH ALL THE VECTOR
        for(int i=1; i < hist.length;i++){
            //ADD ONE TO THE CURRENT VALUE OF HISTOGRAM
            histogram[hist[i]] ++;

        }

        return histogram;
    }

    //THIS METHOD MATCHES A LIST OF HISTOGRAMS WITH A SINGLE HISTOGRAM APPLYING KNN (K=1)
    private int matchTestWithTraining(List<int[]> training,int[] test){
        //INITIALISING THE DISTANCE (INF)
        int currentDistance = Integer.MAX_VALUE;
        //CURRENT INDEX (NONE)
        int currentIndex = -1;
        //LOOP THROUGH ALL THE HISTOGRAMS
        for(int i =0 ; i < training.size(); i++){
            int accummulator = 0;
            //CALCULATE EACH DISTANCE AND ADD THEM TO THE ACCUMULATORS
            for(int j =0;j < test.length; j++)
                accummulator = accummulator + (training.get(i)[j] - test[j])*(training.get(i)[j] - test[j]);
            //SQRT THE SUM OF ALL THE DISTANCES
            Math.sqrt(accummulator);
            //IF CURRENT FACE ITS CLOSER, SAVE INDEX
            if(accummulator < currentDistance){
                currentDistance = accummulator;
                currentIndex = i;
            }
        }

        return currentIndex;
    }

}