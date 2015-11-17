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
import android.view.Gravity;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.Toast;

import org.opencv.android.Utils;
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

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;

public class NewPerson extends Activity  {

    private Button add;
    private EditText nickname;
    private String selectedImagePath;
    private ImageView img;
    //NUMBER OF CLUSTERS TO USE IN FISHERS ALGORITHM
    private final static int NUM_CLUSTERS = 1000;
    //CENTRES AND COVARIANCES OF GMM VARIABLE
    private Mat centres = new Mat();
    //HISTOGRAMS OF THE DATABASE
    private List<int[]> histograms = new ArrayList<>();
    //FACES IN THE DATABASE
    private Bitmap[] faces;

    //FEATURE DETECTOR VARIABLE
    FeatureDetector featureDetector = FeatureDetector.create(FeatureDetector.DENSE);;
    //DESCRIPTOR EXTRACTOR VARIABLE
    DescriptorExtractor descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
    //DESCRIPTOR MATCHER VARIABLE
    DescriptorMatcher descriptorMatcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE);


    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_new_person);

        add = (Button) findViewById(R.id.button_add);
        nickname = (EditText) findViewById(R.id.textfield_nickname);
        img = (ImageView) findViewById(R.id.imageview);

        //add the nickname and picture to database
        add.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //addToPreferences();
                //if (nicknameIsInDatabase()) {
                //tellUser(nickname.getText().toString() + " is already in database, please use a different nickname");
                // }
                //else {
                addToDatabase();
                tellUser(nickname.getText().toString() + " has been added");
                //end the activity
                onBackPressed();
                // }
            }
        });

        initialise();
    }

    //show the image taken
    private void initialise() {
        selectedImagePath = getIntent().getStringExtra("path");
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inScaled = false;
        Bitmap bmp = BitmapFactory.decodeFile(selectedImagePath, options);
        img.setImageBitmap(bmp);

        checkFaceIsCorrect();
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

    //save the image and nickname to preferences
    private void addToPreferences() {
        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(this);
        SharedPreferences.Editor editor = prefs.edit();
        String name = nickname.getText().toString();
        //save to database
        editor.putString(name, selectedImagePath);
        editor.commit();
    }

    private boolean nicknameIsInDatabase() {
        String nicknameEntered = nickname.getText().toString();
        boolean nicknameInDatabase = false;
        DatabaseOperations dop = new DatabaseOperations(this);
        Cursor CR = dop.getInformation(dop);
        if (CR != null && CR.getCount() != 0) {
            CR.moveToFirst();
            //Check every nickname to see if any of them match the nickname entered
            do {
                //Get the values from the first column, as that is where the nicknames are stored
                String nickname = CR.getString(0);
                // if its already in the database
                if (nicknameEntered.equals(nickname)) {
                    nicknameInDatabase = true;
                    break;
                }
            } while (CR.moveToNext());
        }
        return nicknameInDatabase;
    }

    private void addToDatabase() {
        String name = nickname.getText().toString();
        DatabaseOperations dop = new DatabaseOperations(this);
        dop.putInformation(dop, name, selectedImagePath, null, null);

    }

    private void saveCentresToFile(Mat centres) {
        Bitmap bmp = Bitmap.createBitmap(centres.width(), centres.height(), Bitmap.Config.ARGB_8888);
        centres.convertTo(centres,CvType.CV_8UC1);
        Utils.matToBitmap(centres,bmp);
        File pictureFile = getOutputMediaFile();
        try {
            FileOutputStream fOut = new FileOutputStream(pictureFile);
            bmp.compress(Bitmap.CompressFormat.PNG, 100, fOut);
            fOut.flush();
            fOut.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    //Set up the filename and folder of the picture being taken
    private File getOutputMediaFile() {
        //make a new file directory inside the local directory folder
        File mediaStorageDir = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES).getPath(), "Selfie_Secure");

        //if the selfie secure folder does not exist
        if (!mediaStorageDir.exists()) {
            //if you cannot make this folder return
            if (!mediaStorageDir.mkdirs()) {
                return null;
            }
        }

        File mediaFile;
        //and make a media file:
        mediaFile = new File(mediaStorageDir.getPath() + File.separator + "Centroids.png");

        return mediaFile;
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
        super.onDestroy();
    }

    public void onBackPressed() {
        Intent intent = new Intent(NewPerson.this, MainActivity.class);
        startActivity(intent);
        finish();
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

}