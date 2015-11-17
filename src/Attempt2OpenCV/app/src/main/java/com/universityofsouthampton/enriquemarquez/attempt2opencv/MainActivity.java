package com.universityofsouthampton.enriquemarquez.attempt2opencv;

import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.os.Bundle;
import android.support.v7.app.ActionBarActivity;
import android.util.Log;
import android.view.Gravity;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;

import org.opencv.android.OpenCVLoader;


public class MainActivity extends ActionBarActivity {

    Button newPersonButton, scanButton, deleteButton;
    int selectedActivity;
    private Context myContext;

    static {

        if(!OpenCVLoader.initDebug()){
            Log.i("openCV","OpenCV was not initialized correctly!!");
        }else{
            Log.i("openCV","OpenCV was initialized correctly!!");
        }

    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        myContext = this;
        newPersonButton = (Button) findViewById(R.id.insert_Button);
        scanButton = (Button) findViewById(R.id.scan_Button);
        deleteButton = (Button) findViewById(R.id.delete_Button);

        selectedActivity = 0;

        newPersonButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, TakePicture.class);
                intent.putExtra("New Person", true);
                startActivity(intent);
                finish();
            }
        });
        scanButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                orborfisher();
            }
        });
        deleteButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                checkDeleteDatabase();
            }
        });

    }

    private void orborfisher() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Method");
        builder.setMessage("Which method would you like");
        builder.setPositiveButton("Fisher", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                //User clicked fisher
                Intent intent = new Intent(MainActivity.this, TakePicture.class);
                intent.putExtra("New Person", false);
                intent.putExtra("Fisher", true);
                startActivity(intent);
                finish();
            }
        });
        builder.setNegativeButton("Feature Matching", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                //User clicked orb
                Intent intent = new Intent(MainActivity.this, TakePicture.class);
                intent.putExtra("New Person", false);
                intent.putExtra("Fisher", false);
                startActivity(intent);
                finish();
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

    private void checkDeleteDatabase() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Delete Database Confirmation");
        builder.setMessage("Are you sure you want to delete the database");
        builder.setPositiveButton("Yes", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                //User clicked yes so delete the database
                DatabaseOperations DB = new DatabaseOperations(myContext);
                DB.removeAll(DB);
            }
        });
        builder.setNegativeButton("No", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                //User clicked no so return to main screen
                dialog.cancel();
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
        finish();

    }
}
