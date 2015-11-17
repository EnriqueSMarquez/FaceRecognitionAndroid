package com.universityofsouthampton.enriquemarquez.attempt2opencv;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import com.universityofsouthampton.enriquemarquez.attempt2opencv.Table.TableInfo;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;

/**
 * Created by Adam on 30/04/2015.
 */
public class DatabaseOperations extends SQLiteOpenHelper {

    public static String strSeparator = "__,__";
    public static final int database_version = 1;
    public String CREATE_QUERY = "CREATE TABLE " + TableInfo.TABLE_NAME + "(" + TableInfo.NICKNAME
            + " TEXT, " + TableInfo.FILE_PATH + " TEXT, " + TableInfo.HISTOGRAM + " TEXT, "
            + TableInfo.CENTRES + " BLOB );";

    public DatabaseOperations(Context context) {
        super(context, TableInfo.DATABASE_NAME, null, database_version);
        Log.e("Database operations", "Database created");
    }

    @Override
    public void onCreate(SQLiteDatabase db) {
        db.execSQL(CREATE_QUERY);
    }

    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {

    }

    public void putInformation(DatabaseOperations dop, String nickname, String filepath,
                               int[] histogram, Mat centres) {
        SQLiteDatabase SQ = dop.getWritableDatabase();
        ContentValues cv = new ContentValues();
        cv.put(TableInfo.NICKNAME, nickname);
        cv.put(TableInfo.FILE_PATH, filepath);
        //String stringHistogram = convertArrayToString(histogram);
        //cv.put(TableInfo.HISTOGRAM, stringHistogram);
        if(centres != null) {
            byte[] blobCentres = serializeMat(centres);
            cv.put(TableInfo.CENTRES, blobCentres);
        }
        SQ.insert(TableInfo.TABLE_NAME, null, cv);
    }

    public Cursor getInformation(DatabaseOperations dop) {
        SQLiteDatabase SQ = dop.getReadableDatabase();
        String[] columns = {TableInfo.NICKNAME, TableInfo.FILE_PATH, TableInfo.HISTOGRAM, TableInfo.CENTRES};
        Cursor CR = SQ.query(TableInfo.TABLE_NAME, columns, null, null, null, null, null);
        return CR;
    }

    public Cursor getNickname(DatabaseOperations dop, String filepath) {
        SQLiteDatabase SQ = dop.getReadableDatabase();
        String selection = TableInfo.FILE_PATH + " LIKE ?";
        String[] columns = {TableInfo.NICKNAME};
        String[] args = {filepath};
        Cursor CR = SQ.query(TableInfo.TABLE_NAME, columns, selection, args, null, null, null);
        return CR;
    }

    public Mat getCentres(DatabaseOperations dop) {
        SQLiteDatabase SQ = dop.getReadableDatabase();
        Cursor CR = getInformation(dop);
        if (CR.getCount() != 0) {
            CR.moveToFirst();
            byte[] centresBlob = CR.getBlob(3);
            return deserializeMat(centresBlob);
        }
        else {
            return null;
        }
    }

    public void updateCentres(DatabaseOperations dop, Mat centres) {
        SQLiteDatabase SQ = dop.getWritableDatabase();
        Cursor CR = getInformation(dop);
        if (CR.getCount() != 0) {
            CR.moveToFirst();
            String nickname = CR.getString(0);
            String selection = TableInfo.NICKNAME + " LIKE ?";
            String[] args = {nickname};
            ContentValues cv = new ContentValues();
            cv.put(TableInfo.CENTRES, serializeMat(centres));
            SQ.update(TableInfo.TABLE_NAME, cv, selection, args);
        }
    }

    public int[] getHistogram(DatabaseOperations dop, String nickname) {
        SQLiteDatabase SQ = dop.getReadableDatabase();
        String selection = TableInfo.NICKNAME + " LIKE ?";
        String[] columns = {TableInfo.HISTOGRAM};
        String[] args = {nickname};
        Cursor CR = SQ.query(TableInfo.TABLE_NAME, columns, selection, args, null, null, null);
        if (CR.getCount() != 0) {
            CR.moveToFirst();
            String stringHistogram = CR.getString(0);
            return convertStringToArray(stringHistogram);
        }
        else {
            return null;
        }
    }

    public void updateHistogram(DatabaseOperations dop, String nickname, int[] histogram) {
        SQLiteDatabase SQ = dop.getWritableDatabase();
        String selection = TableInfo.NICKNAME + " LIKE ?";
        String[] args = {nickname};
        ContentValues cv = new ContentValues();
        cv.put(TableInfo.HISTOGRAM, convertArrayToString(histogram));
        SQ.update(TableInfo.TABLE_NAME, cv, selection, args);
    }

    public void removeAll(DatabaseOperations dop) {
        SQLiteDatabase SQ = dop.getWritableDatabase();
        SQ.delete(TableInfo.TABLE_NAME, null, null);
    }

    public static String convertArrayToString(int[] array){
        String str = "";
        for (int i = 0; i<array.length; i++) {
            str = str+array[i];
            // Do not append comma at the end of last element
            if(i<array.length-1){
                str = str+strSeparator;
            }
        }
        return str;
    }

    public static int[] convertStringToArray(String str){
        String[] arr = str.split(strSeparator);
        int[] intarr = new int[arr.length];
        for (int i = 0; i<arr.length; i++) {
            intarr[i] = Integer.parseInt(arr[i]);
        }
        return intarr;
    }

    public static byte[] serializeMat(Mat m) {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        Bitmap bmp = Bitmap.createBitmap(m.width(),m.height(),Bitmap.Config.ARGB_8888);
        bmp.compress(Bitmap.CompressFormat.PNG, 100, bos);
        byte[] byteArray = bos.toByteArray();


       /* try {
            ObjectOutput out = new ObjectOutputStream(bos);
            out.writeObject(m);
            out.close();

            // Get the bytes of the serialized object
            byte[] buf = bos.toByteArray();

            return buf;
        } catch(IOException ioe) {
            Log.e("serializeObject", "error", ioe);

            return null;
        }*/
        return  byteArray;
    }

    public static Mat deserializeMat(byte[] b) {
       // try {
            Bitmap bmp = BitmapFactory.decodeByteArray(b,0,b.length);
            Mat m = new Mat(bmp.getHeight(),bmp.getWidth(), CvType.CV_32F);
            Utils.bitmapToMat(bmp,m);


            return m;
//        } catch(ClassNotFoundException cnfe) {
//            Log.e("deserializeObject", "class not found error", cnfe);
//
//            return null;
//        } catch(IOException ioe) {
//            Log.e("deserializeObject", "io error", ioe);
//
//            return null;
//        }
    }

}
