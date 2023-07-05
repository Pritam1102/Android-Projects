package com.example.tensorflowml;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.databinding.DataBindingUtil;

import com.example.tensorflowml.databinding.ActivityMainBinding;
import com.example.tensorflowml.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    ActivityMainBinding binding;
    int imageSize = 32;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding= DataBindingUtil.setContentView(this,R.layout.activity_main);

        binding.Camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    if(checkSelfPermission(Manifest.permission.CAMERA)== PackageManager.PERMISSION_GRANTED){
                        Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                        startActivityForResult(cameraIntent,3);
                    }else {
                        requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                    }

                }
            }
        });

        binding.Gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
            Intent galleryIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
            startActivityForResult(galleryIntent, 1);

            }
        });

    }

    public void classifyImage(Bitmap image){
        try {

            Model model = Model.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 32, 32, 3}, DataType.FLOAT32);

            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4*imageSize*3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize*imageSize];
            image.getPixel(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());

            int pixel = 0;
            for(int i =0; i<imageSize;i++){
                for(int j=0; j<imageSize; j++){
                    int val = intValues[pixel++];
                    byteBuffer.putFloat(((val>>16)&0xFF)*(1.f/1));
                    byteBuffer.putFloat(((val>>8)&0xFF)*(1.f/1));
                    byteBuffer.putFloat(((val &0xFF)*(1.f/1)));

                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {

        if(resultCode==RESULT_OK){
            if(requestCode==3){
                Bitmap image = (Bitmap) data.getExtras().get("data");
                int dimension = Math.min(image.getWidth(),image.getHeight());
                image = ThumbnailUtils.extractThumbnail(image,dimension,dimension);
                binding.imageView.setImageBitmap(image);

                image =Bitmap.createScaledBitmap(image,imageSize,imageSize,false);
                classifyImage(image);
            }else{
                Uri dat = data.getData();
                Bitmap image =null;
                try {
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                binding.imageView.setImageBitmap(image);
                image =Bitmap.createScaledBitmap(image,imageSize,imageSize,false);
                classifyImage(image);
            }
        }

        super.onActivityResult(requestCode, resultCode, data);
    }
}