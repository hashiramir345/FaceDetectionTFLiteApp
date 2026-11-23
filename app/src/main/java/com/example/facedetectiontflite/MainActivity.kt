package com.example.facedetectiontflite

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat

/**
 * Main Activity handling UI interactions, image selection from Gallery/Camera,
 * and passing Bitmaps to FaceDetectorHelper for on-device TFLite inference.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var ivPreview: ImageView
    private lateinit var tvResult: TextView
    private lateinit var faceDetectorHelper: FaceDetectorHelper

    // Launcher for picking an image from device gallery
    private val galleryLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let {
            val bitmap = uriToBitmap(it)
            bitmap?.let { bmp -> processImage(bmp) }
        }
    }

    // Launcher for capturing a thumbnail photo using device camera
    private val cameraLauncher = registerForActivityResult(ActivityResultContracts.TakePicturePreview()) { bitmap: Bitmap? ->
        bitmap?.let { bmp -> processImage(bmp) }
    }

    // Runtime camera permission launcher
    private val permissionLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) {
            cameraLauncher.launch(null)
        } else {
            Toast.makeText(this, "Camera permission is required to take photos.", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        ivPreview = findViewById(R.id.ivPreview)
        tvResult = findViewById(R.id.tvResult)
        val btnGallery: Button = findViewById(R.id.btnSelectGallery)
        val btnCamera: Button = findViewById(R.id.btnCaptureCamera)

        // Initialize TFLite Face Detector
        faceDetectorHelper = FaceDetectorHelper(this)

        btnGallery.setOnClickListener {
            galleryLauncher.launch("image/*")
        }

        btnCamera.setOnClickListener {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                cameraLauncher.launch(null)
            } else {
                permissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }
    }

    /**
     * Converts selected Uri into a software ARGB_8888 Bitmap compatible with Canvas & TFLite.
     */
    private fun uriToBitmap(uri: Uri): Bitmap? {
        return try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                val source = ImageDecoder.createSource(contentResolver, uri)
                ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                    decoder.isMutableRequired = true
                    decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                }
            } else {
                @Suppress("DEPRECATION")
                MediaStore.Images.Media.getBitmap(contentResolver, uri)
            }
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    /**
     * Sends Bitmap to TFLite model and updates the UI with bounding boxes and confidence score.
     */
    private fun processImage(bitmap: Bitmap) {
        val (annotatedBitmap, resultMessage) = faceDetectorHelper.detectFaces(bitmap)
        ivPreview.setImageBitmap(annotatedBitmap)
        tvResult.text = resultMessage
    }

    override fun onDestroy() {
        super.onDestroy()
        faceDetectorHelper.close()
    }
}
