package com.example.facedetectiontflite

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.nio.MappedByteBuffer

/**
 * Helper class responsible for loading the pretrained BlazeFace TFLite model,
 * preprocessing input bitmaps, running inference via the TFLite Interpreter,
 * and drawing bounding boxes around detected faces.
 */
class FaceDetectorHelper(private val context: Context) {

    // Core TensorFlow Lite Interpreter instance
    private var interpreter: Interpreter? = null

    // BlazeFace short-range model expects 128x128 RGB input images
    private val inputImageSize = 128

    // Model filename located inside app/src/main/assets/
    private val modelFileName = "blaze_face_short_range.tflite"

    init {
        setupInterpreter()
    }

    /**
     * Step 1: Load the .tflite model from the assets folder into a MappedByteBuffer
     * and initialize the TensorFlow Lite Interpreter.
     */
    private fun setupInterpreter() {
        try {
            val modelBuffer: MappedByteBuffer = FileUtil.loadMappedFile(context, modelFileName)
            val options = Interpreter.Options().apply {
                setNumThreads(4) // Use 4 CPU threads for fast on-device inference
            }
            interpreter = Interpreter(modelBuffer, options)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    /**
     * Step 2: Preprocess input Bitmap, run TFLite inference, and return annotated Bitmap + count.
     */
    fun detectFaces(originalBitmap: Bitmap): Pair<Bitmap, String> {
        val currentInterpreter = interpreter
            ?: return Pair(originalBitmap, "Error: Model file '$modelFileName' not found in assets folder!")

        // 1. Prepare Input: Resize image to 128x128 and normalize pixel values to [-1, 1]
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(inputImageSize, inputImageSize, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(127.5f, 127.5f))
            .build()

        var tensorImage = TensorImage.fromBitmap(originalBitmap)
        tensorImage = imageProcessor.process(tensorImage)

        // 2. Prepare Output Buffers matching BlazeFace output tensors:
        // Output 0: Regressors [1, 896, 16] (bounding box coordinates & 6 facial keypoints)
        // Output 1: Classificators [1, 896, 1] (raw confidence logits for each anchor)
        val outputBoxes = Array(1) { Array(896) { FloatArray(16) } }
        val outputScores = Array(1) { Array(896) { FloatArray(1) } }

        val outputs = mapOf(
            0 to outputBoxes,
            1 to outputScores
        )

        // 3. Run Inference
        val startTime = System.currentTimeMillis()
        currentInterpreter.runForMultipleInputsOutputs(arrayOf(tensorImage.buffer), outputs)
        val inferenceTime = System.currentTimeMillis() - startTime

        // 4. Post-process / Interpret Output Results
        val mutableBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)
        val boxPaint = Paint().apply {
            color = Color.GREEN
            style = Paint.Style.STROKE
            strokeWidth = 6f
        }
        val textPaint = Paint().apply {
            color = Color.GREEN
            textSize = 42f
            isFakeBoldText = true
        }

        var detectedCount = 0
        var maxConfidence = 0f

        // Find highest confidence anchor detections (Sigmoid activation on raw score)
        for (i in 0 until 896) {
            val rawScore = outputScores[0][i][0]
            val confidence = (1.0f / (1.0f + Math.exp((-rawScore).toDouble()))).toFloat()

            if (confidence > 0.75f && confidence > maxConfidence) {
                maxConfidence = confidence
                detectedCount = 1

                // Decode normalized box coordinates relative to image dimensions
                val cx = (outputBoxes[0][i][0] / inputImageSize) * originalBitmap.width
                val cy = (outputBoxes[0][i][1] / inputImageSize) * originalBitmap.height
                val w = (outputBoxes[0][i][2] / inputImageSize) * originalBitmap.width
                val h = (outputBoxes[0][i][3] / inputImageSize) * originalBitmap.height

                val left = (cx - w / 2f).coerceAtLeast(0f)
                val top = (cy - h / 2f).coerceAtLeast(0f)
                val right = (cx + w / 2f).coerceAtMost(originalBitmap.width.toFloat())
                val bottom = (cy + h / 2f).coerceAtMost(originalBitmap.height.toFloat())

                canvas.drawRect(RectF(left, top, right, bottom), boxPaint)
                val label = "Face: ${(confidence * 100).toInt()}%"
                canvas.drawText(label, left, (top - 10f).coerceAtLeast(40f), textPaint)
            }
        }

        val statusMessage = if (detectedCount > 0) {
            "Face Detected! Confidence: ${(maxConfidence * 100).toInt()}% | Time: ${inferenceTime}ms"
        } else {
            "No face detected (Threshold > 75%) | Time: ${inferenceTime}ms"
        }

        return Pair(mutableBitmap, statusMessage)
    }

    fun close() {
        interpreter?.close()
    }
}
