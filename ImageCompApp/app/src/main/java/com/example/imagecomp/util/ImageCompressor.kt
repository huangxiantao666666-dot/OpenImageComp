package com.example.imagecomp.util

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import java.io.ByteArrayOutputStream

/**
 * Compress images before uploading to the server.
 *
 * Android camera photos are often 4000×3000 (4-8 MB), but the server
 * processes at 256×256 internally.  We resize to max 1024 px and
 * compress as JPEG to keep uploads fast.
 */
object ImageCompressor {

    private const val MAX_DIM = 1024
    private const val JPEG_QUALITY = 80

    /**
     * Read an image URI, resize so longest side ≤ [MAX_DIM],
     * and return JPEG bytes suitable for upload.
     */
    fun compress(context: Context, uri: Uri): ByteArray {
        val inputStream = context.contentResolver.openInputStream(uri)
            ?: throw IllegalStateException("Cannot open image: $uri")
        val original = BitmapFactory.decodeStream(inputStream)
        inputStream.close()

        val w = original.width
        val h = original.height
        val ratio = minOf(MAX_DIM.toFloat() / w, MAX_DIM.toFloat() / h, 1.0f)

        val resized = if (ratio < 1.0f) {
            Bitmap.createScaledBitmap(original, (w * ratio).toInt(), (h * ratio).toInt(), true)
        } else {
            original
        }

        val output = ByteArrayOutputStream()
        resized.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, output)

        if (resized !== original) resized.recycle()
        original.recycle()

        return output.toByteArray()
    }
}
