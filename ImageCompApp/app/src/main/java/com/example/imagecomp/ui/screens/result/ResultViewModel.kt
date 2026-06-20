package com.example.imagecomp.ui.screens.result

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Paint
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.example.imagecomp.data.api.dto.InterpretResponse
import com.example.imagecomp.data.repository.CompRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.io.ByteArrayOutputStream

/**
 * UI state for the "Explain" tab in [ResultScreen].
 *
 *  - [selectedRank] == null         → show the gallery of composites
 *  - [selectedRank] != null         → show the 5 visualizations for that rank
 */
data class ResultUiState(
    val selectedRank: Int? = null,
    val isInterpreting: Boolean = false,
    val interpretResult: InterpretResponse? = null,
    val error: String? = null,
)

class ResultViewModel(private val repository: CompRepository) : ViewModel() {

    private val _uiState = MutableStateFlow(ResultUiState())
    val uiState: StateFlow<ResultUiState> = _uiState.asStateFlow()

    /**
     * Send the chosen composite + a bbox-derived foreground mask to
     * ``/api/interpret`` and store the result in [uiState].
     *
     * @param rank            1-based rank in the result gallery.
     * @param compositeBytes  the composite JPEG bytes (decoded from base64
     *                        in [com.example.imagecomp.data.api.dto.CompositeEntry.image]).
     * @param bboxNorm        normalized [0, 1] bounding box
     *                        [x1, y1, x2, y2] in composite coordinates.
     */
    fun interpret(rank: Int, compositeBytes: ByteArray, bboxNorm: List<Double>) {
        viewModelScope.launch {
            _uiState.update {
                it.copy(
                    selectedRank = rank,
                    isInterpreting = true,
                    interpretResult = null,
                    error = null,
                )
            }
            val maskBytes = buildBboxMask(compositeBytes, bboxNorm)
            repository.interpret(compositeBytes, maskBytes, classIdx = 1)
                .onSuccess { resp ->
                    _uiState.update {
                        it.copy(isInterpreting = false, interpretResult = resp)
                    }
                }
                .onFailure { e ->
                    _uiState.update {
                        it.copy(isInterpreting = false, error = e.message ?: "Unknown error")
                    }
                }
        }
    }

    /** Return to the gallery view, dropping the current explanations. */
    fun backToGallery() {
        _uiState.update { ResultUiState() }
    }

    fun clearError() {
        _uiState.update { it.copy(error = null) }
    }

    class Factory(private val repository: CompRepository) : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>): T {
            return ResultViewModel(repository) as T
        }
    }

    companion object {
        /**
         * Build a single-channel PNG mask from a normalized [0, 1] bbox.
         * White pixels = foreground, black = background. The mask is the same
         * width / height as the composite (the server resizes it to 256x256).
         */
        private fun buildBboxMask(
            compositeBytes: ByteArray,
            bboxNorm: List<Double>,
        ): ByteArray {
            val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeByteArray(compositeBytes, 0, compositeBytes.size, opts)
            val w = opts.outWidth.coerceAtLeast(1)
            val h = opts.outHeight.coerceAtLeast(1)
            val x1 = (bboxNorm[0] * w).toInt().coerceIn(0, w)
            val y1 = (bboxNorm[1] * h).toInt().coerceIn(0, h)
            val x2 = (bboxNorm[2] * w).toInt().coerceIn(0, w)
            val y2 = (bboxNorm[3] * h).toInt().coerceIn(0, h)
            val mask = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(mask)
            canvas.drawColor(0xFF000000.toInt())
            val paint = Paint().apply { color = 0xFFFFFFFF.toInt() }
            canvas.drawRect(
                x1.toFloat(), y1.toFloat(), x2.toFloat(), y2.toFloat(), paint,
            )
            val baos = ByteArrayOutputStream()
            mask.compress(Bitmap.CompressFormat.PNG, 100, baos)
            mask.recycle()
            return baos.toByteArray()
        }
    }
}
