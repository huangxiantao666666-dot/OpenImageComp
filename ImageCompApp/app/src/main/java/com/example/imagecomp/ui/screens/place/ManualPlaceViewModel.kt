package com.example.imagecomp.ui.screens.place

import android.content.Context
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.example.imagecomp.data.repository.CompRepository
import com.example.imagecomp.util.Constants
import com.example.imagecomp.util.ImageCompressor
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

// ---------------------------------------------------------------------------
//  UI State
// ---------------------------------------------------------------------------

data class ManualPlaceUiState(
    val bgUri: Uri? = null,
    val fgUri: Uri? = null,
    val maskUri: Uri? = null,
    val scale: Float = Constants.DEFAULT_MANUAL_SCALE,
    val rotation: Float = Constants.DEFAULT_MANUAL_ROTATION,
    val harmonize: Boolean = false,
    val shadow: Boolean = true,             // UI toggle — server doesn't support yet
    val maskSource: String = Constants.DEFAULT_MASK_SOURCE,
    val isLoading: Boolean = false,
    val error: String? = null,
    // --- placement state ---
    val placedCount: Int = 0,
    val lastScore: Double? = null,
    val lastClickX: Int = 0,
    val lastClickY: Int = 0,
    val bgDisplayBytes: ByteArray? = null,  // current composite shown to user
    val originalBgBytes: ByteArray? = null,  // original bg for reset
)

// ---------------------------------------------------------------------------
//  ViewModel
// ---------------------------------------------------------------------------

class ManualPlaceViewModel(private val repository: CompRepository) : ViewModel() {

    private val _uiState = MutableStateFlow(ManualPlaceUiState())
    val uiState: StateFlow<ManualPlaceUiState> = _uiState.asStateFlow()

    // Stack of composites for undo: each entry is the composite BEFORE the
    // corresponding placement.  Index 0 = original background.
    private val compositeStack = mutableListOf<ByteArray>()

    // ── Setters ───────────────────────────────────────────────────

    fun setBg(uri: Uri) {
        _uiState.update { it.copy(bgUri = uri, placedCount = 0, lastScore = null) }
        compositeStack.clear()
    }
    fun setFg(uri: Uri) { _uiState.update { it.copy(fgUri = uri) } }
    fun setMask(uri: Uri?) { _uiState.update { it.copy(maskUri = uri) } }
    fun setScale(s: Float) { _uiState.update { it.copy(scale = s) } }
    fun setRotation(r: Float) { _uiState.update { it.copy(rotation = r) } }
    fun setHarmonize(on: Boolean) { _uiState.update { it.copy(harmonize = on) } }
    fun setShadow(on: Boolean) { _uiState.update { it.copy(shadow = on) } }
    fun setMaskSource(s: String) { _uiState.update { it.copy(maskSource = s) } }
    fun clearError() { _uiState.update { it.copy(error = null) } }

    // ── Actions ───────────────────────────────────────────────────

    /**
     * Called when the user taps on the displayed composite image.
     * Sends the current composite as bg to accumulate objects.
     */
    fun onTap(context: Context, tapX: Float, tapY: Float) {
        val state = _uiState.value
        val fgUri = state.fgUri ?: run {
            _uiState.update { it.copy(error = "Please select a foreground image.") }
            return
        }

        // Determine which bg bytes to send: current composite or original bg
        val bgBytes = state.bgDisplayBytes ?: run {
            _uiState.update { it.copy(error = "Please select a background image first.") }
            return
        }

        viewModelScope.launch {
            _uiState.update { it.copy(isLoading = true, error = null) }

            try {
                val fgBytes = ImageCompressor.compress(context, fgUri)
                val maskBytes = state.maskUri?.let { ImageCompressor.compress(context, it) }

                val result = repository.manualPlace(
                    bgBytes = bgBytes,
                    fgBytes = fgBytes,
                    maskBytes = maskBytes,
                    x = tapX.toInt(),
                    y = tapY.toInt(),
                    scale = state.scale,
                    rotation = state.rotation,
                    harmonize = state.harmonize,
                )

                result
                    .onSuccess { response ->
                        // Decode the returned composite
                        val newComposite = CompRepository.decodeBase64Jpeg(response.composite)

                        // Push current display to stack for undo
                        compositeStack.add(bgBytes)
                        // Update display to new composite
                        _uiState.update {
                            it.copy(
                                isLoading = false,
                                bgDisplayBytes = newComposite,
                                placedCount = compositeStack.size,
                                lastScore = response.scoreSimopa,
                                lastClickX = tapX.toInt(),
                                lastClickY = tapY.toInt(),
                            )
                        }
                    }
                    .onFailure { e ->
                        _uiState.update {
                            it.copy(isLoading = false, error = e.message ?: "Placement failed")
                        }
                    }
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(isLoading = false, error = e.message ?: "Unexpected error")
                }
            }
        }
    }

    /**
     * Load background bytes from URI. Call when bg is first selected.
     */
    fun loadBg(context: Context) {
        val uri = _uiState.value.bgUri ?: return
        viewModelScope.launch {
            try {
                val bytes = ImageCompressor.compress(context, uri)
                compositeStack.clear()
                _uiState.update {
                    it.copy(
                        bgDisplayBytes = bytes,
                        originalBgBytes = bytes,
                        placedCount = 0,
                        lastScore = null,
                    )
                }
            } catch (e: Exception) {
                _uiState.update { it.copy(error = "Cannot load background: ${e.message}") }
            }
        }
    }

    /**
     * Undo the last placement — restore previous composite from stack.
     */
    fun undo() {
        if (compositeStack.isEmpty()) {
            _uiState.update { it.copy(error = "Nothing to undo.") }
            return
        }
        val previous = compositeStack.removeAt(compositeStack.lastIndex)
        _uiState.update {
            it.copy(
                bgDisplayBytes = previous,
                placedCount = compositeStack.size,
                lastScore = if (compositeStack.isEmpty()) null else it.lastScore,
            )
        }
    }

    /**
     * Reset to original background, clearing all placements.
     */
    fun reset() {
        val original = _uiState.value.originalBgBytes
        compositeStack.clear()
        _uiState.update {
            it.copy(
                bgDisplayBytes = original,
                placedCount = 0,
                lastScore = null,
            )
        }
    }

    // ── Factory ───────────────────────────────────────────────────

    class Factory(private val repository: CompRepository) : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>): T {
            return ManualPlaceViewModel(repository) as T
        }
    }
}
