package com.example.imagecomp.ui.screens.place

import android.content.Context
import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.example.imagecomp.data.api.dto.PlaceResponse
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

data class PlaceUiState(
    val bgUri: Uri? = null,
    val fgUri: Uri? = null,
    val fgScale: Float = Constants.DEFAULT_FG_SCALE,
    val harmonize: Boolean = false,
    val topK: Int = Constants.DEFAULT_TOP_K,
    val method: String = "topnet",
    val gridSize: Int = Constants.DEFAULT_GRID_SIZE,
    val nScales: Int = Constants.DEFAULT_N_SCALES,
    val scoreMode: String = Constants.DEFAULT_SCORE_MODE,
    val maskSource: String = Constants.DEFAULT_MASK_SOURCE,
    val isLoading: Boolean = false,
    val error: String? = null,
    val result: PlaceResponse? = null,      // non-null → navigate to results
    val serverOnline: Boolean = false,
    val serverInfo: String = "Checking...",
)

// ---------------------------------------------------------------------------
//  ViewModel
// ---------------------------------------------------------------------------

class PlaceViewModel(private val repository: CompRepository) : ViewModel() {

    private val _uiState = MutableStateFlow(PlaceUiState())
    val uiState: StateFlow<PlaceUiState> = _uiState.asStateFlow()

    init {
        checkHealth()
    }

    // ── Setters ───────────────────────────────────────────────────

    fun setBg(uri: Uri) { _uiState.update { it.copy(bgUri = uri) } }
    fun setFg(uri: Uri) { _uiState.update { it.copy(fgUri = uri) } }
    fun setFgScale(scale: Float) { _uiState.update { it.copy(fgScale = scale) } }
    fun setHarmonize(on: Boolean) { _uiState.update { it.copy(harmonize = on) } }
    fun setTopK(k: Int) { _uiState.update { it.copy(topK = k) } }
    fun setMethod(m: String) { _uiState.update { it.copy(method = m) } }
    fun setGridSize(s: Int) { _uiState.update { it.copy(gridSize = s) } }
    fun setNScales(s: Int) { _uiState.update { it.copy(nScales = s) } }
    fun setScoreMode(m: String) { _uiState.update { it.copy(scoreMode = m) } }
    fun setMaskSource(s: String) { _uiState.update { it.copy(maskSource = s) } }
    fun clearResult() { _uiState.update { it.copy(result = null) } }
    fun clearError() { _uiState.update { it.copy(error = null) } }

    // ── Actions ───────────────────────────────────────────────────

    fun checkHealth() {
        viewModelScope.launch {
            repository.checkHealth()
                .onSuccess { health ->
                    val loaded = health.modelsLoaded.joinToString(", ")
                    _uiState.update {
                        it.copy(
                            serverOnline = true,
                            serverInfo = "Online · ${health.device} · $loaded",
                        )
                    }
                }
                .onFailure { e ->
                    _uiState.update {
                        it.copy(
                            serverOnline = false,
                            serverInfo = "Offline — ${e.message}",
                        )
                    }
                }
        }
    }

    fun analyze(context: Context) {
        val bgUri = _uiState.value.bgUri
        val fgUri = _uiState.value.fgUri
        if (bgUri == null || fgUri == null) {
            _uiState.update { it.copy(error = "Please select both background and foreground images.") }
            return
        }

        viewModelScope.launch {
            _uiState.update { it.copy(isLoading = true, error = null) }

            val result = repository.placeObjects(
                bgBytes = ImageCompressor.compress(context, bgUri),
                fgBytes = ImageCompressor.compress(context, fgUri),
                maskBytes = null,     // server auto-generates from mask_source
                method = _uiState.value.method,
                fgScale = _uiState.value.fgScale,
                topK = _uiState.value.topK,
                harmonize = _uiState.value.harmonize,
                // Forward the four params the UI exposes (radios + sliders).
                // Server side now reads each of these on /api/place.
                scoreMode = _uiState.value.scoreMode,
                maskSource = _uiState.value.maskSource,
                gridSize = _uiState.value.gridSize,
                nScales = _uiState.value.nScales,
            )

            result
                .onSuccess { response ->
                    _uiState.update { it.copy(isLoading = false, result = response) }
                }
                .onFailure { e ->
                    _uiState.update {
                        it.copy(isLoading = false, error = e.message ?: "Unknown error")
                    }
                }
        }
    }

    // ── Factory ───────────────────────────────────────────────────

    class Factory(private val repository: CompRepository) : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>): T {
            return PlaceViewModel(repository) as T
        }
    }
}
