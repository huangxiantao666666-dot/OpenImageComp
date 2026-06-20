package com.example.imagecomp.ui.navigation

import com.example.imagecomp.data.api.dto.PlaceResponse

/**
 * In-memory holder for passing [PlaceResponse] between screens.
 *
 * WHY: PlaceResponse contains base64-encoded image data (0.5–1.5 MB as a
 * JSON string). Passing this through Android Navigation arguments triggers
 * TransactionTooLargeException and crashes the app.
 *
 * Instead, the result is stashed here before navigation, and the
 * destination screen reads it back.  Call [consume] to get the data
 * and null out the holder (prevents stale data on config changes).
 */
object ResultHolder {
    private var _result: PlaceResponse? = null

    fun put(result: PlaceResponse) {
        _result = result
    }

    fun consume(): PlaceResponse? {
        val r = _result
        _result = null
        return r
    }
}
