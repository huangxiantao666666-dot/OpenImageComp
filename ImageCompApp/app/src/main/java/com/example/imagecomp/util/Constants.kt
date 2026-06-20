package com.example.imagecomp.util

/**
 * App-wide constants. Change [BASE_URL] to point to your server.
 */
object Constants {
    // ── Server ─────────────────────────────────────────────────────
    // Android emulator → host machine:   http://10.0.2.2:8000/
    // Physical device on same WiFi:       http://192.168.x.x:8000/
    // ngrok / production:                 https://your-domain.com/
    const val BASE_URL = "http://192.168.221.173:8000/"

    // ── HTTP timeouts (seconds) ───────────────────────────────────
    const val CONNECT_TIMEOUT = 30L
    const val READ_TIMEOUT = 120L
    const val WRITE_TIMEOUT = 60L

    // ── Inference defaults ────────────────────────────────────────
    const val DEFAULT_FG_SCALE = 30f
    const val DEFAULT_TOP_K = 5
    const val DEFAULT_MAX_DIM = 1024
    const val DEFAULT_GRID_SIZE = 5
    const val DEFAULT_N_SCALES = 5
    const val DEFAULT_SCORE_MODE = "4ch"
    const val DEFAULT_MASK_SOURCE = "opencv"
    const val DEFAULT_MANUAL_SCALE = 25f
    const val DEFAULT_MANUAL_ROTATION = 0f
}
