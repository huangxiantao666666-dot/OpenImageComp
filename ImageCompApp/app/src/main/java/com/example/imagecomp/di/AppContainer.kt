package com.example.imagecomp.di

import com.example.imagecomp.data.api.CompApiService
import com.example.imagecomp.data.repository.CompRepository
import com.example.imagecomp.util.Constants
import com.squareup.moshi.Moshi
import com.squareup.moshi.kotlin.reflect.KotlinJsonAdapterFactory
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.moshi.MoshiConverterFactory
import java.util.concurrent.TimeUnit

/**
 * Manual dependency-injection container.
 *
 * In a larger app you'd use Hilt or Koin; for this prototype,
 * a simple hand-rolled container keeps things straightforward.
 */
class AppContainer {

    // ── Moshi ─────────────────────────────────────────────────────
    private val moshi: Moshi = Moshi.Builder()
        .addLast(KotlinJsonAdapterFactory())
        .build()

    // ── OkHttp ────────────────────────────────────────────────────
    private val loggingInterceptor = HttpLoggingInterceptor().apply {
        level = HttpLoggingInterceptor.Level.BODY
    }

    private val okHttpClient: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(Constants.CONNECT_TIMEOUT, TimeUnit.SECONDS)
        .readTimeout(Constants.READ_TIMEOUT, TimeUnit.SECONDS)
        .writeTimeout(Constants.WRITE_TIMEOUT, TimeUnit.SECONDS)
        .addInterceptor(loggingInterceptor)
        .build()

    // ── Retrofit ──────────────────────────────────────────────────
    private val retrofit: Retrofit = Retrofit.Builder()
        .baseUrl(Constants.BASE_URL)
        .client(okHttpClient)
        .addConverterFactory(MoshiConverterFactory.create(moshi))
        .build()

    // ── Public dependencies ───────────────────────────────────────
    val apiService: CompApiService = retrofit.create(CompApiService::class.java)
    val repository: CompRepository = CompRepository(apiService)
}
