# Add project specific ProGuard rules here.
# By default, the flags in this file are appended to flags specified
# in the SDK tools.

# Moshi
-keepattributes Signature
-keepattributes *Annotation*
-keep class com.example.imagecomp.data.api.dto.** { *; }
-keep class com.squareup.moshi.** { *; }
-dontwarn com.squareup.moshi.**

# OkHttp
-dontwarn okhttp3.**
-dontwarn okio.**
