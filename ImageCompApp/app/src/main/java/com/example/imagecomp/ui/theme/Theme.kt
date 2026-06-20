package com.example.imagecomp.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable

private val LightColorScheme = lightColorScheme(
    primary = Primary,
    secondary = Secondary,
    surface = SurfaceLight,
    background = Background,
    onPrimary = OnPrimary,
    onSecondary = OnSecondary,
    primaryContainer = PrimaryLight,
)

@Composable
fun ImageCompTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = LightColorScheme,
        content = content,
    )
}
