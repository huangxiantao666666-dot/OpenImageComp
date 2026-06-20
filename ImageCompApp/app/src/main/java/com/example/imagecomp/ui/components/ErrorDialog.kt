package com.example.imagecomp.ui.components

import androidx.compose.material3.*
import androidx.compose.runtime.Composable

/**
 * Standard error dialog with a single dismiss/retry button.
 */
@Composable
fun ErrorDialog(
    message: String,
    onDismiss: () -> Unit,
    title: String = "Error",
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(title) },
        text = { Text(message) },
        confirmButton = {
            TextButton(onClick = onDismiss) {
                Text("OK")
            }
        },
    )
}
