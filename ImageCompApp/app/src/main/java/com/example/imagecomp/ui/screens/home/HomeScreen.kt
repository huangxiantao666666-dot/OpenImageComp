package com.example.imagecomp.ui.screens.home

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp

@Composable
fun HomeScreen(
    serverOnline: Boolean,
    serverInfo: String,
    onNavigateToPlace: () -> Unit,
    onNavigateToManualPlace: () -> Unit = {},
    modifier: Modifier = Modifier,
) {
    Scaffold(modifier = modifier) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 24.dp, vertical = 16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Spacer(Modifier.height(24.dp))

            Icon(
                imageVector = Icons.Default.AutoAwesome,
                contentDescription = null,
                modifier = Modifier.size(64.dp),
                tint = MaterialTheme.colorScheme.primary,
            )
            Spacer(Modifier.height(12.dp))

            Text(
                text = "ImageCompose",
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary,
            )
            Spacer(Modifier.height(4.dp))
            Text(
                text = "AI-Powered Object Placement",
                style = MaterialTheme.typography.titleSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )

            Spacer(Modifier.height(20.dp))

            // Features card
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
                ),
            ) {
                Column(Modifier.padding(16.dp)) {
                    FeatureItem("1", "Select background & foreground images")
                    FeatureItem("2", "Auto: AI finds the best placement")
                    FeatureItem("3", "Manual: Tap to place objects")
                    FeatureItem("4", "View heatmaps, composites & scores")
                }
            }

            Spacer(Modifier.height(20.dp))

            // Auto Placement button
            Button(
                onClick = onNavigateToPlace,
                modifier = Modifier.fillMaxWidth().height(52.dp),
                shape = MaterialTheme.shapes.large,
            ) {
                Text("Auto Placement", style = MaterialTheme.typography.titleMedium)
            }

            Spacer(Modifier.height(12.dp))

            // Manual Placement button
            OutlinedButton(
                onClick = onNavigateToManualPlace,
                modifier = Modifier.fillMaxWidth().height(52.dp),
                shape = MaterialTheme.shapes.large,
            ) {
                Text("Manual Placement", style = MaterialTheme.typography.titleMedium)
            }

            Spacer(Modifier.height(20.dp))

            // Server status
            ServerStatusBadge(online = serverOnline, info = serverInfo)

            Spacer(Modifier.height(16.dp))
        }
    }
}

@Composable
private fun FeatureItem(number: String, text: String) {
    Row(
        modifier = Modifier.padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Surface(
            shape = MaterialTheme.shapes.small,
            color = MaterialTheme.colorScheme.primary,
            modifier = Modifier.size(24.dp),
        ) {
            Box(contentAlignment = Alignment.Center) {
                Text(
                    text = number,
                    color = MaterialTheme.colorScheme.onPrimary,
                    style = MaterialTheme.typography.labelSmall,
                )
            }
        }
        Spacer(Modifier.width(8.dp))
        Text(text = text, style = MaterialTheme.typography.bodyMedium)
    }
}

@Composable
private fun ServerStatusBadge(online: Boolean, info: String) {
    Surface(
        shape = MaterialTheme.shapes.small,
        color = if (online) MaterialTheme.colorScheme.primaryContainer
                else MaterialTheme.colorScheme.errorContainer,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Surface(
                shape = MaterialTheme.shapes.extraSmall,
                color = if (online) MaterialTheme.colorScheme.primary
                        else MaterialTheme.colorScheme.error,
                modifier = Modifier.size(8.dp),
            ) {}
            Spacer(Modifier.width(8.dp))
            Text(
                text = info,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}
