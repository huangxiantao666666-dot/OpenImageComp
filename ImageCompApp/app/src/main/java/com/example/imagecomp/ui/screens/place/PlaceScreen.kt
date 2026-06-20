package com.example.imagecomp.ui.screens.place

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.imagecomp.data.api.dto.PlaceResponse
import com.example.imagecomp.di.AppContainer
import com.example.imagecomp.ui.components.ErrorDialog
import com.example.imagecomp.ui.components.ImagePicker
import com.example.imagecomp.ui.components.LoadingOverlay

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PlaceScreen(
    onNavigateBack: () -> Unit,
    onNavigateToResult: (PlaceResponse) -> Unit,
    modifier: Modifier = Modifier,
    appContainer: AppContainer = remember { AppContainer() },
) {
    val viewModel: PlaceViewModel = viewModel(
        factory = PlaceViewModel.Factory(appContainer.repository)
    )
    val state by viewModel.uiState.collectAsStateWithLifecycle()
    val context = LocalContext.current

    // Clear stale result when screen is displayed
    LaunchedEffect(Unit) {
        viewModel.clearResult()
    }

    // Navigate when result arrives
    LaunchedEffect(state.result) {
        if (state.result != null) onNavigateToResult(state.result!!)
    }

    // ── UI ────────────────────────────────────────────────────────
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("New Composition") },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back")
                    }
                },
            )
        },
    ) { padding ->
        Box(Modifier.padding(padding)) {
            Column(
                modifier = modifier
                    .fillMaxSize()
                    .verticalScroll(rememberScrollState())
                    .padding(16.dp),
            ) {
                Text("Select Images", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                Spacer(Modifier.height(12.dp))

                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    ImagePicker(label = "Background", imageUri = state.bgUri, onPicked = { viewModel.setBg(it) }, modifier = Modifier.weight(1f))
                    ImagePicker(label = "Foreground", imageUri = state.fgUri, onPicked = { viewModel.setFg(it) }, modifier = Modifier.weight(1f))
                }

                Spacer(Modifier.height(24.dp))

                Text("Parameters", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                Spacer(Modifier.height(12.dp))

                // ── Method ───────────────────────────────────────────
                Text("Placement Method", style = MaterialTheme.typography.labelLarge)
                Spacer(Modifier.height(4.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    FilterChip(selected = state.method == "topnet", onClick = { viewModel.setMethod("topnet") },
                        label = { Text("TopNet") })
                    FilterChip(selected = state.method == "grid", onClick = { viewModel.setMethod("grid") },
                        label = { Text("Grid") })
                }

                Spacer(Modifier.height(12.dp))

                // ── FG Scale ─────────────────────────────────────────
                Text("Foreground Size: ${state.fgScale.toInt()}%")
                Slider(value = state.fgScale, onValueChange = { viewModel.setFgScale(it) }, valueRange = 5f..100f, steps = 18)

                // ── Grid params (visible only for grid method) ────────
                AnimatedVisibility(visible = state.method == "grid") {
                    Column {
                        Spacer(Modifier.height(8.dp))
                        Text("Grid Density: ${state.gridSize}×${state.gridSize}")
                        Slider(
                            value = state.gridSize.toFloat(),
                            onValueChange = { viewModel.setGridSize(it.toInt()) },
                            valueRange = 3f..9f, steps = 5,
                        )
                        Spacer(Modifier.height(4.dp))
                        Text("Number of Scales: ${state.nScales}")
                        Slider(
                            value = state.nScales.toFloat(),
                            onValueChange = { viewModel.setNScales(it.toInt()) },
                            valueRange = 1f..7f, steps = 5,
                        )
                    }
                }

                Spacer(Modifier.height(8.dp))

                // ── Score Mode ────────────────────────────────────────
                Text("SimOPA Scoring Mode", style = MaterialTheme.typography.labelLarge)
                Spacer(Modifier.height(4.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    for (mode in listOf("4ch", "3ch", "crop")) {
                        FilterChip(
                            selected = state.scoreMode == mode,
                            onClick = { viewModel.setScoreMode(mode) },
                            label = { Text(mode) },
                        )
                    }
                }

                Spacer(Modifier.height(12.dp))

                // ── Mask Source ──────────────────────────────────────
                Text("Mask Source", style = MaterialTheme.typography.labelLarge)
                Spacer(Modifier.height(4.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    for (src in listOf("sam2", "opencv", "alpha")) {
                        FilterChip(
                            selected = state.maskSource == src,
                            onClick = { viewModel.setMaskSource(src) },
                            label = { Text(src.uppercase()) },
                        )
                    }
                }

                Spacer(Modifier.height(12.dp))

                // ── Harmonize ─────────────────────────────────────────
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                    Text("Colour Harmonization")
                    Switch(checked = state.harmonize, onCheckedChange = { viewModel.setHarmonize(it) })
                }

                Spacer(Modifier.height(8.dp))

                // ── Top-K ─────────────────────────────────────────────
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                    Text("Results to show")
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        for (k in 1..5) {
                            FilterChip(
                                selected = state.topK == k,
                                onClick = { viewModel.setTopK(k) },
                                label = { Text("$k") },
                                modifier = Modifier.padding(horizontal = 2.dp),
                            )
                        }
                    }
                }

                Spacer(Modifier.height(32.dp))

                Button(
                    onClick = { viewModel.analyze(context) },
                    modifier = Modifier.fillMaxWidth().height(56.dp),
                    shape = MaterialTheme.shapes.large,
                    enabled = !state.isLoading,
                ) {
                    Text("Analyze", style = MaterialTheme.typography.titleMedium)
                }
            }

            if (state.isLoading) {
                LoadingOverlay(message = "AI is finding the best placement...")
            }
        }
    }

    if (state.error != null) {
        ErrorDialog(message = state.error!!, onDismiss = { viewModel.clearError() })
    }
}