package com.example.imagecomp.ui.screens.place

import android.graphics.BitmapFactory
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.TouchApp
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.imagecomp.di.AppContainer
import com.example.imagecomp.ui.components.ErrorDialog
import com.example.imagecomp.ui.components.ImagePicker
import com.example.imagecomp.ui.components.LoadingOverlay
import com.example.imagecomp.ui.components.ScoreBadge

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ManualPlaceScreen(
    onNavigateBack: () -> Unit,
    modifier: Modifier = Modifier,
    appContainer: AppContainer = remember { AppContainer() },
) {
    val viewModel: ManualPlaceViewModel = viewModel(
        factory = ManualPlaceViewModel.Factory(appContainer.repository)
    )
    val state by viewModel.uiState.collectAsStateWithLifecycle()
    val context = LocalContext.current

    // Track the layout size of the composite image to map tap coords
    var imageLayoutSize by remember { mutableStateOf(IntSize.Zero) }

    // Load bg bytes when bgUri changes
    LaunchedEffect(state.bgUri) {
        if (state.bgUri != null) viewModel.loadBg(context)
    }

    // ── UI ────────────────────────────────────────────────────────
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Manual Placement") },
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
                // ── Image pickers ──────────────────────────────────
                Text("Select Images", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                Spacer(Modifier.height(12.dp))

                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    ImagePicker(
                        label = "Background",
                        imageUri = state.bgUri,
                        onPicked = { viewModel.setBg(it) },
                        modifier = Modifier.weight(1f),
                    )
                    ImagePicker(
                        label = "Foreground",
                        imageUri = state.fgUri,
                        onPicked = { viewModel.setFg(it) },
                        modifier = Modifier.weight(1f),
                    )
                }

                Spacer(Modifier.height(16.dp))

                // ── Clickable composite image ──────────────────────
                Text("Tap to Place", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                Spacer(Modifier.height(4.dp))
                Text(
                    "Tap on the background image to place the foreground at that position.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Spacer(Modifier.height(8.dp))

                val displayBitmap = remember(state.bgDisplayBytes) {
                    state.bgDisplayBytes?.let {
                        BitmapFactory.decodeByteArray(it, 0, it.size)
                    }
                }

                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .aspectRatio(4f / 3f)
                        .clip(RoundedCornerShape(12.dp))
                        .border(2.dp, MaterialTheme.colorScheme.primary, RoundedCornerShape(12.dp))
                        .onSizeChanged { imageLayoutSize = it }
                        .pointerInput(state.bgDisplayBytes) {
                            detectTapGestures { offset ->
                                if (displayBitmap != null && imageLayoutSize.width > 0 && imageLayoutSize.height > 0) {
                                    // Map tap coords from layout space to image pixel space
                                    val imgW = displayBitmap!!.width.toFloat()
                                    val imgH = displayBitmap!!.height.toFloat()
                                    val layoutW = imageLayoutSize.width.toFloat()
                                    val layoutH = imageLayoutSize.height.toFloat()

                                    // ContentScale.Fit: compute letterbox offsets
                                    val scale = minOf(layoutW / imgW, layoutH / imgH)
                                    val renderedW = imgW * scale
                                    val renderedH = imgH * scale
                                    val offsetX = (layoutW - renderedW) / 2f
                                    val offsetY = (layoutH - renderedH) / 2f

                                    val imgX = ((offset.x - offsetX) / scale).toInt().coerceIn(0, displayBitmap!!.width - 1)
                                    val imgY = ((offset.y - offsetY) / scale).toInt().coerceIn(0, displayBitmap!!.height - 1)

                                    viewModel.onTap(context, imgX.toFloat(), imgY.toFloat())
                                }
                            }
                        },
                    contentAlignment = Alignment.Center,
                ) {
                    if (displayBitmap != null) {
                        Image(
                            bitmap = displayBitmap.asImageBitmap(),
                            contentDescription = "Tap to place foreground",
                            contentScale = ContentScale.Fit,
                            modifier = Modifier.fillMaxSize(),
                        )
                    } else {
                        // Placeholder
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            Icon(
                                imageVector = Icons.Default.TouchApp,
                                contentDescription = null,
                                modifier = Modifier.size(48.dp),
                                tint = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                            Spacer(Modifier.height(8.dp))
                            Text(
                                "Select a background image\nthen tap here to place",
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                textAlign = TextAlign.Center,
                            )
                        }
                    }
                }

                // ── Placed marker indicator ────────────────────────
                if (state.placedCount > 0 && state.lastScore != null) {
                    Spacer(Modifier.height(8.dp))
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        ScoreBadge(score = state.lastScore!!)
                        Spacer(Modifier.width(8.dp))
                        Text(
                            "${state.placedCount} object(s) placed",
                            style = MaterialTheme.typography.bodyMedium,
                        )
                        Spacer(Modifier.weight(1f))
                        Text(
                            "tap=(${state.lastClickX},${state.lastClickY})",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }

                Spacer(Modifier.height(20.dp))

                // ── Controls ───────────────────────────────────────
                Text("Placement Controls", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                Spacer(Modifier.height(12.dp))

                Text("FG Scale: ${state.scale.toInt()}% of background")
                Slider(
                    value = state.scale,
                    onValueChange = { viewModel.setScale(it) },
                    valueRange = 5f..80f,
                    steps = 14,
                )

                Spacer(Modifier.height(4.dp))

                Text("Rotation: ${state.rotation.toInt()}°")
                Slider(
                    value = state.rotation,
                    onValueChange = { viewModel.setRotation(it) },
                    valueRange = -180f..180f,
                    steps = 35,
                )

                Spacer(Modifier.height(12.dp))

                // ── Toggles ────────────────────────────────────────
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Column {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text("Shadow", style = MaterialTheme.typography.bodyLarge)
                            Spacer(Modifier.width(8.dp))
                            Switch(checked = state.shadow, onCheckedChange = { viewModel.setShadow(it) })
                        }
                        Text(
                            "(server support pending)",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    Column {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text("Harmonize", style = MaterialTheme.typography.bodyLarge)
                            Spacer(Modifier.width(8.dp))
                            Switch(checked = state.harmonize, onCheckedChange = { viewModel.setHarmonize(it) })
                        }
                    }
                }

                Spacer(Modifier.height(16.dp))

                // ── Mask source ────────────────────────────────────
                Text("Mask Source", style = MaterialTheme.typography.labelLarge)
                Spacer(Modifier.height(4.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    for (src in listOf("sam2", "opencv", "alpha", "upload")) {
                        FilterChip(
                            selected = state.maskSource == src,
                            onClick = { viewModel.setMaskSource(src) },
                            label = { Text(src.uppercase()) },
                        )
                    }
                }

                // Show mask upload if source=upload
                if (state.maskSource == "upload") {
                    Spacer(Modifier.height(8.dp))
                    ImagePicker(
                        label = "Upload Mask",
                        imageUri = state.maskUri,
                        onPicked = { viewModel.setMask(it) },
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(120.dp),
                    )
                }

                Spacer(Modifier.height(24.dp))

                // ── Action buttons ─────────────────────────────────
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    // Place button
                    Button(
                        onClick = {
                            // Place at center if no tap has been made yet
                            val bmp = displayBitmap
                            if (bmp != null) {
                                viewModel.onTap(context, (bmp.width / 2).toFloat(), (bmp.height / 2).toFloat())
                            }
                        },
                        modifier = Modifier.weight(1f).height(48.dp),
                        shape = MaterialTheme.shapes.large,
                        enabled = !state.isLoading && state.bgDisplayBytes != null && state.fgUri != null,
                    ) {
                        Text("Place at Center")
                    }
                }

                Spacer(Modifier.height(8.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    OutlinedButton(
                        onClick = { viewModel.undo() },
                        modifier = Modifier.weight(1f),
                        enabled = state.placedCount > 0 && !state.isLoading,
                    ) {
                        Text("Undo Last")
                    }
                    OutlinedButton(
                        onClick = { viewModel.reset() },
                        modifier = Modifier.weight(1f),
                        enabled = state.placedCount > 0 && !state.isLoading,
                        colors = ButtonDefaults.outlinedButtonColors(
                            contentColor = MaterialTheme.colorScheme.error,
                        ),
                    ) {
                        Text("Reset All")
                    }
                }

                Spacer(Modifier.height(32.dp))
            }

            if (state.isLoading) {
                LoadingOverlay(message = "Placing object...")
            }
        }
    }

    if (state.error != null) {
        ErrorDialog(message = state.error!!, onDismiss = { viewModel.clearError() })
    }
}
