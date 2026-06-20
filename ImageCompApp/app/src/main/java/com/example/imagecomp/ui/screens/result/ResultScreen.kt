package com.example.imagecomp.ui.screens.result

import android.graphics.BitmapFactory
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.GridItemSpan
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.itemsIndexed
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Fullscreen
import androidx.compose.material.icons.filled.Save
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.example.imagecomp.data.api.dto.PlaceResponse
import com.example.imagecomp.data.repository.CompRepository
import com.example.imagecomp.ui.components.ErrorDialog
import com.example.imagecomp.ui.components.LoadingOverlay
import com.example.imagecomp.ui.components.ScoreBadge

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ResultScreen(
    result: PlaceResponse,
    onBack: () -> Unit,
    viewModel: ResultViewModel,
    modifier: Modifier = Modifier,
) {
    var selectedTab by remember { mutableIntStateOf(0) }
    val tabs = listOf("Heatmap", "Gallery", "Scores", "Explain")

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Results") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back")
                    }
                },
            )
        },
    ) { padding ->
        Column(
            modifier = modifier
                .fillMaxSize()
                .padding(padding),
        ) {
            // ── Summary bar ───────────────────────────────────────
            Surface(
                color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
                modifier = Modifier.fillMaxWidth(),
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 10.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Text(
                        "${result.totalCandidates} candidates",
                        style = MaterialTheme.typography.labelLarge,
                    )
                    Text(
                        "${result.elapsedSeconds}s",
                        style = MaterialTheme.typography.labelLarge,
                        color = MaterialTheme.colorScheme.primary,
                    )
                }
            }

            // ── Tabs ──────────────────────────────────────────────
            TabRow(selectedTabIndex = selectedTab) {
                tabs.forEachIndexed { idx, title ->
                    Tab(
                        selected = selectedTab == idx,
                        onClick = { selectedTab = idx },
                        text = { Text(title) },
                    )
                }
            }

            // ── Content ───────────────────────────────────────────
            when (selectedTab) {
                0 -> HeatmapTab(result)
                1 -> GalleryTab(result)
                2 -> ScoresTab(result)
                3 -> ExplainTab(result, viewModel)
            }
        }
    }
}

// -------------------------------------------------------------------
//  Tab 1: Heatmap
// -------------------------------------------------------------------
@Composable
private fun HeatmapTab(result: PlaceResponse) {
    // Heatmap is a base64 JPEG string — decode and display
    val heatmapBytes = remember(result) { CompRepository.decodeBase64Jpeg(result.heatmap) }
    val bitmap = remember(result) { BitmapFactory.decodeByteArray(heatmapBytes, 0, heatmapBytes.size) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
    ) {
        if (bitmap != null) {
            Card(
                shape = RoundedCornerShape(12.dp),
                elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
            ) {
                Image(
                    bitmap = bitmap.asImageBitmap(),
                    contentDescription = "Placement heatmap",
                    contentScale = ContentScale.FillWidth,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
            Spacer(Modifier.height(12.dp))
            Text(
                "Green = high score (good placement)  ·  Red = low score",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

// -------------------------------------------------------------------
//  Tab 2: Gallery (top-K composites in a 2-column grid)
// -------------------------------------------------------------------
@Composable
private fun GalleryTab(result: PlaceResponse) {
    var fullscreenImage by remember { mutableStateOf<ByteArray?>(null) }

    LazyVerticalGrid(
        columns = GridCells.Fixed(2),
        contentPadding = PaddingValues(12.dp),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        itemsIndexed(result.composites) { _index, entry ->
            val imgBytes = remember(entry) { CompRepository.decodeBase64Jpeg(entry.image) }
            val bmp = remember(entry) { BitmapFactory.decodeByteArray(imgBytes, 0, imgBytes.size) }

            Card(
                shape = RoundedCornerShape(12.dp),
                elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable { fullscreenImage = imgBytes },
            ) {
                Column {
                    // Composite thumbnail
                    Box {
                        if (bmp != null) {
                            Image(
                                bitmap = bmp.asImageBitmap(),
                                contentDescription = "Result #${entry.rank}",
                                contentScale = ContentScale.Crop,
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .aspectRatio(4f / 3f),
                            )
                        }else {
                            // 加上这段：如果解码失败，显示一个灰红色的占位框
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .aspectRatio(4f / 3f)
                                    .background(MaterialTheme.colorScheme.errorContainer),
                                contentAlignment = Alignment.Center
                            ) {
                                Text(
                                    "图片解析失败",
                                    color = MaterialTheme.colorScheme.onErrorContainer,
                                    style = MaterialTheme.typography.bodySmall
                                )
                            }
                        }
                        // Rank badge
                        Surface(
                            shape = RoundedCornerShape(bottomEnd = 8.dp),
                            color = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.align(Alignment.TopStart),
                        ) {
                            Text(
                                "#${entry.rank}",
                                color = MaterialTheme.colorScheme.onPrimary,
                                style = MaterialTheme.typography.labelLarge,
                                modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
                            )
                        }
                    }

                    // Score + info
                    Column(Modifier.padding(10.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            ScoreBadge(score = entry.scoreSimopa)
                            Spacer(Modifier.width(6.dp))
                            Text(
                                "SimOPA",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                        if (entry.scoreTopnet != null) {
                            Spacer(Modifier.height(2.dp))
                            Text(
                                "TopNet: ${String.format("%.3f", entry.scoreTopnet)}",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                        Spacer(Modifier.height(4.dp))
                        Text(
                            "(${entry.bbox[0]},${entry.bbox[1]})",
                            style = MaterialTheme.typography.labelSmall,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                        )
                    }
                }
            }
        }
    }

    // Fullscreen dialog
    if (fullscreenImage != null) {
        val fsBmp = remember(fullscreenImage) {
            fullscreenImage?.let { BitmapFactory.decodeByteArray(it, 0, it.size) }
        }
        Dialog(
            onDismissRequest = { fullscreenImage = null },
            properties = DialogProperties(usePlatformDefaultWidth = false),
        ) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .clickable { fullscreenImage = null },
                contentAlignment = Alignment.Center,
            ) {
                if (fsBmp != null) {
                    Image(
                        bitmap = fsBmp.asImageBitmap(),
                        contentDescription = "Fullscreen composite",
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp)
                            .clip(RoundedCornerShape(12.dp)),
                        contentScale = ContentScale.Fit,
                    )
                }
            }
        }
    }
}

// -------------------------------------------------------------------
//  Tab 3: Scores table
// -------------------------------------------------------------------
@Composable
private fun ScoresTab(result: PlaceResponse) {
    val table = result.table

    LazyColumn(
        contentPadding = PaddingValues(12.dp),
    ) {
        // Header
        item {
            Surface(
                color = MaterialTheme.colorScheme.surfaceVariant,
                shape = RoundedCornerShape(topStart = 12.dp, topEnd = 12.dp),
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 12.dp, vertical = 10.dp),
                ) {
                    table.columns.forEachIndexed { i, col ->
                        Text(
                            text = col,
                            style = MaterialTheme.typography.labelMedium,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.weight(
                                when (i) {
                                    0 -> 0.6f  // Rank
                                    table.columns.size - 1 -> 2.0f  // Position
                                    else -> 1.0f
                                }
                            ),
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                        )
                    }
                }
            }
        }

        // Rows
        itemsIndexed(table.rows) { idx, row ->
            val bg = if (idx % 2 == 0)
                MaterialTheme.colorScheme.surface
            else
                MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.3f)

            Surface(color = bg) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 12.dp, vertical = 8.dp),
                ) {
                    row.forEachIndexed { i, cell ->
                        Text(
                            text = cell,
                            style = MaterialTheme.typography.bodySmall,
                            modifier = Modifier.weight(
                                when (i) {
                                    0 -> 0.6f
                                    row.size - 1 -> 2.0f
                                    else -> 1.0f
                                }
                            ),
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                        )
                    }
                }
            }
            if (idx == table.rows.lastIndex) {
                Spacer(Modifier.height(8.dp))
            }
        }
    }
}

// -------------------------------------------------------------------
//  Tab 4: Explain  (interpretability — pick a composite to inspect)
// -------------------------------------------------------------------
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ExplainTab(
    result: PlaceResponse,
    viewModel: ResultViewModel,
) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()

    Box(Modifier.fillMaxSize()) {
        if (state.selectedRank == null) {
            ExplainGallery(
                result = result,
                onPick = { rank, bytes, bbox ->
                    viewModel.interpret(rank, bytes, bbox)
                },
            )
        } else {
            ExplainResult(
                rank = state.selectedRank!!,
                state = state,
                onBack = { viewModel.backToGallery() },
            )
        }

        if (state.isInterpreting) {
            LoadingOverlay(message = "Generating explanations ...")
        }
    }

    if (state.error != null) {
        ErrorDialog(message = state.error!!, onDismiss = { viewModel.clearError() })
    }
}

@Composable
private fun ExplainGallery(
    result: PlaceResponse,
    onPick: (rank: Int, compositeBytes: ByteArray, bboxNorm: List<Double>) -> Unit,
) {
    LazyVerticalGrid(
        columns = GridCells.Fixed(2),
        contentPadding = PaddingValues(12.dp),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        item(span = { GridItemSpan(maxLineSpan) }) {
            Column(Modifier.padding(bottom = 4.dp)) {
                Text(
                    "Pick a composite to explain",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                )
                Spacer(Modifier.height(2.dp))
                Text(
                    "Server runs Grad-CAM, Saliency, Occlusion and feature maps.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
        itemsIndexed(result.composites) { _index, entry ->
            val imgBytes = remember(entry) { CompRepository.decodeBase64Jpeg(entry.image) }
            val bmp = remember(entry) {
                BitmapFactory.decodeByteArray(imgBytes, 0, imgBytes.size)
            }

            Card(
                shape = RoundedCornerShape(12.dp),
                elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable { onPick(entry.rank, imgBytes, entry.bbox) },
            ) {
                Column {
                    Box {
                        if (bmp != null) {
                            Image(
                                bitmap = bmp.asImageBitmap(),
                                contentDescription = "Result #${entry.rank}",
                                contentScale = ContentScale.Crop,
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .aspectRatio(4f / 3f),
                            )
                        } else {
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .aspectRatio(4f / 3f)
                                    .background(MaterialTheme.colorScheme.errorContainer),
                                contentAlignment = Alignment.Center,
                            ) {
                                Text(
                                    "图片解析失败",
                                    color = MaterialTheme.colorScheme.onErrorContainer,
                                    style = MaterialTheme.typography.bodySmall,
                                )
                            }
                        }
                        Surface(
                            shape = RoundedCornerShape(bottomEnd = 8.dp),
                            color = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.align(Alignment.TopStart),
                        ) {
                            Text(
                                "#${entry.rank}",
                                color = MaterialTheme.colorScheme.onPrimary,
                                style = MaterialTheme.typography.labelLarge,
                                modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
                            )
                        }
                        // "Explain" hint icon on hover-tap
                        Surface(
                            shape = RoundedCornerShape(topStart = 8.dp),
                            color = MaterialTheme.colorScheme.tertiary,
                            modifier = Modifier.align(Alignment.TopEnd),
                        ) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                            ) {
                                Icon(
                                    Icons.Filled.AutoAwesome,
                                    contentDescription = null,
                                    modifier = Modifier.size(14.dp),
                                    tint = MaterialTheme.colorScheme.onTertiary,
                                )
                                Spacer(Modifier.width(4.dp))
                                Text(
                                    "Explain",
                                    color = MaterialTheme.colorScheme.onTertiary,
                                    style = MaterialTheme.typography.labelSmall,
                                )
                            }
                        }
                    }
                    Column(Modifier.padding(10.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            ScoreBadge(score = entry.scoreSimopa)
                            Spacer(Modifier.width(6.dp))
                            Text(
                                "SimOPA",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun ExplainResult(
    rank: Int,
    state: ResultUiState,
    onBack: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState()),
    ) {
        // Top bar
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 8.dp, vertical = 6.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            IconButton(onClick = onBack) {
                Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back to gallery")
            }
            Text(
                "Explanation · #$rank",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
            )
        }
        HorizontalDivider()

        val resp = state.interpretResult
        if (resp == null) {
            // Loading state handled by the parent Box; show a placeholder here
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(200.dp),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    "Waiting for server response ...",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            return@Column
        }

        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            InterpretCard("Grad-CAM", "Where the model 'looks' when scoring this placement.", resp.gradcam)
            InterpretCard("Saliency", "Input pixels whose change would alter the score the most.", resp.saliency)
            InterpretCard("Occlusion", "Score drop after grey-masking a sliding window.", resp.occlusion)
            InterpretCard("Features — Layer 2", "First 64 channels of ResNet layer2, normalized.", resp.featuresLayer2)
            InterpretCard("Features — Layer 4", "First 64 channels of ResNet layer4, normalized.", resp.featuresLayer4)
        }
    }
}

@Composable
private fun InterpretCard(title: String, subtitle: String, base64Png: String?) {
    Card(
        shape = RoundedCornerShape(12.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(Modifier.padding(12.dp)) {
            Text(
                title,
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold,
            )
            Spacer(Modifier.height(2.dp))
            Text(
                subtitle,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.height(8.dp))
            if (base64Png == null) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(120.dp)
                        .background(MaterialTheme.colorScheme.errorContainer),
                    contentAlignment = Alignment.Center,
                ) {
                    Text(
                        "Server failed to produce this view.",
                        color = MaterialTheme.colorScheme.onErrorContainer,
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            } else {
                val bytes = remember(base64Png) { CompRepository.decodeBase64Image(base64Png) }
                val bmp = remember(bytes) {
                    BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                }
                if (bmp != null) {
                    Image(
                        bitmap = bmp.asImageBitmap(),
                        contentDescription = title,
                        contentScale = ContentScale.FillWidth,
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(8.dp)),
                    )
                } else {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(120.dp)
                            .background(MaterialTheme.colorScheme.errorContainer),
                        contentAlignment = Alignment.Center,
                    ) {
                        Text(
                            "Decode failed",
                            color = MaterialTheme.colorScheme.onErrorContainer,
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                }
            }
        }
    }
}
