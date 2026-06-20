package com.example.imagecomp.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.example.imagecomp.ui.theme.ScoreHigh
import com.example.imagecomp.ui.theme.ScoreLow
import com.example.imagecomp.ui.theme.ScoreMedium

/**
 * Small coloured badge showing a SimOPA score.
 */
@Composable
fun ScoreBadge(
    score: Double,
    modifier: Modifier = Modifier,
) {
    val bg = when {
        score >= 0.7 -> ScoreHigh
        score >= 0.5 -> ScoreMedium
        else -> ScoreLow
    }
    val label = String.format("%.2f", score)

    Text(
        text = label,
        color = Color.White,
        style = MaterialTheme.typography.labelMedium,
        modifier = modifier
            .clip(RoundedCornerShape(6.dp))
            .background(bg)
            .padding(horizontal = 8.dp, vertical = 3.dp),
    )
}
