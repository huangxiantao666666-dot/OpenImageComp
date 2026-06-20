package com.example.imagecomp

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.navigation.compose.rememberNavController
import com.example.imagecomp.ui.navigation.NavGraph
import com.example.imagecomp.ui.theme.ImageCompTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            ImageCompTheme {
                val navController = rememberNavController()
                NavGraph(navController = navController)
            }
        }
    }
}
