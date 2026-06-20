package com.example.imagecomp.ui.navigation

import androidx.compose.runtime.*
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import com.example.imagecomp.di.AppContainer
import com.example.imagecomp.ui.screens.home.HomeScreen
import com.example.imagecomp.ui.screens.place.ManualPlaceScreen
import com.example.imagecomp.ui.screens.place.PlaceScreen
import com.example.imagecomp.ui.screens.place.PlaceViewModel
import com.example.imagecomp.ui.screens.result.ResultScreen
import com.example.imagecomp.ui.screens.result.ResultViewModel

/**
 * Navigation route constants — simplified: no large data in arguments.
 */
object Routes {
    const val HOME = "home"
    const val PLACE = "place"
    const val MANUAL_PLACE = "manual_place"
    const val RESULT = "result"
}

@Composable
fun NavGraph(
    navController: NavHostController,
    appContainer: AppContainer = remember { AppContainer() },
) {
    val placeViewModel: PlaceViewModel = viewModel(
        factory = PlaceViewModel.Factory(appContainer.repository),
    )

    NavHost(
        navController = navController,
        startDestination = Routes.HOME,
    ) {
        composable(Routes.HOME) {
            val state by placeViewModel.uiState.collectAsStateWithLifecycle()

            HomeScreen(
                serverOnline = state.serverOnline,
                serverInfo = state.serverInfo,
                onNavigateToPlace = {
                    placeViewModel.checkHealth()
                    navController.navigate(Routes.PLACE)
                },
                onNavigateToManualPlace = {
                    placeViewModel.checkHealth()
                    navController.navigate(Routes.MANUAL_PLACE)
                },
            )
        }

        composable(Routes.PLACE) {
            PlaceScreen(
                onNavigateBack = { navController.popBackStack() },
                onNavigateToResult = { result ->
                    // Stash result in memory BEFORE navigation (avoids
                    // passing huge base64 strings as Nav arguments).
                    ResultHolder.put(result)
                    navController.navigate(Routes.RESULT)
                },
                appContainer = appContainer,
            )
        }

        composable(Routes.MANUAL_PLACE) {
            ManualPlaceScreen(
                onNavigateBack = { navController.popBackStack() },
                appContainer = appContainer,
            )
        }

        composable(Routes.RESULT) {
            // Consume the result from the in-memory holder.
            val result = remember { ResultHolder.consume() }
            val resultViewModel: ResultViewModel = viewModel(
                factory = ResultViewModel.Factory(appContainer.repository),
            )

            if (result != null) {
                ResultScreen(
                    result = result,
                    onBack = {
                        // Pop all the way to HOME to avoid PlaceScreen
                        // immediately re-navigating with a stale result.
                        navController.popBackStack(Routes.HOME, inclusive = false)
                    },
                    viewModel = resultViewModel,
                )
            } else {
                // Edge case: process was killed and the holder is empty.
                // Just go back — the user can try again.
                LaunchedEffect(Unit) {
                    navController.popBackStack()
                }
            }
        }
    }
}
