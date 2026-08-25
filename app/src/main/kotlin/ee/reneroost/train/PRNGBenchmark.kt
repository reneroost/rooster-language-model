package ee.reneroost.train

import ee.reneroost.data.RandomNumberGenerator
import java.io.File
import kotlin.system.measureTimeMillis

class PRNGBenchmark {

    fun runBenchmark(iterations: Int = 10_000_000, outputFile: File? = null) {
        val rng = RandomNumberGenerator(42L)
        val logger = outputFile?.let { MetricsLogger(it) }

        repeat(100_000) { rng.nextFloat() }

        val elapsedMs = measureTimeMillis {
            repeat(iterations) {
                rng.nextFloat()
            }
        }

        val tokensPerSec = (iterations.toDouble() / (elapsedMs / 1000.0))

        logger?.use {
            it.log(
                step = 1,
                loss = 0.0f,
                valLoss = null,
                tokensPerSec = tokensPerSec,
                elapsedMs = elapsedMs
            )
        }
    }
}