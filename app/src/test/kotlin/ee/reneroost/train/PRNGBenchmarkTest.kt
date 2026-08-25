package ee.reneroost.train

import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import java.io.File
import kotlin.test.assertEquals

class PRNGBenchmarkTest {

    @Test
    fun `benchmark executes and logs metrics correctly`() {
        val tempFile = File.createTempFile("benchmark_test_", ".csv")
        tempFile.deleteOnExit()

        val benchmark = PRNGBenchmark()
        benchmark.runBenchmark(iterations = 100_000, outputFile = tempFile)

        val lines = tempFile.readLines()
        assertEquals(2, lines.size, "Should contain header and 1 result row")

        val dataParts = lines[1].split(",")
        assertEquals("1", dataParts[0], "Step should be 1")
        assertTrue(dataParts[3].toDouble() > 0.0, "tokensPerSec should be positive")
        assertTrue(dataParts[4].toLong() >= 0L, "elapsedMs should be valid duration")
    }
}