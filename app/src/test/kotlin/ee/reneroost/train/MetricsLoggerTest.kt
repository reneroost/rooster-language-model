package ee.reneroost.train

import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class MetricsLoggerTest {

    @Test
    fun `creates file with header on initialization`() {
        val tempFile = File.createTempFile("metrics_test_", ".csv")
        tempFile.deleteOnExit()

        MetricsLogger(tempFile).use { }

        val lines = tempFile.readLines()
        assertEquals(1, lines.size, "Should contain only header line")
        assertEquals("step,loss,valLoss,tokensPerSec,elapsedMs", lines[0])
    }

    @Test
    fun `appends metrics correctly`() {
        val tempFile = File.createTempFile("metrics_test_", ".csv")
        tempFile.deleteOnExit()

        MetricsLogger(tempFile).use { logger ->
            logger.log(step = 0, loss = 4.1823f, valLoss = 4.2105f, tokensPerSec = 1250.5, elapsedMs = 150)
            logger.log(step = 100, loss = 2.4101f, valLoss = null, tokensPerSec = 1310.0, elapsedMs = 1200)
        }

        val lines = tempFile.readLines()
        assertEquals(3, lines.size, "Should contain header and 2 metric rows")
        assertEquals("0,4.1823,4.2105,1250.5,150", lines[1])
        assertEquals("100,2.4101,,1310.0,1200", lines[2], "Null valLoss should format as empty field")
    }
}