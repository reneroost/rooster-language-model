package ee.reneroost.train

import java.io.File
import java.io.PrintWriter
import java.io.FileWriter

class MetricsLogger(private val logFile: File) : AutoCloseable {
    private val writer: PrintWriter
    
    init {
        val exists = logFile.exists() && logFile.length() > 0L
        writer = PrintWriter(FileWriter(logFile, true))
        if (!exists) {
            writer.println("step,loss,valLoss,tokensPerSec,elapsedMs")
            writer.flush()
        }
    }

    fun log(step: Int, loss: Float, valLoss: Float?, tokensPerSec: Double, elapsedMs: Long) {
        val valLossStr = valLoss?.let { String.format("%.4f", it) } ?: ""
        writer.println("$step,${String.format("%.4f", loss)},$valLossStr,${String.format("%.1f", tokensPerSec)},$elapsedMs")
        writer.flush()
    }

    override fun close() {
        writer.close()
    }
}