package ee.reneroost.data

import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.math.cos
import kotlin.math.PI

class RandomNumberGenerator(seed: Long) {
    private var s0: Long = seed
    private var s1: Long = seed xor 0x9E3779B97F4A7C15UL.toLong()

    private var nextGaussianSample: Float? = null

    init {
        if (s0 == 0L && s1 == 0L) s1 = 1L
        repeat(10) { nextLong() }
    }

    fun nextLong(): Long {
        var x = s0
        val y = s1
        s0 = y
        x = x xor (x shl 23)
        s1 = x xor y xor (x ushr 17) xor (y ushr 26)
        return s1 + y
    }

    fun nextFloat(): Float {
        val bits = (nextLong() ushr 11)
        return (bits.toDouble() / (1L shl 53).toDouble()).toFloat()
    }

    fun nextInt(bound: Int): Int {
        require(bound > 0) { "Bound must be positive" }
        val r = (nextFloat() * bound).toInt()
        return if (r >= bound) bound - 1 else r
    }

    fun nextGaussian(): Float {
        nextGaussianSample?.let {
            nextGaussianSample = null
            return it
        }

        var u1 = nextFloat()
        while (u1 <= 1e-7f) { u1 = nextFloat() }
        val u2 = nextFloat()

        val radius = sqrt(-2.0f * ln(u1))
        val theta = 2.0f * PI.toFloat() * u2

        nextGaussianSample = radius * cos(theta)
        return radius * kotlin.math.sin(theta)
    }
}