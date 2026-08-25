package ee.reneroost.data

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sqrt

class GaussianSampler(private val rng: RandomNumberGenerator) {

    fun nextGaussian(mean: Float = 0.0f, stdDev: Float = 1.0f): Float {
        var u1 = rng.nextFloat()
        while (u1 <= 0.0f) u1 = rng.nextFloat()
        val u2 = rng.nextFloat()

        val radius = sqrt(-2.0f * ln(u1))
        val theta = (2.0f * PI * u2).toFloat()
        val z0 = radius * cos(theta)

        return mean + z0 * stdDev
    }
}