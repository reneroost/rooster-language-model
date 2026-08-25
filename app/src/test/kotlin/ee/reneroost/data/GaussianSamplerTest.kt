package ee.reneroost.data

import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class GaussianSamplerTest {

    @Test
    fun `determinism preserved across sampler instances`() {
        val sampler1 = GaussianSampler(RandomNumberGenerator(42L))
        val sampler2 = GaussianSampler(RandomNumberGenerator(42L))

        repeat(100) {
            assertEquals(sampler1.nextGaussian(), sampler2.nextGaussian())
        }
    }

    @Test
    fun `statistical properties approximate normal distribution`() {
        val sampler = GaussianSampler(RandomNumberGenerator(42L))
        val sampleCount = 100_000
        var sum = 0.0
        var sumSq = 0.0

        val targetMean = 0.0f
        val targetStdDev = 1.0f

        repeat(sampleCount) {
            val sample = sampler.nextGaussian(targetMean, targetStdDev)
            sum += sample
            sumSq += sample * sample
        }

        val empiricalMean = sum / sampleCount
        val empiricalVariance = (sumSq / sampleCount) - (empiricalMean * empiricalMean)
        val empiricalStdDev = sqrt(empiricalVariance)

        assertTrue(abs(empiricalMean - targetMean) < 0.02, "Empirical mean $empiricalMean shifted from target $targetMean")
        assertTrue(abs(empiricalStdDev - targetStdDev) < 0.02, "Empirical stdDev $empiricalStdDev shifted from target $targetStdDev")
    }

    @Test
    fun `scales correctly with custom mean and stdDev`() {
        val sampler = GaussianSampler(RandomNumberGenerator(42L))
        val targetMean = 5.0f
        val targetStdDev = 2.0f
        val sampleCount = 100_000
        var sum = 0.0

        repeat(sampleCount) {
            sum += sampler.nextGaussian(targetMean, targetStdDev)
        }

        val empiricalMean = sum / sampleCount
        assertTrue(abs(empiricalMean - targetMean) < 0.05, "Scaled empirical mean $empiricalMean shifted from $targetMean")
    }
}