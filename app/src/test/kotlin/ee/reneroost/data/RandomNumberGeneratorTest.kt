package ee.reneroost.data

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class RandomNumberGeneratorTest {

    @Test
    fun `same seed produces identical sequence of floats`() {
        val rng1 = RandomNumberGenerator(42L)
        val rng2 = RandomNumberGenerator(42L)

        repeat(100) {
            assertEquals(rng1.nextFloat(), rng2.nextFloat(), "Outputs should be bit-identical across identical seeds")
        }
    }

    @Test
    fun `nextFloat output is strictly bounded between 0 and 1`() {
        val rng = RandomNumberGenerator(12345L)

        repeat(10_000) {
            val valFloat = rng.nextFloat()
            assertTrue(valFloat >= 0.0f && valFloat < 1.0f, "float output $valFloat out of range [0.0, 1.0)")
        }
    }

    @Test
    fun `nextInt respects bounds and throws on non-positive input`() {
        val rng = RandomNumberGenerator(999L)
        val bound = 65

        repeat(1_000) {
            val valInt = rng.nextInt(bound)
            assertTrue(valInt >= 0 && valInt < bound, "int output $valInt out of bound $bound")
        }

        assertFailsWith<IllegalArgumentException> {
            rng.nextInt(0)
        }
    }

    @Test
    fun `nextGaussian follows standard normal distribution statistics`() {
        val rng = RandomNumberGenerator(777L)
        val samples = 100_000
        var sum = 0.0
        var sqSum = 0.0

        repeat(samples) {
            val g = rng.nextGaussian().toDouble()
            sum += g
            sqSum += g * g
        }

        val mean = sum / samples
        val variance = (sqSum / samples) - (mean * mean)

        // Empirical assertion: N(0, 1) mean should be close to 0.0 and variance close to 1.0
        assertTrue(abs(mean) < 0.02, "Sample mean $mean deviated too far from expected 0.0")
        assertTrue(abs(variance - 1.0) < 0.05, "Sample variance $variance deviated too far from expected 1.0")
    }
}