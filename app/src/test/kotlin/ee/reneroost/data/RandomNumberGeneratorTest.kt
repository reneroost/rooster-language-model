package ee.reneroost.data

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

class RandomNumberGeneratorTest {

    @Test
    fun `seed determinism produces identical sequence`() {
        val rng1 = RandomNumberGenerator(42L)
        val rng2 = RandomNumberGenerator(42L)

        repeat(100) {
            assertEquals(rng1.nextLong(), rng2.nextLong())
        }
    }

    @Test
    fun `different seeds produce different outputs`() {
        val rng1 = RandomNumberGenerator(42L)
        val rng2 = RandomNumberGenerator(1337L)

        assertNotEquals(rng1.nextLong(), rng2.nextLong())
    }

    @Test
    fun `nextFloat output within range 0 to 1`() {
        val rng = RandomNumberGenerator(42L)

        repeat(1_000) {
            val value = rng.nextFloat()
            assertTrue(value >= 0.0f && value < 1.0f, "Float value $value out of range [0, 1)")
        }
    }

    @Test
    fun `nextInt stays within specified bound`() {
        val rng = RandomNumberGenerator(42L)
        val bound = 10

        repeat(1_000) {
            val value = rng.nextInt(bound)
            assertTrue(value in 0 until bound, "Int value $value out of bounds [0, $bound)")
        }
    }

    @Test
    fun `nextInt throws exception on non-positive bound`() {
        val rng = RandomNumberGenerator(42L)

        assertFailsWith<IllegalArgumentException> {
            rng.nextInt(0)
        }
        assertFailsWith<IllegalArgumentException> {
            rng.nextInt(-5)
        }
    }
}