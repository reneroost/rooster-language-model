package ee.reneroost.model

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class ModelConfigTest {

    @Test
    fun `default config computes headSize correctly`() {
        val config = ModelConfig(nEmbd = 32, nHead = 4)
        assertEquals(8, config.headSize, "headSize must equal nEmbd / nHead")
    }

    @Test
    fun `throws exception when nEmbd is not divisible by nHead`() {
        assertFailsWith<IllegalArgumentException> {
            ModelConfig(nEmbd = 33, nHead = 4)
        }
    }

    @Test
    fun `throws exception on non-positive vocab or block size`() {
        assertFailsWith<IllegalArgumentException> {
            ModelConfig(vocabSize = 0)
        }
        assertFailsWith<IllegalArgumentException> {
            ModelConfig(blockSize = -1)
        }
    }
}