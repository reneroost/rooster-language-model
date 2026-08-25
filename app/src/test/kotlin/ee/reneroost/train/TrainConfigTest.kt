package ee.reneroost.train

import kotlin.test.Test
import kotlin.test.assertFailsWith

class TrainConfigTest {

    @Test fun `throws exception on non-positive batchSize or steps`() {
        assertFailsWith<IllegalArgumentException> {
            TrainConfig(batchSize = 0)
        }
        assertFailsWith<IllegalArgumentException> {
            TrainConfig(maxSteps = -10)
        }
    }
}