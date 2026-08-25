package ee.reneroost.model

data class ModelConfig(
    val vocabSize: Int = 65,
    val blockSize: Int = 8,
    val nEmbd: Int = 32,
    val nLayer: Int = 4,
    val nHead: Int = 4,
    val dropout: Float = 0.0f
) {
    init {
        require(vocabSize > 0) { "vocabSize must be positive" }
        require(blockSize > 0) { "blockSize must be positive" }
        require(nEmbd > 0 && nEmbd % nHead == 0) { "nEmbd must be divisible by nHead" }
    }

    val headSize: Int get() = nEmbd / nHead
}