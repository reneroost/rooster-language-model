package ee.reneroost.train

data class TrainConfig(
    val batchSize: Int = 32,
    val maxSteps: Int = 1000,
    val learningRate: Float = 1e-3f,
    val minLearningRate: Float = 1e-4f,
    val warmupSteps: Int = 100,
    val weightDecay: Float = 0.1f,
    val gradClip: Float = 1.0f,
    val seed: Long = 42L,
    val evalInterval: Int = 100,
    val evalSteps: Int = 20
) {
    init {
        require(batchSize > 0) { "batchSize must be positive" }
        require(maxSteps > 0) { "maxSteps must be positive" }
        require(learningRate > 0f) { "learningRate must be positive" }
    }
}