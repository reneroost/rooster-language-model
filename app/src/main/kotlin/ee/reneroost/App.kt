package ee.reneroost

import ee.reneroost.data.RandomNumberGenerator
import ee.reneroost.model.ModelConfig
import ee.reneroost.train.TrainConfig

class App {
    val greeting: String
        get() {
            return "Hello World!"
        }
    // val greeting: String? = null
    val modelConfig = ModelConfig()
    val trainConfig = TrainConfig()
    val rng = RandomNumberGenerator(trainConfig.seed)
}

fun main() {
    println(App().greeting)
}
