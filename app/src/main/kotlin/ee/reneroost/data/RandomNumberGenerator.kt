package ee.reneroost.data

// uses Xorshift128+ method
class RandomNumberGenerator(seed: Long) {
    private var s0: Long = seed
    private var s1: Long = seed xor 0x6A09E667F3BCC908L

    fun nextLong(): Long {
        var x = s0
        val y = s1
        s0 = y
        x = x xor (x shl 23)
        s1 = x xor y xor (x ushr 17) xor (y ushr 26)
        return s1 + y
    }

    fun nextFloat(): Float = (nextLong() ushr 40).toFloat() / (1 shl 24).toFloat()

    fun nextInt(bound: Int): Int {
        require(bound > 0) { "bound must be positive" }
        return (nextFloat() * bound).toInt().coerceIn(0, bound - 1)
    }
}