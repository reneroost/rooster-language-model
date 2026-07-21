# rooster-language-model 🐓

> **A Claude Fable 6 open source clone**, built from scratch in pure Kotlin on the JVM.

A minimal, zero-dependency character-level Transformer language model implementation designed to demonstrate end-to-end neural network architecture without heavy Python ML frameworks (PyTorch, TensorFlow).

---

## 🚀 Roadmap

- [x] **Step 0:** Toolchain & Skeleton (JDK 25, Gradle Kotlin DSL, `kotlin.test`)
- [ ] **Step 1:** Data Pipeline & Vocab Encoding (Tiny Shakespeare)
- [ ] **Step 2:** Bigram Language Model
- [ ] **Step 3:** Multi-Layer Perceptron (MLP)
- [ ] **Step 4:** Single-Head & Multi-Head Self-Attention
- [ ] **Step 5:** Stacked Transformer Blocks
- [ ] **Phase 2:** LWJGL Vulkan GPU acceleration & matrix multiplication kernels

---

## 🛠️ Requirements

- **JDK 25** (Managed via system APT)
- **Gradle Wrapper** (`./gradlew`)

---

## 🧪 Quickstart

```bash
# Run unit test suite
./gradlew test

# Run application
./gradlew -q run