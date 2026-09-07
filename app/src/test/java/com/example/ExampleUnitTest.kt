package com.example

import org.junit.Assert.*
import org.junit.Test

class ExampleUnitTest {
  @Test
  fun addition_isCorrect() {
    assertEquals(4, 2 + 2)
  }

  @Test
  fun testGestureScaleConstraints() {
    val initialScale = 1.0f
    val scaleFactorMagnify = 1.5f
    val scaleFactorMinify = 0.5f

    val magnified = (initialScale * scaleFactorMagnify).coerceIn(0.02f, 25.0f)
    assertEquals(1.5f, magnified, 0.001f)

    val minified = (initialScale * scaleFactorMinify).coerceIn(0.02f, 25.0f)
    assertEquals(0.5f, minified, 0.001f)

    // Verify boundary constraints
    val extremeMin = (0.001f).coerceIn(0.02f, 25.0f)
    assertEquals(0.02f, extremeMin, 0.001f)

    val extremeMax = (50.0f).coerceIn(0.02f, 25.0f)
    assertEquals(25.0f, extremeMax, 0.001f)
  }

  @Test
  fun testGestureRotationAndTranslationAccumulation() {
    var rotation = 0f
    var offsetX = 0f
    var offsetY = 0f

    // 1-finger horizontal swipe moves Right/Left
    val dx = 50f
    offsetX += dx * 0.0025f
    assertEquals(0.125f, offsetX, 0.0001f)

    // 1-finger vertical swipe moves Up/Down
    val dy = -40f
    offsetY -= dy * 0.0025f
    assertEquals(0.1f, offsetY, 0.0001f)

    // 2-finger horizontal drag or rotate gesture rotates Yaw
    val deltaYaw = 45f
    rotation += deltaYaw
    assertEquals(45f, rotation, 0.0001f)
  }
}
