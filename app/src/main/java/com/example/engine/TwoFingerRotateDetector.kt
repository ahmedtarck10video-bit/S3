package com.example.engine

import android.view.MotionEvent
import kotlin.math.atan2

/**
 * Custom Two-Finger Twist / Rotation gesture detector.
 * Computes angular rotation delta between two touch pointers in degrees.
 */
class TwoFingerRotateDetector(
  private val onRotateListener: (deltaDegrees: Float) -> Unit
) {

  private var initialAngle: Float = 0f
  private var isRotating: Boolean = false

  fun onTouchEvent(event: MotionEvent): Boolean {
    when (event.actionMasked) {
      MotionEvent.ACTION_POINTER_DOWN -> {
        if (event.pointerCount == 2) {
          initialAngle = calculateAngle(event)
          isRotating = true
        }
      }
      MotionEvent.ACTION_MOVE -> {
        if (isRotating && event.pointerCount >= 2) {
          val currentAngle = calculateAngle(event)
          var delta = currentAngle - initialAngle

          // Normalize delta to [-180, 180]
          if (delta > 180f) delta -= 360f
          if (delta < -180f) delta += 360f

          if (Math.abs(delta) > 0.5f) {
            onRotateListener(delta)
            initialAngle = currentAngle
          }
        }
      }
      MotionEvent.ACTION_POINTER_UP -> {
        if (event.pointerCount <= 2) {
          isRotating = false
        }
      }
      MotionEvent.ACTION_CANCEL, MotionEvent.ACTION_UP -> {
        isRotating = false
      }
    }
    return isRotating
  }

  private fun calculateAngle(event: MotionEvent): Float {
    if (event.pointerCount < 2) return 0f
    val dx = event.getX(1) - event.getX(0)
    val dy = event.getY(1) - event.getY(0)
    val radians = atan2(dy.toDouble(), dx.toDouble())
    return Math.toDegrees(radians).toFloat()
  }
}
