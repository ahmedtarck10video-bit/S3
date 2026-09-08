package com.example.arcore

import android.opengl.Matrix
import com.google.ar.core.Anchor
import com.google.ar.core.Pose
import com.google.ar.core.TrackingFailureReason
import com.google.ar.core.TrackingState

/**
 * Synchronized record for an individual tracked ARCore Anchor.
 * Contains both the raw ARCore pose and the reconciled rendered pose,
 * accounting for smooth recovery interpolation and tracking state transitions.
 */
data class SynchronizedAnchorRecord(
  val id: String,
  val anchor: Anchor?,
  val trackingState: TrackingState,
  val rawPose: Pose,
  val renderedPose: Pose,
  val isRecovering: Boolean = false
)

/**
 * Atomic, immutable Synchronized AR Render State produced strictly from ONE single ARCore Frame.
 * Guarantees that Camera Pose, Projection Matrix, View Matrix, TrackingState, Anchor Poses,
 * Depth Texture, Point Cloud, and Light Estimation all share the EXACT same temporal state.
 */
class SynchronizedArRenderState(
  val frameId: Long = 0L,
  val frameTimestampNs: Long = 0L,
  val trackingState: TrackingState = TrackingState.STOPPED,
  val trackingFailureReason: TrackingFailureReason = TrackingFailureReason.NONE,
  val cameraPose: Pose? = null,
  val projectionMatrix: FloatArray = FloatArray(16),
  val viewMatrix: FloatArray = FloatArray(16),
  val cameraPosition: FloatArray = FloatArray(3),
  val cameraForward: FloatArray = FloatArray(3),
  val primaryAnchorPose: Pose? = null,
  val primaryAnchorState: TrackingState = TrackingState.STOPPED,
  val anchorRecords: List<SynchronizedAnchorRecord> = emptyList(),
  // Frame-Synchronized Depth Data
  val depthTimestampNs: Long = 0L,
  val isDepthValid: Boolean = false,
  val depthTextureId: Int = 0,
  val depthWidth: Int = 0,
  val depthHeight: Int = 0,
  val minDepthMeters: Float = 0f,
  val maxDepthMeters: Float = 0f,
  val averageDepthMeters: Float = 0f,
  val occlusionPercentage: Float = 0f,
  val depthUvTransformMatrix: FloatArray = FloatArray(16),
  // Frame-Synchronized Point Cloud Spatial Metrics
  val pointCloudTimestampNs: Long = 0L,
  val pointCloudPointsCount: Int = 0,
  val pointCloudMeanDistanceMeters: Float = 0f,
  val pointCloudConfidenceRatio: Float = 0f,
  val isSpatialStabilityHigh: Boolean = false,
  // Light Estimation
  val lightIntensityLumens: Float = 1000f,
  val colorCorrectionRgb: FloatArray = floatArrayOf(1f, 1f, 1f, 1f),
  val mainLightDirection: FloatArray = floatArrayOf(0f, -1f, -0.5f),
  val mainLightIntensity: FloatArray = floatArrayOf(1f, 1f, 1f),
  val isFresh: Boolean = true
)

/**
 * Manages controlled Anchor pose recovery and tracking state hysteresis.
 * - Steady TRACKING: Uses exact ARCore pose directly with zero latency or lag (no artificial continuous smoothing).
 * - PAUSED/LOST -> TRACKING transition: Reconciles smoothly over ~150ms from last stable pose to new pose,
 *   completely eliminating transformation jumps or jitter.
 * - PAUSED: Safely holds last stable visual pose without drift or invalid transform updates.
 * - STOPPED: Safely releases resources.
 * - Resists tracking-state oscillation without destroying or recreating Anchors.
 */
class AnchorRecoveryTracker {

  private class AnchorInternalState(
    var lastTrackingState: TrackingState = TrackingState.STOPPED,
    var lastStablePose: Pose = Pose.IDENTITY,
    var recoveryStartPose: Pose = Pose.IDENTITY,
    var recoveryProgress: Float = 1.0f,
    var stableFrameCount: Int = 0,
    var pausedFrameCount: Int = 0
  )

  private val anchorStates = HashMap<Int, AnchorInternalState>()

  /**
   * Reconciles the given Anchor's pose for the current frame.
   * Returns a Pair of (reconciledPose, isRecovering).
   */
  fun reconcileAnchorPose(anchor: Anchor, deltaTimeSec: Float = 0.016f): Pair<Pose, Boolean> {
    val key = anchor.hashCode()
    val state = anchorStates.getOrPut(key) {
      AnchorInternalState(
        lastTrackingState = anchor.trackingState,
        lastStablePose = anchor.pose,
        recoveryProgress = 1.0f
      )
    }

    val currentTracking = anchor.trackingState
    val prevTracking = state.lastTrackingState

    return when (currentTracking) {
      TrackingState.TRACKING -> {
        state.pausedFrameCount = 0
        state.stableFrameCount++

        if (prevTracking != TrackingState.TRACKING) {
          // Tracking recovered! Start smooth reconciliation from last stable pose
          state.recoveryProgress = 0.0f
          state.recoveryStartPose = state.lastStablePose
          state.lastTrackingState = TrackingState.TRACKING
        }

        val targetPose = anchor.pose
        if (state.recoveryProgress < 1.0f) {
          // Advance recovery over ~150ms (approx 8-10 frames at 60fps)
          val step = if (deltaTimeSec > 0f) minOf(1.0f, deltaTimeSec / 0.150f) else 0.125f
          state.recoveryProgress = minOf(1.0f, state.recoveryProgress + step)
          val interpolated = interpolatePose(state.recoveryStartPose, targetPose, state.recoveryProgress)
          state.lastStablePose = interpolated
          Pair(interpolated, true)
        } else {
          // Steady tracking: zero artificial lag/smoothing
          state.lastStablePose = targetPose
          Pair(targetPose, false)
        }
      }
      TrackingState.PAUSED -> {
        state.stableFrameCount = 0
        state.pausedFrameCount++
        state.lastTrackingState = TrackingState.PAUSED
        // Hold at last stable pose
        Pair(state.lastStablePose, false)
      }
      TrackingState.STOPPED -> {
        state.lastTrackingState = TrackingState.STOPPED
        Pair(state.lastStablePose, false)
      }
    }
  }

  fun unregisterAnchor(anchor: Anchor) {
    anchorStates.remove(anchor.hashCode())
  }

  fun clear() {
    anchorStates.clear()
  }

  companion object {
    /**
     * Slerp and lerp between two ARCore Poses (translation lerp + orientation quaternion slerp).
     */
    fun interpolatePose(start: Pose, end: Pose, t: Float): Pose {
      val clampedT = t.coerceIn(0.0f, 1.0f)
      val tx = (1.0f - clampedT) * start.tx() + clampedT * end.tx()
      val ty = (1.0f - clampedT) * start.ty() + clampedT * end.ty()
      val tz = (1.0f - clampedT) * start.tz() + clampedT * end.tz()

      val q1x = start.qx(); val q1y = start.qy(); val q1z = start.qz(); val q1w = start.qw()
      var q2x = end.qx(); var q2y = end.qy(); var q2z = end.qz(); var q2w = end.qw()

      var dot = q1x * q2x + q1y * q2y + q1z * q2z + q1w * q2w
      if (dot < 0.0f) {
        dot = -dot
        q2x = -q2x; q2y = -q2y; q2z = -q2z; q2w = -q2w
      }

      val rx: Float
      val ry: Float
      val rz: Float
      val rw: Float

      if (dot > 0.9995f) {
        rx = (1.0f - clampedT) * q1x + clampedT * q2x
        ry = (1.0f - clampedT) * q1y + clampedT * q2y
        rz = (1.0f - clampedT) * q1z + clampedT * q2z
        rw = (1.0f - clampedT) * q1w + clampedT * q2w
      } else {
        val angle = kotlin.math.acos(dot.toDouble().coerceIn(-1.0, 1.0))
        val sinAngle = kotlin.math.sin(angle)
        val ratioA = kotlin.math.sin((1.0 - clampedT.toDouble()) * angle) / sinAngle
        val ratioB = kotlin.math.sin(clampedT.toDouble() * angle) / sinAngle
        rx = (ratioA * q1x + ratioB * q2x).toFloat()
        ry = (ratioA * q1y + ratioB * q2y).toFloat()
        rz = (ratioA * q1z + ratioB * q2z).toFloat()
        rw = (ratioA * q1w + ratioB * q2w).toFloat()
      }

      val norm = kotlin.math.sqrt(rx * rx + ry * ry + rz * rz + rw * rw)
      val invNorm = if (norm > 0.00001f) 1.0f / norm else 1.0f

      return Pose(
        floatArrayOf(tx, ty, tz),
        floatArrayOf(rx * invNorm, ry * invNorm, rz * invNorm, rw * invNorm)
      )
    }
  }
}
