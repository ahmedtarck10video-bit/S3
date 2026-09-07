package com.example.renderer

import android.app.Activity
import android.content.Context
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.os.Handler
import android.os.Looper
import android.util.AttributeSet
import android.util.Log
import android.view.Choreographer
import android.view.MotionEvent
import android.view.PixelCopy
import android.view.ScaleGestureDetector
import android.view.SurfaceHolder
import android.view.SurfaceView
import com.example.arcore.ArCoreSessionManager
import com.example.arcore.ArCoreTrackingData
import com.example.arcore.DepthOcclusionManager
import com.example.arcore.ExhibitMarker
import com.example.arcore.ExhibitSource
import com.example.arcore.ImageMarkerCatalog
import com.example.arcore.ImageTrackingState
import com.example.engine.DiagnosticsLogger
import com.example.engine.SensorsManager
import com.example.engine.TwoFingerRotateDetector
import com.example.model.DisplayMode
import com.example.parser.GltfAssetFactory
import com.google.ar.core.Anchor
import com.google.ar.core.AugmentedImage
import com.google.ar.core.Pose
import com.google.ar.core.TrackingState
import java.nio.ByteBuffer
import kotlin.math.abs

/**
 * High-performance Android SurfaceView bridging Google Filament and Google ARCore.
 * Supports:
 * - 60+ FPS Choreographer-driven rendering with zero allocations in doFrame.
 * - ARCore 6DoF Camera synchronization & plane anchoring.
 * - Image Target Recognition & Automatic 3D Exhibit Spawning & Anchoring.
 * - Multi-Object Scene persistence while walking in 6DoF space.
 * - Real 16-bit Depth extraction & Depth Occlusion processing.
 * - Real-world 1:1 Metric Scale (1 unit = 1 physical meter).
 * - Dual-Viewport Asymmetric Off-Axis Stereoscopic MR Pipeline.
 * - Two-finger rotation, pan, pinch zoom gestures.
 * - Hardware PixelCopy frame snapshots.
 */
class SpatialSurfaceView @JvmOverloads constructor(
  context: Context,
  attrs: AttributeSet? = null
) : SurfaceView(context, attrs), SurfaceHolder.Callback, Choreographer.FrameCallback {

  companion object {
    private const val TAG = "SpatialSurfaceView"
  }

  val filamentEngine = FilamentEngineHolder(context)
  val arCoreSessionManager = ArCoreSessionManager(context)
  val depthOcclusionManager = DepthOcclusionManager()

  var dualCameraGLSurfaceView: DualCameraGLSurfaceView? = null
    set(value) {
      field = value
      value?.arCoreSessionManager = arCoreSessionManager
      value?.depthOcclusionManager = depthOcclusionManager
      value?.displayMode = displayMode
      if (value != null && (displayMode == DisplayMode.AR || displayMode == DisplayMode.MR)) {
        val tex = value.textureId
        if (tex != 0) {
          arCoreSessionManager.setCameraTextureName(tex)
          (context as? Activity)?.let { arCoreSessionManager.resumeSession(it) }
        }
      }
    }

  private var isSurfaceReady = false
  private var isRendering = false

  // Frame timing
  private var lastFrameTimestamp = 0L
  private var frameCount = 0
  private var fpsTimer = 0L

  // Active Display Mode
  var displayMode: DisplayMode = DisplayMode.OBJECT
    set(value) {
      field = value
      updateModeConfiguration()
    }

  // Active Placed Anchors mapped to ARCore Anchor instances
  val activeArAnchors = mutableListOf<Anchor>()

  // Set of already spawned image markers to prevent duplicate spawning
  private val spawnedMarkerIds = mutableSetOf<String>()

  // Gesture Detectors
  private val scaleGestureDetector: ScaleGestureDetector
  private val rotateGestureDetector: TwoFingerRotateDetector
  private var lastTouchX = 0f
  private var lastTouchY = 0f
  private var activePointerCount = 0
  private var touchStartTime = 0L

  // Current selected model ID for manual plane tap-placement
  var currentSelectedModelId: String = "drone_v1"
  var currentSelectedModelTitle: String = "Autonomous Drone X-1"

  // User Configured Interpupillary Distance (IPD) in millimeters for Stereoscopic MR
  var userIpdMm: Float = 64.0f

  // Telemetry and Event callbacks
  var onTelemetryUpdate: ((fps: Float, drawCalls: Int, vertexCount: Int, trackingData: ArCoreTrackingData) -> Unit)? = null
  var onAnchorPlaced: ((Anchor, FloatArray, ExhibitSource, String, String) -> Unit)? = null
  var onExhibitMarkerRecognized: ((ExhibitMarker, FloatArray) -> Unit)? = null

  // Latest AR tracking data
  private var latestTrackingData = ArCoreTrackingData()

  // Preallocated per-frame scratch buffers for zero garbage collection overhead
  private val scratchProjMatrix = FloatArray(16)
  private val scratchViewMatrix = FloatArray(16)
  private val scratchHeadPoseMatrix = FloatArray(16)
  private val scratchAnchorPoses = mutableListOf<Pose>()
  private val scratchObjPos = FloatArray(3)
  private val scratchCamForward = floatArrayOf(0f, 0f, -1f)
  private var lastDepthTimeMs: Long = 0L
  private var lastLodTimeMs: Long = 0L

  // Retained head pose matrix for graceful tracking loss recovery (no black screen or sudden jump)
  private val lastValidHeadPoseMatrix = FloatArray(16).apply {
    android.opengl.Matrix.setIdentityM(this, 0)
  }
  private var hasStoredHeadPose: Boolean = false
  // Map of last valid pose per anchor hash to hold anchors in world space during tracking pause
  private val anchorLastKnownPoses = mutableMapOf<Int, Pose>()

  // Gesture state: seamless finger interaction for Rotate, Move, and Scale
  private var isOneFingerRotateMode: Boolean = false
  private var lastTapTime: Long = 0L
  private var consecutiveNullFrames: Int = 0

  private var sensorPitch = 0f
  private var sensorRoll = 0f
  private var sensorYaw = 0f

  private val sensorsManager = SensorsManager(context) { pitch, roll, yaw ->
    sensorPitch = pitch
    sensorRoll = roll
    sensorYaw = yaw
  }

  init {
    holder.addCallback(this)
    holder.setFormat(PixelFormat.TRANSLUCENT)
    setZOrderMediaOverlay(true)

    filamentEngine.initialize()

    arCoreSessionManager.onTrackingDataUpdated = { trackingData ->
      latestTrackingData = trackingData

      // Update Environmental HDR lighting in Filament
      filamentEngine.updateEnvironmentalHdrLighting(
        mainLightDir = trackingData.mainLightDirection,
        mainLightIntensityRgb = trackingData.mainLightIntensity,
        colorCorrection = trackingData.colorCorrectionRgb
      )
    }

    arCoreSessionManager.onImageTrackingStateChanged = { markerName, state, anchor, pose ->
      handleImageTrackingState(markerName, state, anchor, pose)
    }

    scaleGestureDetector = ScaleGestureDetector(
      context,
      object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(detector: ScaleGestureDetector): Boolean {
          val scaleFactor = detector.scaleFactor
          if (displayMode == DisplayMode.OBJECT) {
            filamentEngine.orbitDistance = (filamentEngine.orbitDistance / scaleFactor).coerceIn(1.0f, 6.0f)
            filamentEngine.modelScale = (filamentEngine.modelScale * scaleFactor).coerceIn(0.1f, 8.0f)
          } else {
            filamentEngine.modelScale = (filamentEngine.modelScale * scaleFactor).coerceIn(0.1f, 8.0f)
          }
          return true
        }
      }
    )

    rotateGestureDetector = TwoFingerRotateDetector { deltaDegrees ->
      if (displayMode == DisplayMode.OBJECT) {
        filamentEngine.orbitYaw -= deltaDegrees
      } else {
        filamentEngine.modelRotationDegrees -= deltaDegrees
      }
    }
  }

  /**
   * Handles ARCore AugmentedImage tracking state changes with robust lifecycle handling:
   * TRACKING -> TRACKING_LOST -> RECOVERING -> STOPPED.
   * Retains 3D model in place during temporary loss and only releases when STOPPED.
   */
  private fun handleImageTrackingState(markerId: String, state: ImageTrackingState, anchor: Anchor?, pose: Pose) {
    val marker = ImageMarkerCatalog.findByMarkerId(markerId) ?: return

    when (state) {
      ImageTrackingState.TRACKING -> {
        val existing = filamentEngine.activeExhibits.firstOrNull { it.markerId == markerId }
        if (existing == null && anchor != null) {
          activeArAnchors.add(anchor)
          val glbBuffer = GltfAssetFactory.getPresetGlbBuffer(marker.modelId)
          if (glbBuffer != null) {
            val exhibitId = "exhibit_marker_${markerId}"
            filamentEngine.spawnExhibit(
              exhibitId = exhibitId,
              modelId = marker.modelId,
              title = marker.title,
              buffer = glbBuffer,
              anchor = anchor,
              source = ExhibitSource.IMAGE_MARKER,
              markerId = markerId
            )
            val pos = floatArrayOf(pose.tx(), pose.ty(), pose.tz())
            onExhibitMarkerRecognized?.invoke(marker, pos)
            onAnchorPlaced?.invoke(anchor, pos, ExhibitSource.IMAGE_MARKER, marker.modelId, marker.title)
            DiagnosticsLogger.log(TAG, "Image Marker Tracked: '${marker.title}' -> Anchored at (${pos[0]}, ${pos[1]}, ${pos[2]})")
          }
        } else if (existing != null && anchor != null) {
          existing.anchor = anchor
        }
      }
      ImageTrackingState.TRACKING_LOST -> {
        // Retain 3D model in place at last known pose; do NOT delete immediately!
        DiagnosticsLogger.log(TAG, "Image Marker '$markerId' temporarily lost -> holding pose")
      }
      ImageTrackingState.RECOVERING -> {
        DiagnosticsLogger.log(TAG, "Image Marker '$markerId' tracking recovering")
      }
      ImageTrackingState.STOPPED -> {
        val existing = filamentEngine.activeExhibits.firstOrNull { it.markerId == markerId }
        if (existing != null) {
          filamentEngine.removeExhibit(existing.id)
          activeArAnchors.remove(existing.anchor)
          DiagnosticsLogger.log(TAG, "Image Marker '$markerId' timed out -> removed 3D Exhibit")
        }
      }
    }
  }

  fun resume(activity: Activity) {
    if (displayMode == DisplayMode.AR || displayMode == DisplayMode.MR) {
      sensorsManager.start()
      if (arCoreSessionManager.isArCorePackageInstalled()) {
        try {
          arCoreSessionManager.resumeSession(activity)
        } catch (e: Exception) {
          Log.w(TAG, "ARCore resume skipped: ${e.message}")
        }
      }
    }
    startRendering()
  }

  fun pause() {
    stopRendering()
    sensorsManager.stop()
    arCoreSessionManager.pauseSession()
  }

  fun destroy() {
    stopRendering()
    clearAnchors()
    arCoreSessionManager.destroySession()
    filamentEngine.destroy()
  }

  private fun startRendering() {
    if (!isRendering && isSurfaceReady) {
      isRendering = true
      Choreographer.getInstance().postFrameCallback(this)
    }
  }

  private fun stopRendering() {
    isRendering = false
    Choreographer.getInstance().removeFrameCallback(this)
  }

  override fun surfaceCreated(holder: SurfaceHolder) {
    try {
      isSurfaceReady = true
      filamentEngine.onSurfaceCreated(holder.surface)
      startRendering()
    } catch (e: Exception) {
      Log.e(TAG, "Error in surfaceCreated: ${e.message}", e)
    }
  }

  override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
    try {
      filamentEngine.onSurfaceResized(width, height)
      arCoreSessionManager.setDisplayGeometry(
        (context as? Activity)?.windowManager?.defaultDisplay?.rotation ?: 0,
        width,
        height
      )
    } catch (e: Exception) {
      Log.e(TAG, "Error in surfaceChanged: ${e.message}", e)
    }
  }

  override fun surfaceDestroyed(holder: SurfaceHolder) {
    isSurfaceReady = false
    stopRendering()
    filamentEngine.onSurfaceDestroyed()
  }

  private fun updateModeConfiguration() {
    filamentEngine.setDisplayMode(displayMode)
    when (displayMode) {
      DisplayMode.OBJECT -> {
        sensorsManager.stop()
        arCoreSessionManager.pauseSession()
        dualCameraGLSurfaceView?.displayMode = DisplayMode.OBJECT
      }
      DisplayMode.AR, DisplayMode.MR -> {
        sensorsManager.start()
        dualCameraGLSurfaceView?.arCoreSessionManager = arCoreSessionManager
        dualCameraGLSurfaceView?.depthOcclusionManager = depthOcclusionManager
        dualCameraGLSurfaceView?.displayMode = displayMode
        val currentTex = dualCameraGLSurfaceView?.textureId ?: 0
        if (currentTex != 0) {
          arCoreSessionManager.setCameraTextureName(currentTex)
        }
        try {
          (context as? Activity)?.let { arCoreSessionManager.resumeSession(it) }
        } catch (t: Throwable) {
          Log.w("SpatialSurfaceView", "Notice resuming ARCore session: ${t.message}")
        }
        startRendering()
      }
    }
  }

  override fun doFrame(frameTimeNanos: Long) {
    if (!isRendering || !isSurfaceReady) return

    try {
      // Calculate FPS & Telemetry once per frame
      frameCount++
      if (lastFrameTimestamp == 0L) lastFrameTimestamp = frameTimeNanos
      val elapsedMs = (frameTimeNanos - fpsTimer) / 1_000_000L
      if (elapsedMs >= 500) {
        val calculatedFps = (frameCount * 1000.0f) / elapsedMs
        frameCount = 0
        fpsTimer = frameTimeNanos
        onTelemetryUpdate?.invoke(
          calculatedFps,
          filamentEngine.drawCalls,
          filamentEngine.vertexCount,
          latestTrackingData
        )
      }

      val nowMs = frameTimeNanos / 1_000_000L

      when (displayMode) {
        DisplayMode.OBJECT -> {
          filamentEngine.updateObjectModeTransform()
          filamentEngine.updateOrbitCamera()
          filamentEngine.renderFrame(frameTimeNanos)
        }

        DisplayMode.AR -> {
          val frame = arCoreSessionManager.latestFrame
          if (frame != null && frame.camera.trackingState == TrackingState.TRACKING) {
            frame.camera.getProjectionMatrix(scratchProjMatrix, 0, 0.05f, 50.0f)
            frame.camera.getViewMatrix(scratchViewMatrix, 0)

            filamentEngine.setCameraFromArCore(scratchProjMatrix, scratchViewMatrix)

            // Extract camera forward vector from view matrix for unanchored placement
            scratchCamForward[0] = -scratchViewMatrix[2]
            scratchCamForward[1] = -scratchViewMatrix[6]
            scratchCamForward[2] = -scratchViewMatrix[10]

            // Time-based Depth Occlusion: evaluate every ~100ms (~10fps) to eliminate CPU bottleneck
            if (nowMs - lastDepthTimeMs >= 100L) {
              lastDepthTimeMs = nowMs
              scratchAnchorPoses.clear()
              for (i in 0 until activeArAnchors.size) {
                val a = activeArAnchors[i]
                if (a.trackingState == TrackingState.TRACKING) {
                  scratchAnchorPoses.add(a.pose)
                }
              }
              depthOcclusionManager.processFrameDepth(frame, scratchAnchorPoses)
              val activeDist = if (scratchAnchorPoses.isNotEmpty()) {
                val pose = scratchAnchorPoses[0]
                val camPos = latestTrackingData.cameraPosition
                val dx = pose.tx() - camPos[0]
                val dy = pose.ty() - camPos[1]
                val dz = pose.tz() - camPos[2]
                Math.sqrt((dx * dx + dy * dy + dz * dz).toDouble()).toFloat()
              } else 1.2f
              dualCameraGLSurfaceView?.virtualDepthMeters = activeDist
              filamentEngine.updateGpuDepthOcclusion(
                textureId = depthOcclusionManager.depthTextureId,
                width = depthOcclusionManager.depthWidth,
                height = depthOcclusionManager.depthHeight,
                timestampNs = depthOcclusionManager.latestDepthTimestampNs,
                minDepth = depthOcclusionManager.minDepthMeters,
                maxDepth = depthOcclusionManager.maxDepthMeters,
                avgDepth = depthOcclusionManager.averageDepthMeters,
                isReady = depthOcclusionManager.isDepthTextureReady,
                occlusionPercentage = depthOcclusionManager.occlusionPercentage,
                depthUvTransformMatrix = depthOcclusionManager.depthUvTransformMatrix,
                viewMatrix = scratchViewMatrix
              )
            }

            // Time-based Dynamic LOD evaluation: every ~150ms with zero heap allocations
            if (nowMs - lastLodTimeMs >= 150L) {
              lastLodTimeMs = nowMs
              val camPos = latestTrackingData.cameraPosition
              for (exhibit in filamentEngine.activeExhibits) {
                val anchor = exhibit.anchor
                if (anchor != null && anchor.trackingState == TrackingState.TRACKING) {
                  scratchObjPos[0] = anchor.pose.tx()
                  scratchObjPos[1] = anchor.pose.ty()
                  scratchObjPos[2] = anchor.pose.tz()
                  val radius = maxOf(exhibit.physicalWidthMeters, exhibit.physicalHeightMeters) / 2.0f
                  filamentEngine.lodManager.evaluateLod(
                    exhibitId = exhibit.id,
                    cameraPos = camPos,
                    objectPos = scratchObjPos,
                    boundingRadiusMeters = radius,
                    screenWidthPx = width,
                    screenHeightPx = height
                  )
                }
              }
            }

            // Synchronize all exhibit transforms with gesture offsets and rotations
            filamentEngine.updateAllExhibitAnchorTransforms()

            // If no multi-exhibits spawned yet, update the single selected asset
            val currentAsset = filamentEngine.currentAsset
            if (currentAsset != null && filamentEngine.activeExhibits.isEmpty()) {
              val primaryAnchor = activeArAnchors.lastOrNull()
              if (primaryAnchor != null) {
                if (primaryAnchor.trackingState == TrackingState.TRACKING) {
                  anchorLastKnownPoses[primaryAnchor.hashCode()] = primaryAnchor.pose
                  filamentEngine.updateAnchorPose(currentAsset, primaryAnchor.pose)
                } else {
                  // Hold at last valid pose during tracking pause or loss to prevent jumping
                  anchorLastKnownPoses[primaryAnchor.hashCode()]?.let { lastPose ->
                    filamentEngine.updateAnchorPose(currentAsset, lastPose)
                  }
                }
              } else {
                // Initial placement preview mode before user taps to anchor
                filamentEngine.updateUnanchoredPose(currentAsset, latestTrackingData.cameraPosition, scratchCamForward)
              }
            }
          } else {
            // Robust AR Camera with Device Orientation when AR tracking is uninitialized or lost
            filamentEngine.updateArCamera(sensorPitch, sensorYaw, sensorRoll)
            val currentAsset = filamentEngine.currentAsset
            if (currentAsset != null && filamentEngine.activeExhibits.isEmpty()) {
              val primaryAnchor = activeArAnchors.lastOrNull()
              if (primaryAnchor == null) {
                filamentEngine.updateUnanchoredPose(currentAsset, null, scratchCamForward)
              }
              // If already anchored, freeze in place (do not drag relative to phone camera!)
            }
          }
          filamentEngine.renderFrame(frameTimeNanos)
        }

        DisplayMode.MR -> {
          val frame = arCoreSessionManager.latestFrame
          val hasValidTracking = frame != null && frame.camera.trackingState == TrackingState.TRACKING
          if (hasValidTracking && frame != null) {
            frame.camera.getViewMatrix(scratchViewMatrix, 0)
            if (android.opengl.Matrix.invertM(scratchHeadPoseMatrix, 0, scratchViewMatrix, 0)) {
              System.arraycopy(scratchHeadPoseMatrix, 0, lastValidHeadPoseMatrix, 0, 16)
              hasStoredHeadPose = true
            }
            scratchCamForward[0] = -scratchViewMatrix[2]
            scratchCamForward[1] = -scratchViewMatrix[6]
            scratchCamForward[2] = -scratchViewMatrix[10]
          } else if (hasStoredHeadPose) {
            // Decouple camera stream from tracking: retain last valid head pose during tracking loss
            System.arraycopy(lastValidHeadPoseMatrix, 0, scratchHeadPoseMatrix, 0, 16)
          }

          // Process Depth in MR on time-based interval (~100ms)
          if (frame != null && nowMs - lastDepthTimeMs >= 100L) {
            lastDepthTimeMs = nowMs
            scratchAnchorPoses.clear()
            for (i in 0 until activeArAnchors.size) {
              val a = activeArAnchors[i]
              if (a.trackingState == TrackingState.TRACKING) {
                anchorLastKnownPoses[a.hashCode()] = a.pose
                scratchAnchorPoses.add(a.pose)
              } else {
                anchorLastKnownPoses[a.hashCode()]?.let { scratchAnchorPoses.add(it) }
              }
            }
            depthOcclusionManager.processFrameDepth(frame, scratchAnchorPoses)
            filamentEngine.updateGpuDepthOcclusion(
              textureId = depthOcclusionManager.depthTextureId,
              width = depthOcclusionManager.depthWidth,
              height = depthOcclusionManager.depthHeight,
              timestampNs = depthOcclusionManager.latestDepthTimestampNs,
              minDepth = depthOcclusionManager.minDepthMeters,
              maxDepth = depthOcclusionManager.maxDepthMeters,
              avgDepth = depthOcclusionManager.averageDepthMeters,
              isReady = depthOcclusionManager.isDepthTextureReady,
              occlusionPercentage = depthOcclusionManager.occlusionPercentage,
              depthUvTransformMatrix = depthOcclusionManager.depthUvTransformMatrix,
              viewMatrix = scratchViewMatrix
            )
          }

          // Synchronize all exhibit transforms with finger gestures (rotation, scale, position)
          filamentEngine.updateAllExhibitAnchorTransforms()

          // If no multi-exhibits spawned yet, update single selected asset
          val currentAsset = filamentEngine.currentAsset
          if (currentAsset != null && filamentEngine.activeExhibits.isEmpty()) {
            val primaryAnchor = activeArAnchors.lastOrNull()
            if (primaryAnchor != null) {
              if (primaryAnchor.trackingState == TrackingState.TRACKING) {
                anchorLastKnownPoses[primaryAnchor.hashCode()] = primaryAnchor.pose
                filamentEngine.updateAnchorPose(currentAsset, primaryAnchor.pose)
              } else {
                // Hold at last valid pose during tracking pause or loss
                anchorLastKnownPoses[primaryAnchor.hashCode()]?.let { lastPose ->
                  filamentEngine.updateAnchorPose(currentAsset, lastPose)
                }
              }
            } else {
              filamentEngine.updateUnanchoredPose(currentAsset, latestTrackingData.cameraPosition, scratchCamForward)
            }
          }

          // Use user-calibrated IPD (e.g. 52mm - 74mm range)
          val ipdMeters = (userIpdMm / 1000.0f).coerceIn(0.045f, 0.085f)
          filamentEngine.renderStereoFrame(
            frameTimeNanos,
            ipdMeters,
            if (hasValidTracking || hasStoredHeadPose) scratchHeadPoseMatrix else null
          )
        }
      }
    } catch (e: Exception) {
      Log.e(TAG, "Error in doFrame: ${e.message}", e)
    } finally {
      if (isRendering) {
        Choreographer.getInstance().postFrameCallback(this)
      }
    }
  }

  override fun onTouchEvent(event: MotionEvent): Boolean {
    val isRotating = rotateGestureDetector.onTouchEvent(event)
    scaleGestureDetector.onTouchEvent(event)

    when (event.actionMasked) {
      MotionEvent.ACTION_DOWN -> {
        lastTouchX = event.x
        lastTouchY = event.y
        activePointerCount = 1
        touchStartTime = System.currentTimeMillis()
        return true
      }

      MotionEvent.ACTION_POINTER_DOWN -> {
        activePointerCount = event.pointerCount
        return true
      }

      MotionEvent.ACTION_MOVE -> {
        if (!scaleGestureDetector.isInProgress && !isRotating) {
          val dx = event.x - lastTouchX
          val dy = event.y - lastTouchY

          if (event.pointerCount == 1) {
            if (displayMode == DisplayMode.OBJECT) {
              filamentEngine.orbitYaw -= dx * 0.45f
              filamentEngine.orbitPitch = (filamentEngine.orbitPitch - dy * 0.45f).coerceIn(-80f, 80f)
            } else {
              // AR & MR: When model is placed/anchored, 1-finger gesture smoothly rotates in place
              if (activeArAnchors.isNotEmpty()) {
                filamentEngine.modelRotationDegrees -= dx * 0.45f
              } else {
                if (isOneFingerRotateMode || lastTouchY > height * 0.72f) {
                  filamentEngine.modelRotationDegrees -= dx * 0.45f
                } else {
                  filamentEngine.modelOffsetX += dx * 0.0025f
                  filamentEngine.modelOffsetY -= dy * 0.0025f
                }
              }
            }
          } else if (event.pointerCount == 2) {
            if (displayMode == DisplayMode.OBJECT) {
              filamentEngine.panX = (filamentEngine.panX + dx * 0.003f).coerceIn(-0.35f, 0.35f)
              filamentEngine.panY = (filamentEngine.panY - dy * 0.003f).coerceIn(-0.25f, 0.25f)
            } else {
              // AR & MR: 2-finger horizontal drag rotates
              filamentEngine.modelRotationDegrees -= dx * 0.45f
            }
          }
        }
        lastTouchX = event.x
        lastTouchY = event.y
        return true
      }

      MotionEvent.ACTION_UP -> {
        val duration = System.currentTimeMillis() - touchStartTime
        val movedDist = abs(event.x - lastTouchX) + abs(event.y - lastTouchY)
        if (duration < 300 && movedDist < 20) {
          val now = System.currentTimeMillis()
          if (now - lastTapTime < 350) {
            // Double-tap toggles between Move (Right/Left & Up/Down) and Rotate (360° spin)
            isOneFingerRotateMode = !isOneFingerRotateMode
            lastTapTime = 0L
          } else {
            lastTapTime = now
            handleTap(event.x, event.y)
          }
        }
        activePointerCount = 0
        return true
      }

      MotionEvent.ACTION_POINTER_UP -> {
        activePointerCount = event.pointerCount - 1
        return true
      }
    }
    return super.onTouchEvent(event)
  }

  private fun handleTap(xPx: Float, yPx: Float) {
    if (displayMode == DisplayMode.AR || displayMode == DisplayMode.MR) {
      try {
        val mappedX = if (displayMode == DisplayMode.MR && width > 0) {
          val halfWidth = width / 2f
          if (xPx > halfWidth) (xPx - halfWidth) * 2f else xPx * 2f
        } else {
          xPx
        }
        val frame = arCoreSessionManager.latestFrame ?: return
        if (frame.camera.trackingState != TrackingState.TRACKING) {
          DiagnosticsLogger.log(TAG, "Placement deferred: Camera tracking not yet stable (${frame.camera.trackingState})")
          return
        }
        val hit = arCoreSessionManager.hitTest(frame, mappedX, yPx)
        if (hit != null) {
          val hitPose = hit.hitPose
          val hx = hitPose.tx()
          val hy = hitPose.ty()
          val hz = hitPose.tz()

          // Conflict Resolution: Check if tap is right on top of an existing Image Marker exhibit (< 0.35m)
          // Priority: Image Marker > Plane Anchor
          for (exhibit in filamentEngine.activeExhibits) {
            val exAnchor = exhibit.anchor
            if (exAnchor != null) {
              val dx = exAnchor.pose.tx() - hx
              val dy = exAnchor.pose.ty() - hy
              val dz = exAnchor.pose.tz() - hz
              val dist = kotlin.math.sqrt(dx * dx + dy * dy + dz * dz)
              if (dist < 0.35f) {
                DiagnosticsLogger.log(TAG, "Prevented duplicate exhibit: Tap is within ${dist}m of existing '${exhibit.title}'")
                return
              }
            }
          }

          val anchor = arCoreSessionManager.createAnchor(hit)
          if (anchor != null) {
            for (oldAnchor in activeArAnchors) {
              try { oldAnchor.detach() } catch (_: Exception) {}
            }
            activeArAnchors.clear()
            activeArAnchors.add(anchor)
            val posArr = floatArrayOf(hx, hy, hz)

            filamentEngine.clearAllExhibits()
            val currentAsset = filamentEngine.currentAsset
            if (currentAsset != null) {
              filamentEngine.modelOffsetX = 0f
              filamentEngine.modelOffsetY = 0f
              filamentEngine.modelOffsetZ = 0f
              filamentEngine.updateAnchorPose(currentAsset, anchor.pose)
            }

            onAnchorPlaced?.invoke(anchor, posArr, ExhibitSource.PLANE_TAP, currentSelectedModelId, currentSelectedModelTitle)
            Log.i(TAG, "ARCore Anchor pinned on plane at: $hx, $hy, $hz")
            DiagnosticsLogger.log(TAG, "Placed Anchor at ($hx, $hy, $hz)")
          }
        }
      } catch (e: Exception) {
        Log.w(TAG, "Error during AR tap hit test: ${e.message}")
      }
    }
  }

  fun captureSnapshot(onCaptured: (Bitmap) -> Unit, onError: (String) -> Unit) {
    if (!isSurfaceReady || width <= 0 || height <= 0) {
      onError("Surface not ready for snapshot")
      return
    }

    try {
      val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
      PixelCopy.request(
        this,
        bitmap,
        { copyResult ->
          if (copyResult == PixelCopy.SUCCESS) {
            onCaptured(bitmap)
          } else {
            onError("PixelCopy failed with status: $copyResult")
          }
        },
        Handler(Looper.getMainLooper())
      )
    } catch (e: Throwable) {
      Log.e(TAG, "Error capturing snapshot: ${e.message}", e)
      onError("Snapshot capture failed: ${e.message}")
    }
  }

  fun loadGlbBuffer(buffer: ByteBuffer, title: String) {
    filamentEngine.loadAsset(buffer, title)
  }

  fun clearAnchors() {
    for (anchor in activeArAnchors) {
      anchor.detach()
    }
    activeArAnchors.clear()
    spawnedMarkerIds.clear()
    filamentEngine.clearAllExhibits()
  }

  fun clearModelAndScene() {
    filamentEngine.clearAll()
    filamentEngine.clearGpuDepthAndTrackingResources()
    depthOcclusionManager.clear()
    arCoreSessionManager.environmentalMeshManager.clear()
    arCoreSessionManager.cloudAnchorManager.clearAll()
    for (anchor in activeArAnchors) {
      anchor.detach()
    }
    activeArAnchors.clear()
    spawnedMarkerIds.clear()
    anchorLastKnownPoses.clear()
    hasStoredHeadPose = false
    currentSelectedModelId = ""
    currentSelectedModelTitle = ""
    arCoreSessionManager.handleTrackingLostOrReset(resetSession = false)
    arCoreSessionManager.resetWalkingOrigin()
  }

  fun resetView() {
    filamentEngine.resetTransforms()
    filamentEngine.clearGpuDepthAndTrackingResources()
    depthOcclusionManager.clear()
    arCoreSessionManager.environmentalMeshManager.clear()
    clearAnchors()
    anchorLastKnownPoses.clear()
    hasStoredHeadPose = false
    arCoreSessionManager.handleTrackingLostOrReset(resetSession = false)
    arCoreSessionManager.resetWalkingOrigin()
  }
}
