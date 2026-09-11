package com.example.renderer

import android.content.Context
import android.view.MotionEvent
import android.widget.FrameLayout
import androidx.lifecycle.LifecycleOwner
import com.example.model.DisplayMode
import dev.romainguy.kotlin.math.Float3
import io.github.sceneview.SceneView
import io.github.sceneview.ar.ARSceneView
import io.github.sceneview.ar.node.AnchorNode
import io.github.sceneview.node.ModelNode
import java.nio.ByteBuffer

/**
 * SceneViewContainer manages SceneView and ARSceneView instances as Gradle dependencies
 * to support loading GLB models, manipulation, animation, and AR tracking/placement.
 */
class SceneViewContainer(context: Context) : FrameLayout(context) {

  var sceneView: SceneView? = null
    private set

  var arSceneView: ARSceneView? = null
    private set

  private var activeModelNode: ModelNode? = null
  private var currentGlbBuffer: ByteBuffer? = null
  private var currentModelTitle: String = "SceneViewModel"

  var displayMode: DisplayMode = DisplayMode.OBJECT
    set(value) {
      if (field != value) {
        field = value
        updateActiveView()
      }
    }

  var onModelPlaced: ((anchorPos: Float3) -> Unit)? = null

  init {
    updateActiveView()
  }

  private fun updateActiveView() {
    removeAllViews()
    activeModelNode = null

    if (displayMode == DisplayMode.OBJECT) {
      // 3D Object Mode using standard SceneView
      arSceneView?.let {
        it.destroy()
        arSceneView = null
      }

      val sv = SceneView(context).apply {
        layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT)
      }
      sceneView = sv
      addView(sv)

      // Reload active model if present
      currentGlbBuffer?.let { buf ->
        loadGlb(buf, currentModelTitle)
      }
    } else {
      // AR / MR Mode using ARSceneView
      sceneView?.let {
        it.destroy()
        sceneView = null
      }

      val asv = ARSceneView(context).apply {
        layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT)
        planeRenderer.isVisible = true

        // On tap gesture on AR scene, perform hit test and place anchor
        onTouchEvent = { motionEvent, _ ->
          if (motionEvent.action == MotionEvent.ACTION_UP) {
            val arHitResult = hitTestAR(motionEvent.x, motionEvent.y)
            if (arHitResult != null) {
              val anchor = arHitResult.createAnchor()
              val anchorNode = AnchorNode(engine, anchor)

              val currentModel = activeModelNode
              if (currentModel != null) {
                currentModel.parent?.removeChildNode(currentModel)
                anchorNode.addChildNode(currentModel)
              }
              addChildNode(anchorNode)
              onModelPlaced?.invoke(anchorNode.worldPosition)
              true
            } else {
              false
            }
          } else {
            false
          }
        }
      }
      arSceneView = asv
      addView(asv)

      // Reload active model if present
      currentGlbBuffer?.let { buf ->
        loadGlb(buf, currentModelTitle)
      }
    }
  }

  fun loadGlb(buffer: ByteBuffer, title: String = "Model") {
    currentGlbBuffer = buffer
    currentModelTitle = title

    val dupBuffer = buffer.duplicate()
    dupBuffer.position(0)

    val currentView = if (displayMode == DisplayMode.OBJECT) sceneView else arSceneView ?: return
    val modelLoader = currentView?.modelLoader ?: return

    try {
      val modelInstance = modelLoader.createModelInstance(dupBuffer)
      if (modelInstance != null) {
        val modelNode = ModelNode(
          modelInstance = modelInstance,
          scaleToUnits = 1.0f,
          centerOrigin = Float3(0f, 0f, 0f)
        ).apply {
          isPositionEditable = true
          isRotationEditable = true
          isScaleEditable = true
        }

        activeModelNode?.let { old ->
          currentView.removeChildNode(old)
          old.destroy()
        }

        activeModelNode = modelNode
        currentView.addChildNode(modelNode)
      }
    } catch (e: Exception) {
      e.printStackTrace()
    }
  }

  fun setAnimationPlaying(playing: Boolean) {
    activeModelNode?.let { node ->
      if (playing) {
        node.playAnimation(0)
      } else {
        node.stopAnimation(0)
      }
    }
  }

  fun setAnimationSpeed(speed: Float) {
    activeModelNode?.setAnimationSpeed(0, speed)
  }

  fun resume(lifecycleOwner: LifecycleOwner? = null) {
    // SceneView automatically tracks lifecycle or can be resumed
  }

  fun pause() {
    // SceneView pauses when detached or lifecycle paused
  }

  fun destroy() {
    activeModelNode?.destroy()
    activeModelNode = null
    sceneView?.destroy()
    sceneView = null
    arSceneView?.destroy()
    arSceneView = null
    removeAllViews()
  }
}
