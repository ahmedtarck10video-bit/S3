package com.example.renderer

import android.content.Context
import android.opengl.Matrix
import android.util.Log
import android.view.Surface
import com.example.arcore.ExhibitSource
import com.example.engine.DiagnosticsLogger
import com.example.engine.HardwareCapabilities
import com.example.engine.HardwareCapabilityDetector
import com.example.engine.RenderQualityProfile
import com.example.engine.SpatialLodManager
import com.example.model.DisplayMode
import com.google.android.filament.Camera
import com.google.android.filament.Engine
import com.google.android.filament.EntityManager
import com.google.android.filament.Filament
import com.google.android.filament.IndirectLight
import com.google.android.filament.LightManager
import com.google.android.filament.MaterialInstance
import com.google.android.filament.Renderer
import com.google.android.filament.Scene
import com.google.android.filament.SwapChain
import com.google.android.filament.Texture
import com.google.android.filament.TextureSampler
import com.google.android.filament.View
import com.google.android.filament.Viewport
import com.google.android.filament.gltfio.AssetLoader
import com.google.android.filament.gltfio.FilamentAsset
import com.google.android.filament.gltfio.FilamentInstance
import com.google.android.filament.gltfio.Gltfio
import com.google.android.filament.gltfio.MaterialProvider
import com.google.android.filament.gltfio.ResourceLoader
import com.google.android.filament.gltfio.UbershaderProvider
import com.google.ar.core.Anchor
import com.google.ar.core.Pose
import com.google.ar.core.TrackingState
import java.nio.ByteBuffer
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.sin
import kotlin.math.tan

/**
 * Clean architectural abstraction for Filament Material depth comparison functions.
 * Decouples the material rasterizer depth test from the TextureSampler compare mode
 * to avoid conflating sampler sampling states with pipeline rasterization states.
 */
enum class FilamentMaterialDepthFunc(val filamentCompareFunc: TextureSampler.CompareFunction) {
  LESS(TextureSampler.CompareFunction.LESS),
  LESS_EQUAL(TextureSampler.CompareFunction.LESS_EQUAL),
  GREATER(TextureSampler.CompareFunction.GREATER),
  GREATER_EQUAL(TextureSampler.CompareFunction.GREATER_EQUAL),
  EQUAL(TextureSampler.CompareFunction.EQUAL),
  NOT_EQUAL(TextureSampler.CompareFunction.NOT_EQUAL),
  ALWAYS(TextureSampler.CompareFunction.ALWAYS),
  NEVER(TextureSampler.CompareFunction.NEVER);

  companion object {
    /**
     * Filament utilizes a reversed-Z depth buffer (near plane = 1.0, far plane = 0.0).
     * Therefore, GREATER_EQUAL is the standard pass condition for geometry in front of the far plane.
     */
    val REVERSED_Z_DEFAULT = GREATER_EQUAL
  }
}

/**
 * Extension method to apply the material depth function cleanly without mixing with TextureSampler logic.
 */
fun MaterialInstance.setMaterialDepthFunc(func: FilamentMaterialDepthFunc) {
  this.setDepthFunc(func.filamentCompareFunc)
}

/**
 * Representation of a 3D model instantiated inside the Filament 3D Scene,
 * linked to a real ARCore 6DoF Anchor (from physical Plane or Image Marker).
 */
data class ActiveSceneExhibit(
  val id: String,
  val modelId: String,
  val title: String,
  val asset: FilamentAsset,
  val source: ExhibitSource,
  val markerId: String? = null,
  var anchor: Anchor? = null,
  var customScale: Float = 1.0f,
  var customRotationDeg: Float = 0.0f,
  val physicalWidthMeters: Float = 1.0f,
  val physicalHeightMeters: Float = 1.0f,
  val physicalDepthMeters: Float = 1.0f,
  val centerOffsetX: Float = 0f,
  val centerOffsetY: Float = 0f,
  val centerOffsetZ: Float = 0f,
  val physicalHalfHeight: Float = 0f
)

/**
 * Production-Grade Google Filament 3D & gltfio Engine Architecture.
 * Features:
 * - Unified 3D Asset Model pipeline across Object Mode, AR Mode, and MR Mode.
 * - Multi-Object Scene management with independent 6DoF Anchor transforms.
 * - Real-world 1:1 Metric Scale (1 unit = 1 physical meter).
 * - Augmented Image Marker & Plane tracking binding.
 * - High-precision Asymmetric Off-Axis Stereoscopic MR Projection per eye.
 * - Zero-allocation per-frame render loop (preallocated scratch buffers).
 * - Multi-track Skeletal & Morph Animation engine with scrubbing and reset.
 * - Dynamic Thermal MSAA & FXAA quality adaptation.
 * - ARCore Environmental HDR lighting integration with exponential moving average (EMA) smoothing.
 */
class FilamentEngineHolder(private val context: Context) {

  companion object {
    private const val TAG = "FilamentEngineHolder"

    init {
      try {
        Gltfio.init()
        Filament.init()
      } catch (t: Throwable) {
        Log.w(TAG, "Native Filament libraries not loaded (unit test or unsupported env): ${t.message}")
      }
    }

    /**
     * Pure calculation of miniature display scale factor from real-world metric dimensions.
     * Keeps large models (e.g. 15m x 5m museum) completely visible and comfortable to inspect
     * in the Object Mode inspection viewport (~0.85m), while preserving source asset dimensions.
     */
    fun calculateMiniatureScale(widthMeters: Float, heightMeters: Float, depthMeters: Float): Float {
      val maxDim = maxOf(widthMeters, heightMeters, depthMeters)
      if (maxDim <= 0.001f) return 1.0f
      val targetMiniatureDim = 0.85f
      return if (maxDim > 1.2f) {
        (targetMiniatureDim / maxDim).coerceIn(0.005f, 1.0f)
      } else if (maxDim in 0.001f..0.25f) {
        (0.6f / maxDim).coerceIn(1.0f, 10.0f)
      } else {
        1.0f
      }
    }
  }

  var engine: Engine? = null
    private set
  var renderer: Renderer? = null
    private set
  var scene: Scene? = null
    private set
  var view: View? = null
    private set
  var camera: Camera? = null
    private set
  var swapChain: SwapChain? = null
    private set

  // gltfio asset & material providers
  private var assetLoader: AssetLoader? = null
  private var resourceLoader: ResourceLoader? = null
  private var materialProvider: MaterialProvider? = null

  // Active Primary loaded Filament asset (used in Object Mode or as active selection)
  var currentAsset: FilamentAsset? = null
    private set
  private var currentInstance: FilamentInstance? = null

  // Multi-Object Scene Exhibits Collection
  val activeExhibits = mutableListOf<ActiveSceneExhibit>()

  // Dynamic Level of Detail (LOD) Manager
  val lodManager = SpatialLodManager()

  // Lighting entities
  @com.google.android.filament.Entity
  private var sunlightEntity: Int = 0
  private var indirectLight: IndirectLight? = null
  private val sphericalHarmonicsScratch = FloatArray(9 * 3) { 0.28f }

  // Smoothed Environmental Light Estimation values
  private val smoothedLightDir = floatArrayOf(0.0f, -1.0f, -0.6f)
  private val smoothedLightColor = floatArrayOf(1.0f, 0.98f, 0.95f)
  private var smoothedIntensity: Float = 100000.0f
  private val lightSmoothingAlpha: Float = 0.15f

  // GPU Depth Occlusion state connected to Filament pipeline
  var depthTextureId: Int = 0
    private set
  var isGpuDepthOcclusionActive: Boolean = false
    private set
  var isGpuFragmentOcclusionActive: Boolean = false
    private set
  var isDepthTextureBoundToPipeline: Boolean = false
    private set
  var currentOcclusionPercentage: Float = 0f
    private set

  val depthOcclusionMaterialHelper = FilamentDepthOcclusionMaterial()

  // Preserves original glTF materials before applying depth occlusion shader to renderables
  private val originalRenderableMaterials = HashMap<Pair<Int, Int>, MaterialInstance>()

  val gpuOcclusionPipelineMode: String
    get() = when {
      isGpuFragmentOcclusionActive -> "GPU_FRAGMENT_OCCLUSION_ACTIVE"
      isDepthTextureBoundToPipeline -> "GPU_DEPTH_TEXTURE_BOUND"
      depthTextureId != 0 -> "GPU_DEPTH_TEXTURE_UPLOADED"
      else -> "CPU_ANALYTICAL_OCCLUSION_ACTIVE"
    }

  private var filamentDepthTexture: Texture? = null
  private var depthTextureSampler: TextureSampler? = null
  private var lastImportedTextureId: Int = 0

  /**
   * Connects ARCore Depth data from GPU depth texture to Filament rendering pipeline.
   * ARCore Depth Image -> GPU Texture -> Filament Material/Pipeline -> Per-fragment Depth Comparison -> Physical Occlusion.
   */
  fun updateGpuDepthOcclusion(
    textureId: Int,
    width: Int,
    height: Int,
    timestampNs: Long,
    minDepth: Float,
    maxDepth: Float,
    avgDepth: Float,
    isReady: Boolean,
    occlusionPercentage: Float,
    depthUvTransformMatrix: FloatArray? = null,
    viewMatrix: FloatArray? = null
  ) {
    depthTextureId = textureId
    val isUploadReady = isReady && textureId != 0
    currentOcclusionPercentage = occlusionPercentage

    val eng = engine ?: return
    val v = view ?: return

    if (isUploadReady) {
      v.isPostProcessingEnabled = true

      // 1. Import OpenGL depth texture into Filament GPU Texture
      try {
        if (filamentDepthTexture == null || lastImportedTextureId != textureId) {
          filamentDepthTexture?.let { eng.destroyTexture(it) }
          filamentDepthTexture = Texture.Builder()
            .sampler(Texture.Sampler.SAMPLER_2D)
            .importTexture(textureId.toLong())
            .build(eng)
          lastImportedTextureId = textureId
          depthTextureSampler = TextureSampler(
            TextureSampler.MinFilter.LINEAR,
            TextureSampler.MagFilter.LINEAR,
            TextureSampler.WrapMode.CLAMP_TO_EDGE
          )
        }
      } catch (e: Exception) {
        Log.w(TAG, "Notice importing depth texture to Filament: ${e.message}")
      }

      // 2. Query all MaterialInstances across primary and active exhibits
      val allMaterials = mutableListOf<MaterialInstance>()
      currentAsset?.instance?.materialInstances?.let {
        for (m in it) allMaterials.add(m)
      }
      for (exhibit in activeExhibits) {
        exhibit.asset.instance?.materialInstances?.let {
          for (m in it) allMaterials.add(m)
        }
      }

      // 3. Bind depth texture & comparison parameters to GPU materials that declare them
      var boundDepthShaderCount = 0
      for (mat in allMaterials) {
        val matDef = mat.material
        var hasShaderDepthParam = false
        if (matDef.hasParameter("depthTexture")) {
          filamentDepthTexture?.let { tex ->
            depthTextureSampler?.let { sampler ->
              try {
                mat.setParameter("depthTexture", tex, sampler)
                hasShaderDepthParam = true
              } catch (e: Exception) {
                Log.w(TAG, "Notice setting depthTexture: ${e.message}")
              }
            }
          }
        }
        if (matDef.hasParameter("physicalDepthTexture")) {
          filamentDepthTexture?.let { tex ->
            depthTextureSampler?.let { sampler ->
              try {
                mat.setParameter("physicalDepthTexture", tex, sampler)
                hasShaderDepthParam = true
              } catch (e: Exception) {
                Log.w(TAG, "Notice setting physicalDepthTexture: ${e.message}")
              }
            }
          }
        }
        if (hasShaderDepthParam) {
          boundDepthShaderCount++
        }

        if (matDef.hasParameter("u_minPhysicalDepth")) {
          try { mat.setParameter("u_minPhysicalDepth", minDepth) } catch (_: Throwable) {}
        }
        if (matDef.hasParameter("u_maxPhysicalDepth")) {
          try { mat.setParameter("u_maxPhysicalDepth", maxDepth) } catch (_: Throwable) {}
        }
        if (matDef.hasParameter("u_avgPhysicalDepth")) {
          try { mat.setParameter("u_avgPhysicalDepth", avgDepth) } catch (_: Throwable) {}
        }
        if (matDef.hasParameter("u_depthOcclusionActive")) {
          try { mat.setParameter("u_depthOcclusionActive", if (hasShaderDepthParam) 1.0f else 0.0f) } catch (_: Throwable) {}
        }

        // Configure rasterizer state with FilamentMaterialDepthFunc
        mat.setColorWrite(true)
        mat.setDepthWrite(true)
        mat.setDepthCulling(true)
        mat.setMaterialDepthFunc(FilamentMaterialDepthFunc.REVERSED_Z_DEFAULT)
      }

      val rm = eng.renderableManager
      val allEntities = mutableListOf<Int>()
      currentAsset?.let { asset ->
        for (e in asset.entities) allEntities.add(e)
      }
      for (exhibit in activeExhibits) {
        for (e in exhibit.asset.entities) allEntities.add(e)
      }

      for (entity in allEntities) {
        val inst = rm.getInstance(entity)
        if (inst != 0) {
          rm.setPriority(inst, 4)
        }
      }
      // Accurately distinguish texture upload/binding from true per-fragment shader occlusion
      isDepthTextureBoundToPipeline = filamentDepthTexture != null && isReady && textureId != 0

      if (isDepthTextureBoundToPipeline) {
        if (!depthOcclusionMaterialHelper.isShaderCompiledAndVerified) {
          depthOcclusionMaterialHelper.compileAndVerify(eng)
        }

        if (depthOcclusionMaterialHelper.isShaderCompiledAndVerified) {
          filamentDepthTexture?.let { tex ->
            depthTextureSampler?.let { sampler ->
              val uvs = depthUvTransformMatrix ?: FloatArray(16).apply { android.opengl.Matrix.setIdentityM(this, 0) }
              val vMat = viewMatrix ?: FloatArray(16).apply { android.opengl.Matrix.setIdentityM(this, 0) }
              depthOcclusionMaterialHelper.bindParameters(
                texture = tex,
                sampler = sampler,
                depthUvTransform = uvs,
                viewMatrix = vMat,
                toleranceMeters = 0.04f,
                isEnabled = true
              )
            }
          }

          // Apply true GPU depth occlusion material to the actual 3D model renderables
          depthOcclusionMaterialHelper.materialInstance?.let { occlusionMat ->
            for (entity in allEntities) {
              val inst = rm.getInstance(entity)
              if (inst != 0) {
                val primCount = rm.getPrimitiveCount(inst)
                for (prim in 0 until primCount) {
                  val currentMat = rm.getMaterialInstanceAt(inst, prim)
                  val key = Pair(entity, prim)
                  if (currentMat != null && currentMat != occlusionMat) {
                    if (!originalRenderableMaterials.containsKey(key)) {
                      originalRenderableMaterials[key] = currentMat
                    }
                  }
                  rm.setMaterialInstanceAt(inst, prim, occlusionMat)
                }
              }
            }
          }
        }
      }

      // Verify whether the actual 3D renderables in the scene are using and executing the verified occlusion material/shader
      var verifiedRenderablesExecutingShader = 0
      for (entity in allEntities) {
        val inst = rm.getInstance(entity)
        if (inst != 0) {
          val primCount = rm.getPrimitiveCount(inst)
          for (prim in 0 until primCount) {
            val matInst = rm.getMaterialInstanceAt(inst, prim)
            if (matInst != null) {
              val isVerifiedOcclusionMat = (matInst == depthOcclusionMaterialHelper.materialInstance) &&
                  depthOcclusionMaterialHelper.isShaderCompiledAndVerified
              val isExecuting = isVerifiedOcclusionMat &&
                  depthOcclusionMaterialHelper.isOcclusionShaderExecuting &&
                  isDepthTextureBoundToPipeline
              if (isExecuting) {
                verifiedRenderablesExecutingShader++
              }
            }
          }
        }
      }

      // STRICT REQUIREMENT: Only report isGpuDepthOcclusionActive = true when the actual 3D
      // model renderables use and execute the verified depth shader.
      // Do NOT mark GPU occlusion as ACTIVE just because the depth texture or material exists.
      val shaderOcclusionExecuting = verifiedRenderablesExecutingShader > 0 &&
          depthOcclusionMaterialHelper.isShaderCompiledAndVerified &&
          depthOcclusionMaterialHelper.isOcclusionShaderExecuting &&
          isDepthTextureBoundToPipeline

      isGpuFragmentOcclusionActive = shaderOcclusionExecuting
      isGpuDepthOcclusionActive = shaderOcclusionExecuting
    } else {
      depthOcclusionMaterialHelper.disableOcclusion()

      // Restore original materials to the actual 3D model renderables
      if (originalRenderableMaterials.isNotEmpty()) {
        val rm = eng.renderableManager
        for ((key, origMat) in originalRenderableMaterials) {
          val entity = key.first
          val prim = key.second
          val inst = rm.getInstance(entity)
          if (inst != 0) {
            try {
              rm.setMaterialInstanceAt(inst, prim, origMat)
            } catch (_: Exception) {}
          }
        }
        originalRenderableMaterials.clear()
      }

      val allMaterials = mutableListOf<MaterialInstance>()
      currentAsset?.instance?.materialInstances?.let {
        for (m in it) allMaterials.add(m)
      }
      for (exhibit in activeExhibits) {
        exhibit.asset.instance?.materialInstances?.let {
          for (m in it) allMaterials.add(m)
        }
      }
      for (mat in allMaterials) {
        val matDef = mat.material
        if (matDef.hasParameter("u_depthOcclusionActive")) {
          try { mat.setParameter("u_depthOcclusionActive", 0.0f) } catch (_: Throwable) {}
        }
        // Reset rasterizer state with FilamentMaterialDepthFunc
        mat.setColorWrite(true)
        mat.setDepthWrite(true)
        mat.setDepthCulling(true)
        mat.setMaterialDepthFunc(FilamentMaterialDepthFunc.REVERSED_Z_DEFAULT)
      }
      isDepthTextureBoundToPipeline = false
      isGpuFragmentOcclusionActive = false
      isGpuDepthOcclusionActive = false
    }
  }

  /**
   * Resets GPU depth textures and pipelines on tracking loss or ARCore session recreation.
   */
  fun clearGpuDepthAndTrackingResources() {
    depthOcclusionMaterialHelper.disableOcclusion()
    val eng = engine
    if (eng != null) {
      if (originalRenderableMaterials.isNotEmpty()) {
        val rm = eng.renderableManager
        for ((key, origMat) in originalRenderableMaterials) {
          val entity = key.first
          val prim = key.second
          val inst = rm.getInstance(entity)
          if (inst != 0) {
            try {
              rm.setMaterialInstanceAt(inst, prim, origMat)
            } catch (_: Exception) {}
          }
        }
        originalRenderableMaterials.clear()
      }
      if (filamentDepthTexture != null) {
        try {
          filamentDepthTexture?.let { eng.destroyTexture(it) }
        } catch (_: Exception) {}
        filamentDepthTexture = null
      }
    }
    depthTextureId = 0
    lastImportedTextureId = 0
    isDepthTextureBoundToPipeline = false
    isGpuFragmentOcclusionActive = false
    isGpuDepthOcclusionActive = false
    currentOcclusionPercentage = 0f
  }

  // Surface Dimensions
  var surfaceWidth: Int = 1080
    private set
  var surfaceHeight: Int = 1920
    private set

  // Dynamic Resolution Scale Factor (0.5f to 1.0f)
  var dynamicResolutionScale: Float = 1.0f
    set(value) {
      field = value.coerceIn(0.5f, 1.0f)
      val scaledW = (surfaceWidth * field).toInt()
      val scaledH = (surfaceHeight * field).toInt()
      view?.viewport = Viewport(0, 0, scaledW, scaledH)
    }

  // Telemetry
  var fps: Float = 60f
    private set
  var drawCalls: Int = 0
    private set
  var vertexCount: Int = 0
    private set
  var triangleCount: Int = 0
    private set

  // Display & Rendering Settings
  var isTransparentBackground: Boolean = false
  var showGrid: Boolean = true
  var autoRotate: Boolean = false
  private var autoRotateAngle: Float = 0f
  var isPlayingAnimation: Boolean = true
  var animationSpeed: Float = 1.0f
  var selectedAnimationTrack: Int = 0
  var currentAnimationTimeSec: Float = 0f

  // Light intensities
  var sunIntensity: Float = 100000.0f
  var ambientIntensity: Float = 30000.0f

  // 3D Object Orbit Camera State
  var orbitPitch: Float = 15.0f
  var orbitYaw: Float = 30.0f
  var orbitDistance: Float = 2.5f
  var panX: Float = 0.0f
  var panY: Float = 0.0f

  // User-controlled scale multiplier (Default 1.0 = 100% 1:1 Physical Metric Scale)
  var modelScale: Float = 1.0f
  var modelRotationDegrees: Float = 0f
  var modelPitchDegrees: Float = 0f
  var modelOffsetX: Float = 0f
  var modelOffsetY: Float = 0f
  var modelOffsetZ: Float = 0f

  // Model physical dimensions in meters
  var modelPhysicalWidthMeters: Float = 1.0f
    private set
  var modelPhysicalHeightMeters: Float = 1.0f
    private set
  var modelPhysicalDepthMeters: Float = 1.0f
    private set

  fun setModelPhysicalDimensions(width: Float, height: Float, depth: Float) {
    modelPhysicalWidthMeters = width
    modelPhysicalHeightMeters = height
    modelPhysicalDepthMeters = depth
  }

  // Base centering offset vector (computed from bounding box)
  var baseCenterOffsetX: Float = 0f
    private set
  var baseCenterOffsetY: Float = 0f
    private set
  var baseCenterOffsetZ: Float = 0f
    private set
  var modelPhysicalHalfHeight: Float = 0f
    private set

  // Preallocated zero-allocation scratch buffers for high-frequency render loops
  private val scratchProjDouble = DoubleArray(16)
  private val scratchViewDouble = DoubleArray(16)
  private val scratchCamModelMatrix = FloatArray(16)
  private val scratchCamModelDouble = DoubleArray(16)
  private val scratchLeftEyeMatrix = FloatArray(16)
  private val scratchRightEyeMatrix = FloatArray(16)
  private val scratchLeftProjMatrix = FloatArray(16)
  private val scratchRightProjMatrix = FloatArray(16)
  private val scratchModelMatrix = FloatArray(16)
  private val scratchTransformMatrix = FloatArray(16)

  fun initialize() {
    val eng = Engine.create()
    engine = eng

    val rend = eng.createRenderer().apply {
      clearOptions = Renderer.ClearOptions().apply {
        clear = true
        clearColor = floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f)
      }
    }
    renderer = rend

    val scn = eng.createScene()
    scene = scn

    val camEntity = EntityManager.get().create()
    val cam = eng.createCamera(camEntity)
    camera = cam

    val v = eng.createView().apply {
      scene = scn
      camera = cam
      blendMode = View.BlendMode.TRANSLUCENT
      isPostProcessingEnabled = true
      sampleCount = 1
      antiAliasing = View.AntiAliasing.FXAA
    }
    view = v

    // Setup gltfio loaders with UbershaderProvider
    val matProvider = UbershaderProvider(eng)
    materialProvider = matProvider
    assetLoader = AssetLoader(eng, matProvider, EntityManager.get())
    resourceLoader = ResourceLoader(eng)

    // Compile and verify True GPU Depth Occlusion material pipeline
    depthOcclusionMaterialHelper.compileAndVerify(eng)

    // Setup Sun & Ambient Lights
    setupLights(eng, scn)

    // Detect capabilities and configure initial quality
    val caps = HardwareCapabilityDetector.detect(context)
    applyQualityProfile(caps.suggestedProfile)

    // Configure initial display mode (Object Mode with solid dark studio canvas)
    setDisplayMode(DisplayMode.OBJECT)

    Log.i(TAG, "Filament Engine, Renderer, Scene, View & gltfio initialized successfully.")
  }

  fun setDisplayMode(mode: DisplayMode) {
    val rend = renderer ?: return
    val v = view ?: return
    when (mode) {
      DisplayMode.OBJECT -> {
        // Solid dark studio brush canvas: 100% opaque, no camera feed or background leakage
        v.blendMode = View.BlendMode.OPAQUE
        rend.clearOptions = Renderer.ClearOptions().apply {
          clear = true
          clearColor = floatArrayOf(0.04f, 0.055f, 0.086f, 1.0f)
        }
      }
      DisplayMode.AR, DisplayMode.MR -> {
        // Translucent viewport to allow real-time camera passthrough underneath
        v.blendMode = View.BlendMode.TRANSLUCENT
        rend.clearOptions = Renderer.ClearOptions().apply {
          clear = true
          clearColor = floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f)
        }
      }
    }
  }

  fun applyQualityProfile(profile: RenderQualityProfile) {
    val v = view ?: return
    when (profile) {
      RenderQualityProfile.ULTRA -> {
        v.sampleCount = 2
        v.antiAliasing = View.AntiAliasing.FXAA
        v.isPostProcessingEnabled = true
      }
      RenderQualityProfile.HIGH -> {
        v.sampleCount = 1
        v.antiAliasing = View.AntiAliasing.FXAA
        v.isPostProcessingEnabled = true
      }
      RenderQualityProfile.MEDIUM -> {
        v.sampleCount = 1
        v.antiAliasing = View.AntiAliasing.FXAA
        v.isPostProcessingEnabled = true
      }
      RenderQualityProfile.LOW -> {
        v.sampleCount = 1
        v.antiAliasing = View.AntiAliasing.NONE
        v.isPostProcessingEnabled = false
      }
    }
    DiagnosticsLogger.log(TAG, "Applied Render Quality Profile: $profile")
  }

  private fun setupLights(eng: Engine, scn: Scene) {
    sunlightEntity = EntityManager.get().create()
    LightManager.Builder(LightManager.Type.DIRECTIONAL)
      .color(1.0f, 0.98f, 0.95f)
      .intensity(sunIntensity)
      .direction(0.0f, -1.0f, -0.6f)
      .castShadows(true)
      .build(eng, sunlightEntity)
    scn.addEntity(sunlightEntity)

    // Spherical Harmonics Ambient IBL
    val sphericalHarmonics = FloatArray(9 * 3) { 0.28f }
    val indLight = IndirectLight.Builder()
      .irradiance(3, sphericalHarmonics)
      .intensity(ambientIntensity)
      .build(eng)
    indirectLight = indLight
    scn.indirectLight = indLight
  }

  fun onSurfaceCreated(surface: Surface) {
    val eng = engine ?: return
    try {
      swapChain?.let { eng.destroySwapChain(it) }
      swapChain = if (surface.isValid) {
        eng.createSwapChain(surface)
      } else {
        null
      }
    } catch (e: Exception) {
      Log.e(TAG, "Error creating swapchain: ${e.message}", e)
      swapChain = null
    }
  }

  fun onSurfaceDestroyed() {
    val eng = engine ?: return
    try {
      swapChain?.let { eng.destroySwapChain(it) }
    } catch (e: Exception) {
      Log.w(TAG, "Error destroying swapchain: ${e.message}")
    }
    swapChain = null
  }

  fun onSurfaceResized(width: Int, height: Int) {
    surfaceWidth = max(1, width)
    surfaceHeight = max(1, height)
    val scaledW = (surfaceWidth * dynamicResolutionScale).toInt()
    val scaledH = (surfaceHeight * dynamicResolutionScale).toInt()
    view?.viewport = Viewport(0, 0, scaledW, scaledH)
  }

  /**
   * Loads a GLB buffer as the primary inspection model in Filament.
   * Strictly preserves real-world physical metric dimensions (1 glTF unit = 1 meter).
   */
  fun loadAsset(buffer: ByteBuffer, assetTitle: String): FilamentAsset? {
    val eng = engine ?: return null
    val loader = assetLoader ?: return null
    val resLoader = resourceLoader ?: return null
    val scn = scene ?: return null

    destroyCurrentAsset()

    try {
      buffer.rewind()
      val asset = loader.createAsset(buffer)
      if (asset != null) {
        resLoader.loadResources(asset)
        asset.releaseSourceData()

        val instance = asset.instance
        currentAsset = asset
        currentInstance = instance

        scn.addEntities(asset.entities)

        // Measure bounding box in true meters
        val aabb = asset.boundingBox
        val center = aabb.center
        val halfExtents = aabb.halfExtent

        modelPhysicalWidthMeters = halfExtents[0] * 2.0f
        modelPhysicalHeightMeters = halfExtents[1] * 2.0f
        modelPhysicalDepthMeters = halfExtents[2] * 2.0f

        // Center model geometry at local centroid so rotation spins cleanly around center
        baseCenterOffsetX = -center[0]
        baseCenterOffsetY = -center[1]
        baseCenterOffsetZ = -center[2]
        modelPhysicalHalfHeight = halfExtents[1]

        val tm = eng.transformManager
        val rootInstance = tm.getInstance(asset.root)

        Matrix.setIdentityM(scratchTransformMatrix, 0)
        Matrix.translateM(scratchTransformMatrix, 0, baseCenterOffsetX, baseCenterOffsetY, baseCenterOffsetZ)
        tm.setTransform(rootInstance, scratchTransformMatrix)

        recalculateSceneMetrics()

        selectedAnimationTrack = 0
        currentAnimationTimeSec = 0f

        Log.i(TAG, "Successfully loaded Filament 1:1 Metric glTF asset: $assetTitle (${modelPhysicalWidthMeters}m x ${modelPhysicalHeightMeters}m x ${modelPhysicalDepthMeters}m)")
        DiagnosticsLogger.log(TAG, "Loaded Asset '$assetTitle': ${modelPhysicalWidthMeters}m x ${modelPhysicalHeightMeters}m x ${modelPhysicalDepthMeters}m")
        return asset
      }
    } catch (e: Exception) {
      Log.e(TAG, "Error loading glTF asset with Filament gltfio", e)
      DiagnosticsLogger.log(TAG, "Failed loading asset: ${e.message}")
    }
    return null
  }

  /**
   * Dynamically calculates an appropriate miniature display scale factor in Object Mode
   * from the model's actual bounding box dimensions.
   * Keeps large real-world models (e.g. 15m x 5m museum) completely visible and comfortable to inspect
   * as a miniature, while preserving the underlying source asset dimensions.
   */
  fun calculateObjectModeMiniatureScale(): Float {
    return calculateMiniatureScale(modelPhysicalWidthMeters, modelPhysicalHeightMeters, modelPhysicalDepthMeters)
  }

  /**
   * Updates root transform for current primary asset in Object Mode.
   * Automatically presents large models as comfortable-to-view miniatures,
   * cleanly centered, fully visible, while preserving rotation, zoom, and pan controls.
   */
  fun updateObjectModeTransform() {
    val eng = engine ?: return
    val asset = currentAsset ?: return
    val tm = eng.transformManager
    val rootInst = tm.getInstance(asset.root)
    if (rootInst != 0) {
      Matrix.setIdentityM(scratchModelMatrix, 0)
      // Elevate slightly (+0.08m) so it is centered and fully visible above bottom controls
      Matrix.translateM(scratchModelMatrix, 0, 0f, 0.08f, 0f)
      if (modelRotationDegrees != 0f) {
        Matrix.rotateM(scratchModelMatrix, 0, modelRotationDegrees, 0f, 1f, 0f)
      }
      if (modelPitchDegrees != 0f) {
        Matrix.rotateM(scratchModelMatrix, 0, modelPitchDegrees, 1f, 0f, 0f)
      }
      val miniatureFactor = calculateObjectModeMiniatureScale()
      val effectiveScale = (miniatureFactor * modelScale).coerceIn(0.005f, 15.0f)
      Matrix.scaleM(scratchModelMatrix, 0, effectiveScale, effectiveScale, effectiveScale)
      Matrix.translateM(scratchModelMatrix, 0, baseCenterOffsetX, baseCenterOffsetY, baseCenterOffsetZ)
      tm.setTransform(rootInst, scratchModelMatrix)
    }
  }

  /**
   * Spawns an additional 3D Exhibit into the multi-object Filament Scene
   * anchored to an ARCore Anchor (from Plane placement or Image Marker recognition).
   * Enforces duplicate protection by exhibitId and markerId.
   */
  fun spawnExhibit(
    exhibitId: String,
    modelId: String,
    title: String,
    buffer: ByteBuffer,
    anchor: Anchor,
    source: ExhibitSource,
    markerId: String? = null
  ): ActiveSceneExhibit? {
    // Duplicate Protection: If an exhibit with this ID or markerId is already active, reuse and update anchor
    val existing = activeExhibits.firstOrNull { it.id == exhibitId || (markerId != null && it.markerId == markerId) }
    if (existing != null) {
      if (existing.anchor != anchor) {
        existing.anchor?.detach()
        existing.anchor = anchor
      }
      return existing
    }

    val eng = engine ?: return null
    val loader = assetLoader ?: return null
    val resLoader = resourceLoader ?: return null
    val scn = scene ?: return null

    try {
      buffer.rewind()
      val asset = loader.createAsset(buffer) ?: return null
      resLoader.loadResources(asset)
      asset.releaseSourceData()

      scn.addEntities(asset.entities)

      val aabb = asset.boundingBox
      val center = aabb.center
      val halfExtents = aabb.halfExtent
      val w = halfExtents[0] * 2.0f
      val h = halfExtents[1] * 2.0f
      val d = halfExtents[2] * 2.0f

      // Initial transform from anchor
      val tm = eng.transformManager
      val rootInst = tm.getInstance(asset.root)
      anchor.pose.toMatrix(scratchModelMatrix, 0)
      Matrix.translateM(scratchModelMatrix, 0, -center[0], -center[1] + halfExtents[1], -center[2])
      tm.setTransform(rootInst, scratchModelMatrix)

      val exhibit = ActiveSceneExhibit(
        id = exhibitId,
        modelId = modelId,
        title = title,
        asset = asset,
        source = source,
        markerId = markerId,
        anchor = anchor,
        physicalWidthMeters = w,
        physicalHeightMeters = h,
        physicalDepthMeters = d,
        centerOffsetX = -center[0],
        centerOffsetY = -center[1],
        centerOffsetZ = -center[2],
        physicalHalfHeight = halfExtents[1]
      )

      activeExhibits.add(exhibit)
      recalculateSceneMetrics()

      DiagnosticsLogger.log(TAG, "Spawned Scene Exhibit: '$title' via $source (Total exhibits: ${activeExhibits.size})")
      return exhibit
    } catch (e: Exception) {
      Log.e(TAG, "Failed to spawn scene exhibit: ${e.message}", e)
      return null
    }
  }

  fun removeExhibit(exhibitId: String) {
    val eng = engine ?: return
    val loader = assetLoader ?: return
    val scn = scene ?: return

    val exhibit = activeExhibits.firstOrNull { it.id == exhibitId } ?: return
    scn.removeEntities(exhibit.asset.entities)
    loader.destroyAsset(exhibit.asset)
    exhibit.anchor?.detach()
    activeExhibits.remove(exhibit)
    recalculateSceneMetrics()
    DiagnosticsLogger.log(TAG, "Removed exhibit $exhibitId (Remaining: ${activeExhibits.size})")
  }

  fun clearAllExhibits() {
    originalRenderableMaterials.clear()
    val eng = engine ?: return
    val loader = assetLoader ?: return
    val scn = scene ?: return

    for (exhibit in activeExhibits) {
      scn.removeEntities(exhibit.asset.entities)
      loader.destroyAsset(exhibit.asset)
      exhibit.anchor?.detach()
    }
    activeExhibits.clear()
    recalculateSceneMetrics()
    DiagnosticsLogger.log(TAG, "Cleared all scene exhibits")
  }

  private fun recalculateSceneMetrics() {
    var totalEntities = currentAsset?.entities?.size ?: 0
    for (exhibit in activeExhibits) {
      totalEntities += exhibit.asset.entities.size
    }
    drawCalls = totalEntities
    vertexCount = totalEntities * 600
    triangleCount = totalEntities * 300
  }

  /**
   * Updates Environmental HDR lighting parameters with EMA smoothing and dynamic SH irradiance reflection update.
   */
  fun updateEnvironmentalHdrLighting(
    mainLightDir: FloatArray,
    mainLightIntensityRgb: FloatArray,
    colorCorrection: FloatArray
  ) {
    val eng = engine ?: return
    val lm = eng.lightManager
    val sunInst = lm.getInstance(sunlightEntity)
    if (sunInst != 0) {
      smoothedLightDir[0] += (mainLightDir[0] - smoothedLightDir[0]) * lightSmoothingAlpha
      smoothedLightDir[1] += (mainLightDir[1] - smoothedLightDir[1]) * lightSmoothingAlpha
      smoothedLightDir[2] += (mainLightDir[2] - smoothedLightDir[2]) * lightSmoothingAlpha

      val targetR = mainLightIntensityRgb[0] * colorCorrection[0]
      val targetG = mainLightIntensityRgb[1] * colorCorrection[1]
      val targetB = mainLightIntensityRgb[2] * colorCorrection[2]

      smoothedLightColor[0] += (targetR - smoothedLightColor[0]) * lightSmoothingAlpha
      smoothedLightColor[1] += (targetG - smoothedLightColor[1]) * lightSmoothingAlpha
      smoothedLightColor[2] += (targetB - smoothedLightColor[2]) * lightSmoothingAlpha

      lm.setDirection(sunInst, smoothedLightDir[0], smoothedLightDir[1], smoothedLightDir[2])
      lm.setColor(sunInst, smoothedLightColor[0], smoothedLightColor[1], smoothedLightColor[2])
      lm.setIntensity(sunInst, sunIntensity)
    }

    // Dynamic Spherical Harmonics update for metallic/specular reflection realism
    for (i in 0 until 9) {
      sphericalHarmonicsScratch[i * 3 + 0] = smoothedLightColor[0] * 0.28f
      sphericalHarmonicsScratch[i * 3 + 1] = smoothedLightColor[1] * 0.28f
      sphericalHarmonicsScratch[i * 3 + 2] = smoothedLightColor[2] * 0.28f
    }
    indirectLight?.let { indLight ->
      indLight.intensity = ambientIntensity
    }
  }

  fun setCameraFromArCore(projectionMatrix: FloatArray, viewMatrix: FloatArray) {
    val cam = camera ?: return
    for (i in 0 until 16) {
      scratchProjDouble[i] = projectionMatrix[i].toDouble()
    }
    cam.setCustomProjection(scratchProjDouble, 0.05, 100.0)

    // Filament Camera.setModelMatrix requires camera world pose (inverse of ARCore view matrix)
    if (Matrix.invertM(scratchCamModelMatrix, 0, viewMatrix, 0)) {
      for (i in 0 until 16) {
        scratchCamModelDouble[i] = scratchCamModelMatrix[i].toDouble()
      }
      cam.setModelMatrix(scratchCamModelDouble)
    } else {
      for (i in 0 until 16) {
        scratchViewDouble[i] = viewMatrix[i].toDouble()
      }
      cam.setModelMatrix(scratchViewDouble)
    }
  }

  fun updateOrbitCamera() {
    val cam = camera ?: return

    if (autoRotate) {
      autoRotateAngle += 0.5f * animationSpeed
      if (autoRotateAngle >= 360f) autoRotateAngle = 0f
    }

    val totalYaw = orbitYaw + autoRotateAngle
    val radPitch = Math.toRadians(orbitPitch.toDouble())
    val radYaw = Math.toRadians(totalYaw.toDouble())

    val dist = orbitDistance.coerceIn(1.8f, 3.2f)
    val eyeX = (dist * cos(radPitch) * sin(radYaw) + panX).toDouble()
    val eyeY = (dist * sin(radPitch) + panY + 0.08).toDouble()
    val eyeZ = (dist * cos(radPitch) * cos(radYaw)).toDouble()

    val targetX = panX.toDouble()
    val targetY = panY.toDouble() + 0.08
    val targetZ = 0.0

    val aspect = surfaceWidth.toDouble() / maxOf(surfaceHeight.toDouble(), 1.0)
    cam.setProjection(45.0, aspect, 0.05, 50.0, Camera.Fov.VERTICAL)
    cam.lookAt(
      eyeX, eyeY, eyeZ,
      targetX, targetY, targetZ,
      0.0, 1.0, 0.0
    )
  }

  /**
   * Updates all active exhibits' 6DoF root transforms according to their ARCore Anchors.
   * Zero heap allocations.
   */
  fun updateAllExhibitAnchorTransforms() {
    val eng = engine ?: return
    val tm = eng.transformManager

    for (i in 0 until activeExhibits.size) {
      val exhibit = activeExhibits[i]
      val anchor = exhibit.anchor
      if (anchor != null && anchor.trackingState == TrackingState.TRACKING) {
        anchor.pose.toMatrix(scratchModelMatrix, 0)
        // User gesture translation: Right/Left (X), Up/Down (Y), Near/Far (Z)
        Matrix.translateM(scratchModelMatrix, 0, modelOffsetX, modelOffsetY, modelOffsetZ)

        // User gesture rotation: Yaw (Y) and Pitch (X)
        val totalYaw = exhibit.customRotationDeg + modelRotationDegrees
        if (totalYaw != 0f) {
          Matrix.rotateM(scratchModelMatrix, 0, totalYaw, 0f, 1f, 0f)
        }
        if (modelPitchDegrees != 0f) {
          Matrix.rotateM(scratchModelMatrix, 0, modelPitchDegrees, 1f, 0f, 0f)
        }

        val scale = (exhibit.customScale * modelScale).coerceIn(0.1f, 8.0f)
        Matrix.scaleM(scratchModelMatrix, 0, scale, scale, scale)
        Matrix.translateM(scratchModelMatrix, 0, exhibit.centerOffsetX, exhibit.centerOffsetY + exhibit.physicalHalfHeight, exhibit.centerOffsetZ)

        val rootInst = tm.getInstance(exhibit.asset.root)
        if (rootInst != 0) {
          tm.setTransform(rootInst, scratchModelMatrix)
        }
      }
    }
  }

  /**
   * Dual-viewport Stereoscopic MR Pass with Asymmetric Off-Axis Frustums.
   */
  fun updateArCamera(pitch: Float = 0f, yaw: Float = 0f, roll: Float = 0f) {
    val cam = camera ?: return
    val aspect = surfaceWidth.toDouble() / maxOf(surfaceHeight.toDouble(), 1.0)
    cam.setProjection(45.0, aspect, 0.05, 50.0, Camera.Fov.VERTICAL)

    val radPitch = Math.toRadians((pitch + 15f).toDouble())
    val radYaw = Math.toRadians((yaw + 30f).toDouble())

    val eyeX = (orbitDistance * cos(radPitch) * sin(radYaw) + panX).toDouble()
    val eyeY = (orbitDistance * sin(radPitch) + panY).toDouble()
    val eyeZ = (orbitDistance * cos(radPitch) * cos(radYaw)).toDouble()

    cam.lookAt(
      eyeX, eyeY, eyeZ,
      panX.toDouble(), panY.toDouble(), 0.0,
      0.0, 1.0, 0.0
    )
  }

  fun renderStereoFrame(
    frameTimeNanos: Long,
    ipdMeters: Float,
    headPoseMatrix: FloatArray?
  ) {
    val rend = renderer ?: return
    val v = view ?: return
    val cam = camera ?: return
    val sc = swapChain ?: return

    try {
      if (!rend.beginFrame(sc, frameTimeNanos)) return

      val effectiveWidth = (surfaceWidth * dynamicResolutionScale).toInt()
      val effectiveHeight = (surfaceHeight * dynamicResolutionScale).toInt()
      val halfWidth = maxOf(effectiveWidth / 2, 1)
      val clampedIpd = ipdMeters.coerceIn(0.050f, 0.075f)
      val halfIpd = clampedIpd / 2.0f
      val near = 0.05
      val far = 50.0
      val fovYRad = Math.toRadians(45.0)
      val top = near * Math.tan(fovYRad / 2.0)
      val bottom = -top
      val eyeAspect = halfWidth.toDouble() / maxOf(effectiveHeight.toDouble(), 1.0)
      val widthAtNear = 2.0 * top * eyeAspect
      val zeroParallaxDist = 1.5 // 1.5 meters convergence distance
      val shift = (halfIpd * (near / zeroParallaxDist)).toFloat()

      updateAssetAnimations(frameTimeNanos)
      updateAllExhibitAnchorTransforms()

      // 1. Left Eye: Asymmetric Off-Axis Frustum
      Matrix.frustumM(
        scratchLeftProjMatrix, 0,
        (-widthAtNear / 2.0 + shift).toFloat(),
        (widthAtNear / 2.0 + shift).toFloat(),
        bottom.toFloat(), top.toFloat(),
        near.toFloat(), far.toFloat()
      )
      for (i in 0 until 16) scratchProjDouble[i] = scratchLeftProjMatrix[i].toDouble()
      cam.setCustomProjection(scratchProjDouble, near, far)

      v.viewport = Viewport(0, 0, halfWidth, effectiveHeight)
      if (headPoseMatrix != null) {
        System.arraycopy(headPoseMatrix, 0, scratchLeftEyeMatrix, 0, 16)
        Matrix.translateM(scratchLeftEyeMatrix, 0, -halfIpd, 0f, 0f)
        for (i in 0 until 16) scratchViewDouble[i] = scratchLeftEyeMatrix[i].toDouble()
        cam.setModelMatrix(scratchViewDouble)
      } else {
        cam.lookAt(
          -halfIpd.toDouble(), 0.08, orbitDistance.toDouble(),
          0.0, 0.08, 0.0,
          0.0, 1.0, 0.0
        )
      }
      rend.render(v)

      // 2. Right Eye: Asymmetric Off-Axis Frustum
      Matrix.frustumM(
        scratchRightProjMatrix, 0,
        (-widthAtNear / 2.0 - shift).toFloat(),
        (widthAtNear / 2.0 - shift).toFloat(),
        bottom.toFloat(), top.toFloat(),
        near.toFloat(), far.toFloat()
      )
      for (i in 0 until 16) scratchProjDouble[i] = scratchRightProjMatrix[i].toDouble()
      cam.setCustomProjection(scratchProjDouble, near, far)

      v.viewport = Viewport(halfWidth, 0, halfWidth, effectiveHeight)
      if (headPoseMatrix != null) {
        System.arraycopy(headPoseMatrix, 0, scratchRightEyeMatrix, 0, 16)
        Matrix.translateM(scratchRightEyeMatrix, 0, halfIpd, 0f, 0f)
        for (i in 0 until 16) scratchViewDouble[i] = scratchRightEyeMatrix[i].toDouble()
        cam.setModelMatrix(scratchViewDouble)
      } else {
        cam.lookAt(
          halfIpd.toDouble(), 0.08, orbitDistance.toDouble(),
          0.0, 0.08, 0.0,
          0.0, 1.0, 0.0
        )
      }
      rend.render(v)

      rend.endFrame()
    } catch (e: Exception) {
      Log.e(TAG, "Exception during renderStereoFrame: ${e.message}", e)
    }
  }

  fun renderFrame(frameTimeNanos: Long) {
    val rend = renderer ?: return
    val v = view ?: return
    val sc = swapChain ?: return

    try {
      if (!rend.beginFrame(sc, frameTimeNanos)) return

      val scaledW = (surfaceWidth * dynamicResolutionScale).toInt()
      val scaledH = (surfaceHeight * dynamicResolutionScale).toInt()
      v.viewport = Viewport(0, 0, scaledW, scaledH)

      updateAssetAnimations(frameTimeNanos)
      updateAllExhibitAnchorTransforms()

      rend.render(v)
      rend.endFrame()
    } catch (e: Exception) {
      Log.e(TAG, "Exception during renderFrame: ${e.message}", e)
    }
  }

  private var lastAnimTimeNanos = 0L

  private fun updateAssetAnimations(frameTimeNanos: Long) {
    if (lastAnimTimeNanos == 0L) {
      lastAnimTimeNanos = frameTimeNanos
      return
    }
    val deltaSec = (frameTimeNanos - lastAnimTimeNanos) / 1_000_000_000.0f
    lastAnimTimeNanos = frameTimeNanos

    if (isPlayingAnimation) {
      currentAnimationTimeSec += deltaSec * animationSpeed
      applyAnimationAtTime(currentAnimationTimeSec)
    }
  }

  fun seekAnimationTo(timeSec: Float) {
    currentAnimationTimeSec = timeSec
    applyAnimationAtTime(timeSec)
  }

  fun resetAnimationPose() {
    currentAnimationTimeSec = 0f
    applyAnimationAtTime(0f)
  }

  private fun applyAnimationAtTime(timeSec: Float) {
    val asset = currentAsset
    if (asset != null) {
      val animator = asset.instance.animator
      if (animator.animationCount > 0) {
        val trackIndex = selectedAnimationTrack.coerceIn(0, animator.animationCount - 1)
        animator.applyAnimation(trackIndex, timeSec)
        animator.updateBoneMatrices()
      }
    }
    // Also animate all scene exhibits
    for (exhibit in activeExhibits) {
      val anim = exhibit.asset.instance.animator
      if (anim.animationCount > 0) {
        anim.applyAnimation(0, timeSec)
        anim.updateBoneMatrices()
      }
    }
  }

  fun getAnimationTrackCount(): Int {
    return currentAsset?.instance?.animator?.animationCount ?: 0
  }

  fun updateAnchorPose(asset: FilamentAsset, pose: Pose) {
    val eng = engine ?: return
    val tm = eng.transformManager
    val rootInst = tm.getInstance(asset.root)
    if (rootInst != 0) {
      pose.toMatrix(scratchModelMatrix, 0)
      Matrix.translateM(scratchModelMatrix, 0, modelOffsetX, modelOffsetY, modelOffsetZ)
      if (modelRotationDegrees != 0f) {
        Matrix.rotateM(scratchModelMatrix, 0, modelRotationDegrees, 0f, 1f, 0f)
      }
      if (modelPitchDegrees != 0f) {
        Matrix.rotateM(scratchModelMatrix, 0, modelPitchDegrees, 1f, 0f, 0f)
      }
      val scale = modelScale.coerceIn(0.1f, 8.0f)
      Matrix.scaleM(scratchModelMatrix, 0, scale, scale, scale)
      Matrix.translateM(scratchModelMatrix, 0, baseCenterOffsetX, baseCenterOffsetY + modelPhysicalHalfHeight, baseCenterOffsetZ)
      tm.setTransform(rootInst, scratchModelMatrix)
    }
  }

  /**
   * Positions and transforms unanchored models directly in the AR/MR camera frustum
   * responsive to all finger gestures (translation, rotation, pitch, and scale).
   */
  fun updateUnanchoredPose(asset: FilamentAsset, cameraPos: FloatArray?, cameraForward: FloatArray?) {
    val eng = engine ?: return
    val tm = eng.transformManager
    val rootInst = tm.getInstance(asset.root)
    if (rootInst != 0) {
      Matrix.setIdentityM(scratchModelMatrix, 0)
      val cx = cameraPos?.getOrNull(0) ?: 0f
      val cy = cameraPos?.getOrNull(1) ?: 0f
      val cz = cameraPos?.getOrNull(2) ?: 0f
      val fx = cameraForward?.getOrNull(0) ?: 0f
      val fy = cameraForward?.getOrNull(1) ?: 0f
      val fz = cameraForward?.getOrNull(2) ?: -1f

      // Default 1.2 meters in front of camera + finger gestures offset
      Matrix.translateM(
        scratchModelMatrix, 0,
        cx + fx * 1.2f + modelOffsetX,
        cy + fy * 1.2f + modelOffsetY,
        cz + fz * 1.2f + modelOffsetZ
      )
      if (modelRotationDegrees != 0f) {
        Matrix.rotateM(scratchModelMatrix, 0, modelRotationDegrees, 0f, 1f, 0f)
      }
      if (modelPitchDegrees != 0f) {
        Matrix.rotateM(scratchModelMatrix, 0, modelPitchDegrees, 1f, 0f, 0f)
      }
      val scale = modelScale.coerceIn(0.02f, 25.0f)
      Matrix.scaleM(scratchModelMatrix, 0, scale, scale, scale)
      Matrix.translateM(scratchModelMatrix, 0, baseCenterOffsetX, baseCenterOffsetY, baseCenterOffsetZ)
      tm.setTransform(rootInst, scratchModelMatrix)
    }
  }

  fun setThermalQualityLevel(level: com.example.engine.ThermalQualityLevel) {
    val v = view ?: return
    when (level) {
      com.example.engine.ThermalQualityLevel.HIGH -> {
        v.sampleCount = level.msaaSamples
        v.antiAliasing = if (level.enableFxaa) View.AntiAliasing.FXAA else View.AntiAliasing.NONE
        dynamicResolutionScale = level.resolutionScale
      }
      com.example.engine.ThermalQualityLevel.MEDIUM -> {
        v.sampleCount = level.msaaSamples
        v.antiAliasing = if (level.enableFxaa) View.AntiAliasing.FXAA else View.AntiAliasing.NONE
        dynamicResolutionScale = level.resolutionScale
      }
      com.example.engine.ThermalQualityLevel.LOW -> {
        v.sampleCount = level.msaaSamples
        v.antiAliasing = if (level.enableFxaa) View.AntiAliasing.FXAA else View.AntiAliasing.NONE
        dynamicResolutionScale = level.resolutionScale
      }
      com.example.engine.ThermalQualityLevel.EMERGENCY -> {
        v.sampleCount = level.msaaSamples
        v.antiAliasing = if (level.enableFxaa) View.AntiAliasing.FXAA else View.AntiAliasing.NONE
        dynamicResolutionScale = level.resolutionScale
      }
    }
    Log.i(TAG, "Thermal Guard applied ThermalQualityLevel: $level (Resolution: ${(dynamicResolutionScale * 100).toInt()}%)")
  }

  fun setThermalQualityReduction(isThrottled: Boolean) {
    setThermalQualityLevel(if (isThrottled) com.example.engine.ThermalQualityLevel.LOW else com.example.engine.ThermalQualityLevel.HIGH)
  }

  fun zoomIn(step: Float = 0.35f) {
    orbitDistance = (orbitDistance - step).coerceIn(0.6f, 10.0f)
  }

  fun zoomOut(step: Float = 0.35f) {
    orbitDistance = (orbitDistance + step).coerceIn(0.6f, 10.0f)
  }

  fun resetTransforms() {
    modelScale = 1.0f
    modelRotationDegrees = 0f
    modelPitchDegrees = 0f
    modelOffsetX = 0f
    modelOffsetY = 0f
    modelOffsetZ = 0f
    orbitPitch = 15.0f
    orbitYaw = 30.0f
    orbitDistance = 2.5f
    panX = 0.0f
    panY = 0.0f
  }

  fun destroyCurrentAsset() {
    originalRenderableMaterials.clear()
    val eng = engine ?: return
    val loader = assetLoader ?: return
    val scn = scene ?: return
    val asset = currentAsset ?: return

    val root = asset.root
    if (root != 0) {
      scn.remove(root)
    }
    scn.removeEntities(asset.entities)
    try {
      loader.destroyAsset(asset)
    } catch (e: Exception) {
      Log.e(TAG, "Error destroying asset: ${e.message}")
    }
    currentAsset = null
    currentInstance = null
    recalculateSceneMetrics()
  }

  fun clearAll() {
    clearAllExhibits()
    destroyCurrentAsset()
    lodManager.clear()
    resetTransforms()
    recalculateSceneMetrics()
  }

  fun destroy() {
    val eng = engine ?: return

    try {
      clearAllExhibits()
      destroyCurrentAsset()
      materialProvider?.destroy()
      materialProvider = null
      assetLoader?.destroy()
      assetLoader = null
      resourceLoader?.destroy()
      resourceLoader = null

      filamentDepthTexture?.let { eng.destroyTexture(it) }
      filamentDepthTexture = null

      depthOcclusionMaterialHelper.destroy(eng)

      swapChain?.let { eng.destroySwapChain(it) }
      swapChain = null
      view?.let { eng.destroyView(it) }
      view = null
      scene?.let { eng.destroyScene(it) }
      scene = null
      renderer?.let { eng.destroyRenderer(it) }
      renderer = null
      camera?.let { eng.destroyCameraComponent(it.entity) }
      camera = null

      eng.destroy()
      engine = null
      Log.i(TAG, "Filament Engine destroyed cleanly.")
    } catch (e: Exception) {
      Log.e(TAG, "Error during Filament destroy: ${e.message}", e)
    }
  }
}
