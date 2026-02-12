const fs = require('fs');
const path = require('path');
const {
  withAppBuildGradle,
  withDangerousMod,
  withMainApplication,
} = require('@expo/config-plugins');

const ORT_DEP = "implementation(\"com.microsoft.onnxruntime:onnxruntime-android:1.18.0\")";
// ONNX model bytes (IR=7, opset=11) for a tiny identity graph: y = x.
// Kept here as base64 so we don't need to commit binary files and to stay compatible with older ORT builds.
const IDENTITY_ONNX_B64 =
  "CAcSFXN0YXRpY3BsYXktc2QxLW1vYmlsZTpWChoKAXgSAXkaCElkZW50aXR5IghJZGVudGl0eRIOaWRlbnRpdHlfZ3JhcGhaEwoBeBIOCgwIARIICgIIAQoCCANiEwoBeRIOCgwIARIICgIIAQoCCANCBAoAEAs=";

function upsertOnce(text, needle, insertAfterRegex, insertion) {
  if (text.includes(needle)) return text;
  const match = text.match(insertAfterRegex);
  if (!match || match.index == null) return text + '\n' + insertion + '\n';
  const idx = match.index + match[0].length;
  return text.slice(0, idx) + '\n' + insertion + text.slice(idx);
}

function withOrtNative(config) {
  config = withAppBuildGradle(config, (cfg) => {
    if (!cfg.modResults.contents.includes(ORT_DEP)) {
      cfg.modResults.contents = upsertOnce(
        cfg.modResults.contents,
        ORT_DEP,
        /dependencies\s*\{\s*/m,
        `  ${ORT_DEP}`
      );
    }
    return cfg;
  });

  config = withMainApplication(config, (cfg) => {
    let contents = cfg.modResults.contents;

    if (!contents.includes('import com.anonymous.staticplaysd1mobile.ort.OrtPackage')) {
      contents = upsertOnce(
        contents,
        'import com.anonymous.staticplaysd1mobile.ort.OrtPackage',
        /^package\s+.+\n/m,
        'import com.anonymous.staticplaysd1mobile.ort.OrtPackage'
      );
    }

    // Add the package manually (legacy/bridge path; works with `newArchEnabled: false`).
    if (!contents.includes('add(OrtPackage())')) {
      contents = contents.replace(
        /\/\/ add\(MyReactNativePackage\(\)\)\s*\n/,
        `// add(MyReactNativePackage())\n              add(OrtPackage())\n`
      );
    }

    cfg.modResults.contents = contents;
    return cfg;
  });

  config = withDangerousMod(config, [
    'android',
    async (cfg) => {
      const projectRoot = cfg.modRequest.projectRoot;
      const androidDir = path.join(projectRoot, 'android');

      // 1) Write Kotlin native module + package
      const pkgDir = path.join(
        androidDir,
        'app',
        'src',
        'main',
        'java',
        'com',
        'anonymous',
        'staticplaysd1mobile',
        'ort'
      );
      fs.mkdirSync(pkgDir, { recursive: true });

      const modulePath = path.join(pkgDir, 'OrtModule.kt');
      const packagePath = path.join(pkgDir, 'OrtPackage.kt');

      fs.writeFileSync(
        modulePath,
        `package com.anonymous.staticplaysd1mobile.ort

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OrtSession.SessionOptions
import ai.onnxruntime.TensorInfo
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReadableArray
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import android.net.Uri
import android.util.Base64
import android.graphics.Bitmap
import java.io.File
import java.io.FileOutputStream
import java.nio.LongBuffer
import java.nio.FloatBuffer
import java.security.MessageDigest
import java.util.Random

class OrtModule(reactContext: ReactApplicationContext) : ReactContextBaseJavaModule(reactContext) {
  override fun getName(): String = "OrtNative"

  private var env: OrtEnvironment? = null
  private var sessionNnapi: OrtSession? = null
  private var sessionCpu: OrtSession? = null
  private var packRoot: String? = null
  private var textCpu: OrtSession? = null
  private var vaeCpu: OrtSession? = null
  private var unetNnapi: OrtSession? = null
  private var unetCpu: OrtSession? = null

  private val identityModelB64: String =
    "${IDENTITY_ONNX_B64}"

  private fun flattenToFloatArray(value: Any?): FloatArray {
    if (value == null) return FloatArray(0)
    return when (value) {
      is FloatArray -> value
      is FloatBuffer -> {
        val buf = value.duplicate()
        val out = FloatArray(buf.remaining())
        buf.get(out)
        out
      }
      is Array<*> -> {
        val out = ArrayList<Float>()
        fun addAny(v: Any?) {
          when (v) {
            null -> {}
            is Number -> out.add(v.toFloat())
            is FloatArray -> for (x in v) out.add(x)
            is FloatBuffer -> {
              val b = v.duplicate()
              while (b.hasRemaining()) out.add(b.get())
            }
            is Array<*> -> for (e in v) addAny(e)
            else -> throw IllegalStateException("Unsupported output element type: \${v.javaClass.name}")
          }
        }
        for (e in value) addAny(e)
        out.toFloatArray()
      }
      else -> throw IllegalStateException("Unsupported output type: \${value.javaClass.name}")
    }
  }

  private fun ensureModelFile(): File {
    val outFile = File(reactApplicationContext.cacheDir, "identity.onnx")
    // Always overwrite to avoid stale/cached models after app updates.
    if (outFile.exists()) outFile.delete()
    val bytes = Base64.decode(identityModelB64, Base64.DEFAULT)
    FileOutputStream(outFile).use { it.write(bytes) }
    return outFile
  }

  private fun sha256Hex(file: File): String {
    val md = MessageDigest.getInstance("SHA-256")
    file.inputStream().use { input ->
      val buf = ByteArray(1024 * 64)
      while (true) {
        val r = input.read(buf)
        if (r <= 0) break
        md.update(buf, 0, r)
      }
    }
    return md.digest().joinToString("") { "%02x".format(it) }
  }

  private fun uriToPath(uri: String): String {
    return if (uri.startsWith("file://")) uri.removePrefix("file://") else uri
  }

  private fun ensureDir(dir: File) {
    if (!dir.exists()) {
      dir.mkdirs()
    }
  }

  private fun isSubPath(base: File, child: File): Boolean {
    val basePath = base.canonicalFile.toPath()
    val childPath = child.canonicalFile.toPath()
    return childPath.startsWith(basePath)
  }

  private fun getEnv(): OrtEnvironment {
    val existing = env
    if (existing != null) return existing
    val created = OrtEnvironment.getEnvironment()
    env = created
    return created
  }

  private fun getSession(preferNnapi: Boolean): Pair<OrtSession, String> {
    val modelFile = ensureModelFile()
    val env = getEnv()

    if (preferNnapi) {
      val existing = sessionNnapi
      if (existing != null) return existing to "nnapi"
      try {
        val opts = SessionOptions()
        try {
          opts.addNnapi()
        } catch (_: Throwable) {
          // If this ORT build doesn't ship NNAPI, session creation will fall back below.
        }
        val created = env.createSession(modelFile.absolutePath, opts)
        sessionNnapi = created
        return created to "nnapi"
      } catch (_e: Throwable) {
        // Fall back to CPU
      }
    }

    val existingCpu = sessionCpu
    if (existingCpu != null) return existingCpu to "cpu"
    try {
      val cpu = env.createSession(modelFile.absolutePath, SessionOptions())
      sessionCpu = cpu
      return cpu to "cpu"
    } catch (e: Throwable) {
      val size = try { modelFile.length() } catch (_: Throwable) { -1L }
      val sha = try { sha256Hex(modelFile).take(16) } catch (_: Throwable) { "?" }
      throw IllegalStateException("Failed to load identity.onnx (size=$size sha256=$sha) at \${modelFile.absolutePath}: \${e.message}", e)
    }
  }

  private fun resetPackIfNeeded(packDir: String) {
    if (packRoot == packDir) return
    packRoot = packDir
    try { textCpu?.close() } catch (_: Throwable) {}
    try { vaeCpu?.close() } catch (_: Throwable) {}
    try { unetNnapi?.close() } catch (_: Throwable) {}
    try { unetCpu?.close() } catch (_: Throwable) {}
    textCpu = null
    vaeCpu = null
    unetNnapi = null
    unetCpu = null
  }

  private fun getTextEncoder(packDir: String): OrtSession {
    resetPackIfNeeded(packDir)
    val existing = textCpu
    if (existing != null) return existing
    val env = getEnv()
    val path = File(packDir, "text_encoder.onnx").absolutePath
    val s = env.createSession(path, SessionOptions())
    textCpu = s
    return s
  }

  private fun getVaeDecoder(packDir: String): OrtSession {
    resetPackIfNeeded(packDir)
    val existing = vaeCpu
    if (existing != null) return existing
    val env = getEnv()
    val path = File(packDir, "vae_decoder.onnx").absolutePath
    val s = env.createSession(path, SessionOptions())
    vaeCpu = s
    return s
  }

  private fun getUnet(packDir: String, preferNnapi: Boolean): Pair<OrtSession, String> {
    resetPackIfNeeded(packDir)
    val env = getEnv()
    val modelPath = File(packDir, "unet.onnx").absolutePath

    val existingNnapi = unetNnapi
    val existingCpu = unetCpu
    if (!preferNnapi) {
      if (existingCpu != null) return existingCpu to "cpu"
    } else {
      if (existingNnapi != null) return existingNnapi to "nnapi"
      if (existingCpu != null) return existingCpu to "cpu"
    }

    if (!preferNnapi) {
      val s = env.createSession(modelPath, SessionOptions())
      unetCpu = s
      return s to "cpu"
    }

    try {
      val opts = SessionOptions()
      try { opts.addNnapi() } catch (_: Throwable) {}
      val s = env.createSession(modelPath, opts)
      unetNnapi = s
      return s to "nnapi"
    } catch (_: Throwable) {
      val s = env.createSession(modelPath, SessionOptions())
      unetCpu = s
      return s to "cpu"
    }
  }

  private fun readInt64Ids(ids: ReadableArray, expected: Int): LongArray {
    if (ids.size() != expected) throw IllegalArgumentException("Expected $expected ids, got \${ids.size()}")
    return LongArray(expected) { i -> ids.getDouble(i).toLong() }
  }

  private fun gaussianLatents(rng: Random, count: Int): FloatArray {
    val out = FloatArray(count)
    var i = 0
    while (i < count) {
      // Box-Muller via nextGaussian
      out[i] = rng.nextGaussian().toFloat()
      i += 1
    }
    return out
  }

  private fun alphasCumprod1000(): DoubleArray {
    val betas = DoubleArray(1000)
    val betaStart = 0.00085
    val betaEnd = 0.012
    val s0 = kotlin.math.sqrt(betaStart)
    val s1 = kotlin.math.sqrt(betaEnd)
    for (i in 0 until 1000) {
      val t = i.toDouble() / 999.0
      val s = s0 + (s1 - s0) * t
      betas[i] = s * s
    }
    val alphasCum = DoubleArray(1000)
    var prod = 1.0
    for (i in 0 until 1000) {
      val alpha = 1.0 - betas[i]
      prod *= alpha
      alphasCum[i] = prod
    }
    return alphasCum
  }

  private fun ddimTimesteps(steps: Int): IntArray {
    val stride = 1000 / steps
    return IntArray(steps) { i -> 999 - i * stride }
  }

  private fun floatArraySub(a: FloatArray, b: FloatArray, out: FloatArray) {
    for (i in out.indices) out[i] = a[i] - b[i]
  }
  private fun floatArrayAddScaled(a: FloatArray, b: FloatArray, scale: Float, out: FloatArray) {
    for (i in out.indices) out[i] = a[i] + (b[i] * scale)
  }
  private fun floatArrayMulScalar(a: FloatArray, s: Float, out: FloatArray) {
    for (i in out.indices) out[i] = a[i] * s
  }

  private fun decodeToPng(packDir: String, latents: FloatArray, outPath: String) {
    val vae = getVaeDecoder(packDir)
    val env = getEnv()
    val scale = 1.0f / 0.18215f
    val scaled = FloatArray(latents.size)
    floatArrayMulScalar(latents, scale, scaled)

    val latentTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(scaled), longArrayOf(1, 4, 64, 64))
    latentTensor.use { lt ->
      val results = vae.run(mapOf("latent_sample" to lt))
      results.use {
        val out = flattenToFloatArray(results[0].value)
        val w = 512
        val h = 512
        val hw = w * h
        val pixels = IntArray(hw)
        for (i in 0 until hw) {
          val r = out[i]
          val g = out[hw + i]
          val b = out[2 * hw + i]
          fun toU8(x: Float): Int {
            val v = ((x / 2.0f) + 0.5f) * 255.0f
            val c = if (v < 0f) 0 else if (v > 255f) 255 else v.toInt()
            return c
          }
          val rr = toU8(r)
          val gg = toU8(g)
          val bb = toU8(b)
          pixels[i] = (0xFF shl 24) or (rr shl 16) or (gg shl 8) or bb
        }
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        bmp.setPixels(pixels, 0, w, 0, 0, w, h)

        val outFile = File(uriToPath(outPath))
        outFile.parentFile?.let { ensureDir(it) }
        FileOutputStream(outFile).use { fos ->
          bmp.compress(Bitmap.CompressFormat.PNG, 100, fos)
        }
      }
    }
  }

  @ReactMethod
  fun generateSd15FromIds(options: ReadableMap, promise: Promise) {
    try {
      val packDir = options.getString("packDir") ?: throw IllegalArgumentException("packDir is required")
      val condIdsArr = options.getArray("condIds") ?: throw IllegalArgumentException("condIds is required")
      val uncondIdsArr = options.getArray("uncondIds") ?: throw IllegalArgumentException("uncondIds is required")
      val steps = if (options.hasKey("steps")) options.getInt("steps") else 20
      val guidance = if (options.hasKey("guidance")) options.getDouble("guidance").toFloat() else 7.5f
      val seed = if (options.hasKey("seed")) options.getDouble("seed").toLong() else -1L
      val outPath = options.getString("outPath") ?: throw IllegalArgumentException("outPath is required")

      val condIds = readInt64Ids(condIdsArr, 77)
      val uncondIds = readInt64Ids(uncondIdsArr, 77)

      val rng = if (seed == -1L) Random() else Random(seed)

      val preferNnapi = if (options.hasKey("preferNnapi")) options.getBoolean("preferNnapi") else false

      val text = getTextEncoder(packDir)
      val (unet, unetProvider) = getUnet(packDir, preferNnapi)
      val env = getEnv()

      fun encode(ids: LongArray): FloatArray {
        val t = OnnxTensor.createTensor(env, LongBuffer.wrap(ids), longArrayOf(1, 77))
        t.use { it2 ->
          val r = text.run(mapOf("input_ids" to it2))
          r.use {
            return flattenToFloatArray(r[0].value)
          }
        }
      }

      val condEmb = encode(condIds) // [1,77,768]
      val uncondEmb = encode(uncondIds)

      val latentCount = 4 * 64 * 64
      var latents = gaussianLatents(rng, latentCount)

      val alphasCum = alphasCumprod1000()
      val timesteps = ddimTimesteps(steps)

      val tmp = FloatArray(latentCount)
      val tmp2 = FloatArray(latentCount)

      fun unetEps(emb: FloatArray, t: Long, sample: FloatArray): FloatArray {
        val sampleTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(sample), longArrayOf(1, 4, 64, 64))
        val tTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(t)), longArrayOf())
        val embTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(emb), longArrayOf(1, 77, 768))
        sampleTensor.use { st ->
          tTensor.use { tt ->
            embTensor.use { et ->
              val r = unet.run(mapOf("sample" to st, "timestep" to tt, "encoder_hidden_states" to et))
              r.use { rr ->
                return flattenToFloatArray(rr[0].value)
              }
            }
          }
        }
      }

      for (i in 0 until timesteps.size) {
        val t = timesteps[i]
        val tPrev = if (i == timesteps.size - 1) -1 else timesteps[i + 1]

        val epsUncond = unetEps(uncondEmb, t.toLong(), latents)
        val epsText = unetEps(condEmb, t.toLong(), latents)
        // eps = epsUncond + guidance * (epsText - epsUncond)
        floatArraySub(epsText, epsUncond, tmp)
        floatArrayAddScaled(epsUncond, tmp, guidance, tmp2)

        val alphaT = alphasCum[t]
        val alphaPrev = if (tPrev == -1) 1.0 else alphasCum[tPrev]
        val sqrtAlphaT = kotlin.math.sqrt(alphaT).toFloat()
        val sqrtOneMinusAlphaT = kotlin.math.sqrt(1.0 - alphaT).toFloat()
        val sqrtAlphaPrev = kotlin.math.sqrt(alphaPrev).toFloat()
        val sqrtOneMinusAlphaPrev = kotlin.math.sqrt(1.0 - alphaPrev).toFloat()

        // pred_x0 = (x_t - sqrt(1-a_t)*eps)/sqrt(a_t)
        for (k in 0 until latentCount) {
          val predX0 = (latents[k] - sqrtOneMinusAlphaT * tmp2[k]) / sqrtAlphaT
          val dir = sqrtOneMinusAlphaPrev * tmp2[k]
          latents[k] = sqrtAlphaPrev * predX0 + dir
        }
      }

      decodeToPng(packDir, latents, outPath)

      val out = Arguments.createMap()
      out.putString("provider", unetProvider)
      out.putString("path", uriToPath(outPath))
      out.putInt("steps", steps)
      promise.resolve(out)
    } catch (e: Throwable) {
      promise.reject("ORT_NATIVE_ERROR", e.message, e)
    }
  }

  @ReactMethod
  fun runIdentity(input: ReadableArray, promise: Promise) {
    try {
      val floats = FloatArray(input.size()) { i -> (input.getDouble(i)).toFloat() }
      val (session, provider) = getSession(true)

      val tensor = OnnxTensor.createTensor(getEnv(), FloatBuffer.wrap(floats), longArrayOf(1, floats.size.toLong()))
      tensor.use {
        val results = session.run(mapOf("x" to it))
        results.use {
          val out = flattenToFloatArray(results[0].value)
          val obj = Arguments.createMap()
          obj.putString("provider", provider)
          val arr = Arguments.createArray()
          for (v in out) arr.pushDouble(v.toDouble())
          obj.putArray("output", arr)
          promise.resolve(obj)
        }
      }
    } catch (e: Throwable) {
      promise.reject("ORT_NATIVE_ERROR", e.message, e)
    }
  }

  @ReactMethod
  fun getExternalFilesDirPath(promise: Promise) {
    try {
      val dir = reactApplicationContext.getExternalFilesDir(null)
      if (dir == null) {
        promise.reject("ORT_NATIVE_ERROR", "External files dir is not available")
        return
      }
      promise.resolve(dir.absolutePath)
    } catch (e: Throwable) {
      promise.reject("ORT_NATIVE_ERROR", e.message, e)
    }
  }

  @ReactMethod
  fun statPath(fileUri: String, promise: Promise) {
    try {
      val res = Arguments.createMap()
      if (fileUri.startsWith("content://")) {
        // Best-effort: scoped storage URI (size is optional and may be unknown).
        val uri = Uri.parse(fileUri)
        val stream = reactApplicationContext.contentResolver.openInputStream(uri)
        stream?.close()
        res.putBoolean("exists", stream != null)
        res.putDouble("size", -1.0)
        promise.resolve(res)
        return
      }

      val path = uriToPath(fileUri)
      val f = File(path)
      res.putBoolean("exists", f.exists())
      res.putDouble("size", if (f.exists()) f.length().toDouble() else 0.0)
      res.putString("path", f.absolutePath)
      promise.resolve(res)
    } catch (e: Throwable) {
      promise.reject("ORT_NATIVE_ERROR", e.message, e)
    }
  }

  @ReactMethod
  fun readTextFile(fileUri: String, promise: Promise) {
    try {
      val res = if (fileUri.startsWith("content://")) {
        val uri = Uri.parse(fileUri)
        reactApplicationContext.contentResolver.openInputStream(uri)
          ?: throw IllegalStateException("Unable to open content URI: $fileUri")
      } else {
        val path = uriToPath(fileUri)
        java.io.FileInputStream(path)
      }
      res.use { input ->
        val text = input.bufferedReader(Charsets.UTF_8).readText()
        promise.resolve(text)
      }
    } catch (e: Throwable) {
      promise.reject("ORT_NATIVE_ERROR", e.message, e)
    }
  }

  private fun typeInfoToMap(typeInfo: Any?): com.facebook.react.bridge.WritableMap {
    val m = Arguments.createMap()
    if (typeInfo == null) return m
    if (typeInfo is TensorInfo) {
      m.putString("type", typeInfo.type.toString())
      val shapeArr = Arguments.createArray()
      for (d in typeInfo.shape) shapeArr.pushDouble(d.toDouble())
      m.putArray("shape", shapeArr)
    } else {
      m.putString("type", (typeInfo as Any).javaClass.name)
    }
    return m
  }

  @ReactMethod
  fun inspectModel(fileUri: String, promise: Promise) {
    try {
      val path = uriToPath(fileUri)
      val env = getEnv()
      val (session, provider) = try {
        val opts = SessionOptions()
        try { opts.addNnapi() } catch (_: Throwable) {}
        env.createSession(path, opts) to "nnapi"
      } catch (_: Throwable) {
        env.createSession(path, SessionOptions()) to "cpu"
      }

      session.use {
        val out = Arguments.createMap()
        out.putString("provider", provider)

        val inputsArr = Arguments.createArray()
        for ((name, nodeInfo) in session.inputInfo) {
          val mi = Arguments.createMap()
          mi.putString("name", name)
          mi.putMap("info", typeInfoToMap(nodeInfo.info))
          inputsArr.pushMap(mi)
        }
        out.putArray("inputs", inputsArr)

        val outputsArr = Arguments.createArray()
        for ((name, nodeInfo) in session.outputInfo) {
          val mo = Arguments.createMap()
          mo.putString("name", name)
          mo.putMap("info", typeInfoToMap(nodeInfo.info))
          outputsArr.pushMap(mo)
        }
        out.putArray("outputs", outputsArr)

        promise.resolve(out)
      }
    } catch (e: Throwable) {
      promise.reject("ORT_NATIVE_ERROR", e.message, e)
    }
  }

  @ReactMethod
  fun inspectModelCpu(fileUri: String, promise: Promise) {
    try {
      val path = uriToPath(fileUri)
      val env = getEnv()
      val session = env.createSession(path, SessionOptions())
      session.use {
        val out = Arguments.createMap()
        out.putString("provider", "cpu")

        val inputsArr = Arguments.createArray()
        for ((name, nodeInfo) in session.inputInfo) {
          val mi = Arguments.createMap()
          mi.putString("name", name)
          mi.putMap("info", typeInfoToMap(nodeInfo.info))
          inputsArr.pushMap(mi)
        }
        out.putArray("inputs", inputsArr)

        val outputsArr = Arguments.createArray()
        for ((name, nodeInfo) in session.outputInfo) {
          val mo = Arguments.createMap()
          mo.putString("name", name)
          mo.putMap("info", typeInfoToMap(nodeInfo.info))
          outputsArr.pushMap(mo)
        }
        out.putArray("outputs", outputsArr)

        promise.resolve(out)
      }
    } catch (e: Throwable) {
      promise.reject("ORT_NATIVE_ERROR", e.message, e)
    }
  }

  @ReactMethod
  fun unzip(zipFileUri: String, destDirUri: String, promise: Promise) {
    try {
      val destDir = File(uriToPath(destDirUri))
      ensureDir(destDir)

      val inputStream = if (zipFileUri.startsWith("content://")) {
        val uri = Uri.parse(zipFileUri)
        reactApplicationContext.contentResolver.openInputStream(uri)
          ?: throw IllegalStateException("Unable to open content URI: $zipFileUri")
      } else {
        val zipPath = uriToPath(zipFileUri)
        java.io.FileInputStream(zipPath)
      }

      val zis = java.util.zip.ZipInputStream(java.io.BufferedInputStream(inputStream))
      zis.use { stream ->
        var entry = stream.nextEntry
        var count = 0
        val buf = ByteArray(1024 * 64)
        while (entry != null) {
          val outFile = File(destDir, entry.name)
          if (!isSubPath(destDir, outFile)) {
            throw IllegalStateException("Zip entry escapes destination: \${entry.name}")
          }
          if (entry.isDirectory) {
            ensureDir(outFile)
          } else {
            outFile.parentFile?.let { ensureDir(it) }
            java.io.FileOutputStream(outFile).use { out ->
              while (true) {
                val read = stream.read(buf)
                if (read <= 0) break
                out.write(buf, 0, read)
              }
            }
            count += 1
          }
          stream.closeEntry()
          entry = stream.nextEntry
        }
        val res = Arguments.createMap()
        res.putString("destPath", destDir.absolutePath)
        res.putInt("files", count)
        promise.resolve(res)
      }
    } catch (e: Throwable) {
      promise.reject("ORT_NATIVE_ERROR", e.message, e)
    }
  }
}
`
      );

      fs.writeFileSync(
        packagePath,
        `package com.anonymous.staticplaysd1mobile.ort

import com.facebook.react.ReactPackage
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.uimanager.ViewManager

class OrtPackage : ReactPackage {
  override fun createNativeModules(reactContext: ReactApplicationContext): List<NativeModule> {
    return listOf(OrtModule(reactContext))
  }

  override fun createViewManagers(reactContext: ReactApplicationContext): List<ViewManager<*, *>> {
    return emptyList()
  }
}
`
      );

      // 2) Copy tiny identity model into Android assets
      const assetDstDir = path.join(androidDir, 'app', 'src', 'main', 'assets', 'models');
      const assetDst = path.join(assetDstDir, 'identity.onnx');
      fs.mkdirSync(assetDstDir, { recursive: true });
      const bytes = Buffer.from(IDENTITY_ONNX_B64, 'base64');
      fs.writeFileSync(assetDst, bytes);

      return cfg;
    },
  ]);

  return config;
}

module.exports = withOrtNative;
