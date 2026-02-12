import React, { useEffect, useMemo, useState } from 'react';
import { StatusBar } from 'expo-status-bar';
import { useKeepAwake } from 'expo-keep-awake';
import { Asset } from 'expo-asset';
import { ActivityIndicator, Pressable, SafeAreaView, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import { NativeModules } from 'react-native';
import * as DocumentPicker from 'expo-document-picker';
import * as FileSystem from 'expo-file-system/legacy';
import { createClipTokenizer } from './clipTokenizer';

const BUILD_ID = '2026-02-12-v2-speed-exp-002';
const QUICK_STEPS = 4;
const QUALITY_STEPS = 20;

const GUIDE_STEPS = [
  { key: 'tapLaunch', label: 'Tap to confirm launch' },
  { key: 'showExternalPath', label: 'Show app external pack path' },
  { key: 'unzipExternal', label: 'Unzip SD15 pack from external' },
  { key: 'loadTokenizer', label: 'Load tokenizer (bundled)' },
  { key: 'testNative', label: 'Test native ONNX (NNAPI -> CPU)' },
  { key: 'inspectUnet', label: 'Inspect pack UNet' },
  { key: 'inspectText', label: 'Inspect pack text_encoder' },
  { key: 'inspectVae', label: 'Inspect pack vae_decoder' },
  { key: 'generateQuick', label: `Generate 512x512 (quick ${QUICK_STEPS})` },
  { key: 'generateQuality', label: `Generate 512x512 (quality ${QUALITY_STEPS})` },
  { key: 'backendToggle', label: 'Backend: NNAPI / CPU' },
  { key: 'importOnnx', label: 'Import ONNX files (offline)' },
  { key: 'inspectImported', label: 'Inspect first imported model' },
  { key: 'importPackZip', label: 'Import SD15 pack (.zip)' },
] as const;

type GuideStepKey = (typeof GUIDE_STEPS)[number]['key'];

export default function App() {
  useKeepAwake();
  const [taps, setTaps] = useState(0);
  const [busy, setBusy] = useState(false);
  const [provider, setProvider] = useState<string>('-');
  const [output, setOutput] = useState<string>('-');
  const [err, setErr] = useState<string>('');
  const [status, setStatus] = useState<string>('Idle');
  const [lastExportPath, setLastExportPath] = useState<string>('-');
  const [importedFiles, setImportedFiles] = useState<string[]>([]);
  const [packDir, setPackDir] = useState<string>('');
  const [packFiles, setPackFiles] = useState<number>(0);
  const [externalDir, setExternalDir] = useState<string>('');
  const [prompt, setPrompt] = useState<string>('A photo of a lion in the wild, ultra realistic');
  const [tokenizerReady, setTokenizerReady] = useState(false);
  const [preferNnapi, setPreferNnapi] = useState(true);
  const [guidedMode, setGuidedMode] = useState(true);
  const [guideIndex, setGuideIndex] = useState(0);

  const canRun = useMemo(() => true, []);
  const guideDone = guideIndex >= GUIDE_STEPS.length;
  const currentGuideStep = guideDone ? null : GUIDE_STEPS[guideIndex];
  const showGuideButton = (key: GuideStepKey) => !guidedMode || guideDone || currentGuideStep?.key === key;
  const completeGuideStep = (key: GuideStepKey) => {
    if (!guidedMode || guideDone) return;
    if (currentGuideStep?.key !== key) return;
    setGuideIndex((n) => n + 1);
  };

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const mod = (NativeModules as any)?.OrtNative;
        if (!mod?.getExternalFilesDirPath) return;
        const p = await mod.getExternalFilesDirPath();
        if (cancelled) return;
        const ext = String(p || '');
        if (!ext) return;
        setExternalDir((prev) => prev || ext);

        if (packDir) return;
        if (!mod?.statPath) return;
        const candidate = `${ext.replace(/\/+$/, '')}/sd15-onnx/sd15-onnx`;
        const st = await mod.statPath(`file://${candidate}/unet.onnx`);
        if (cancelled) return;
        if (st?.exists) setPackDir((prev) => prev || candidate);
      } catch {
        // Non-fatal; user can still set paths via buttons.
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const packFileUri = (filename: string) => {
    const base = packDir.startsWith('file://') ? packDir.slice('file://'.length) : packDir;
    const baseNoSlash = base.replace(/\/+$/, '');
    return `file://${baseNoSlash}/${filename}`;
  };

  const tokenizer = useMemo(() => {
    if (!tokenizerReady) return null;
    return (globalThis as any).__sp_clip_tokenizer__ ?? null;
  }, [tokenizerReady]);

  const yieldToUi = async () => new Promise<void>((r) => setTimeout(r, 0));

  const runGenerate = async (steps: number, forceCpu: boolean = false, lowMemoryMode: boolean = true) => {
    setBusy(true);
    setErr('');
    setProvider('-');
    setOutput('-');
    try {
      if (!packDir) throw new Error('Pack not set. Tap "Unzip SD15 pack from external" (once) if needed.');
      if (!externalDir) throw new Error('External dir is not set. Tap "Show app external pack path" first.');
      if (!tokenizer) throw new Error('Tokenizer not loaded. Tap "Load tokenizer (bundled)" first.');
      const mod = (NativeModules as any)?.OrtNative;
      if (!mod?.generateSd15FromIds) throw new Error('Native module OrtNative.generateSd15FromIds is not available (rebuild required).');

      setStatus('Tokenizing...');
      await yieldToUi();
      const condIds = tokenizer.encode(prompt, 77);
      const uncondIds = tokenizer.encode('', 77);

      const outPath = `file://${externalDir.replace(/\/+$/, '')}/sd_out_${Date.now()}.png`;
      setStatus(`Generating... (${steps} steps)`);
      await yieldToUi();
      const useNnapi = forceCpu ? false : preferNnapi;
      const runOnce = async (nnapi: boolean, lowMem: boolean) =>
        mod.generateSd15FromIds({
          packDir,
          condIds,
          uncondIds,
          steps,
          guidance: 7.5,
          seed: -1,
          outPath,
          preferNnapi: nnapi,
          lowMemoryMode: lowMem,
        });

      let res: any;
      try {
        res = await runOnce(useNnapi, lowMemoryMode);
      } catch (firstErr: any) {
        // In v2, if NNAPI path fails without killing process, retry CPU automatically.
        if (useNnapi) {
          setStatus('NNAPI failed, retrying CPU...');
          await yieldToUi();
          res = await runOnce(false, true);
        } else {
          throw firstErr;
        }
      }
      try {
        if (mod?.exportImageToGallery && res?.path) {
          const fileName = String(res.path).split('/').pop() || `sd_out_${Date.now()}.png`;
          const exported = await mod.exportImageToGallery(String(res.path), fileName);
          setLastExportPath(String(exported || `Pictures/Staticplay/${fileName}`));
        }
      } catch (exportErr: any) {
        // Non-fatal: image remains in app external storage even if gallery export fails.
        const msg = String(exportErr?.message ?? exportErr ?? '').trim();
        if (msg) {
          setErr((prev) => (prev ? `${prev}\nExport: ${msg}` : `Export: ${msg}`));
        }
      }
      setProvider(String(res?.provider ?? 'unknown'));
      setOutput(JSON.stringify(res));
      setStatus('Done');
    } catch (e: any) {
      const msg = String(e?.message ?? e ?? '').trim();
      setErr(msg || 'Generation failed (unknown error)');
      setStatus('Error');
    } finally {
      setBusy(false);
    }
  };

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView contentContainerStyle={styles.container} keyboardShouldPersistTaps="handled">
        <Text style={styles.title}>Staticplay</Text>
        <Text style={styles.sub}>Offline-first SD1 on-device (Android) - {BUILD_ID}</Text>

        <View style={styles.card}>
          <Text style={styles.label}>Guided mode</Text>
          <Text style={styles.value}>
            {guidedMode
              ? guideDone
                ? 'Done. All guided steps completed.'
                : `Step ${guideIndex + 1}/${GUIDE_STEPS.length}: ${currentGuideStep?.label}`
              : 'Off. Showing all buttons.'}
          </Text>
          <View style={styles.row}>
            <Pressable style={[styles.smallBtn, styles.smallBtnPrimary]} onPress={() => setGuidedMode((v) => !v)}>
              <Text style={styles.smallBtnText}>{guidedMode ? 'Show all buttons' : 'Enable guided mode'}</Text>
            </Pressable>
            <Pressable
              style={[styles.smallBtn, styles.smallBtnGhost]}
              onPress={() => {
                setGuideIndex(0);
                setGuidedMode(true);
              }}
            >
              <Text style={styles.smallBtnText}>Restart guide</Text>
            </Pressable>
          </View>
        </View>

        <View style={styles.card}>
          <Text style={styles.label}>Prompt</Text>
          <TextInput
            value={prompt}
            onChangeText={setPrompt}
            placeholder="Describe an image..."
            placeholderTextColor="#5b6f92"
            style={styles.input}
            multiline
          />
          <Text style={[styles.value, { marginTop: 8, color: '#8aa0c4' }]}>
            512x512 - quick {QUICK_STEPS} / quality {QUALITY_STEPS} - guidance 7.5
          </Text>
        </View>

        {showGuideButton('tapLaunch') ? (
          <Pressable
            style={[styles.btn, !canRun ? styles.btnDisabled : null]}
            disabled={!canRun}
            onPress={() => {
              setTaps((n) => n + 1);
              completeGuideStep('tapLaunch');
            }}
          >
            <Text style={styles.btnText}>Tap to confirm launch ({taps})</Text>
          </Pressable>
        ) : null}

        {showGuideButton('backendToggle') ? (
          <Pressable
            style={[styles.btn, busy ? styles.btnDisabled : null]}
            disabled={busy}
            onPress={() => {
              setPreferNnapi((v) => !v);
              completeGuideStep('backendToggle');
            }}
          >
            <Text style={styles.btnText}>{preferNnapi ? 'Backend: NNAPI (experimental)' : 'Backend: CPU (safe)'}</Text>
          </Pressable>
        ) : null}

        {showGuideButton('loadTokenizer') ? (
          <Pressable
            style={[styles.btn, busy ? styles.btnDisabled : null]}
            disabled={busy}
            onPress={async () => {
              setBusy(true);
              setErr('');
              let loaded = false;
              try {
                setStatus('Loading tokenizer...');
                const vocabAsset = Asset.fromModule(require('./assets/tokenizer/vocab.json.txt'));
                const mergesAsset = Asset.fromModule(require('./assets/tokenizer/merges.txt'));
                await vocabAsset.downloadAsync();
                await mergesAsset.downloadAsync();
                const vocabUri = vocabAsset.localUri || vocabAsset.uri;
                const mergesUri = mergesAsset.localUri || mergesAsset.uri;
                if (!vocabUri) throw new Error('Tokenizer vocab asset missing local URI.');
                if (!mergesUri) throw new Error('Tokenizer merges asset missing local URI.');
                const mod = (NativeModules as any)?.OrtNative;
                const readText = mod?.readTextFile
                  ? async (uri: string) => String(await mod.readTextFile(uri))
                  : async (uri: string) => String(await FileSystem.readAsStringAsync(uri));
                let vocabJson = '';
                let mergesTxt = '';
                try {
                  vocabJson = await readText(vocabUri);
                  mergesTxt = await readText(mergesUri);
                } catch {
                  // Fallback: read tokenizer from app external storage if bundled read fails.
                  if (!externalDir) throw new Error('Tokenizer read failed and external dir is not set.');
                  const extBase = externalDir.replace(/\/+$/, '');
                  vocabJson = await readText(`file://${extBase}/tokenizer/vocab.json.txt`);
                  mergesTxt = await readText(`file://${extBase}/tokenizer/merges.txt`);
                }
                (globalThis as any).__sp_clip_tokenizer__ = createClipTokenizer({ vocabJson, mergesTxt });
                setTokenizerReady(true);
                setStatus('OK');
                loaded = true;
              } catch (e: any) {
                const msg = String(e?.message ?? e ?? '').trim();
                setErr(msg || 'Tokenizer load failed (unknown error)');
                setStatus('Error');
              } finally {
                setBusy(false);
                if (loaded) completeGuideStep('loadTokenizer');
              }
            }}
          >
            <Text style={styles.btnText}>{busy ? 'Working...' : tokenizerReady ? 'Tokenizer loaded' : 'Load tokenizer (bundled)'}</Text>
          </Pressable>
        ) : null}

        {showGuideButton('generateQuick') ? (
          <Pressable
            style={[styles.btn, busy ? styles.btnDisabled : null]}
            disabled={busy}
            onPress={async () => {
              // v2 quick path: NNAPI-first and less memory-constrained for speed.
              await runGenerate(QUICK_STEPS, false, false);
              completeGuideStep('generateQuick');
            }}
          >
            <Text style={styles.btnText}>{busy ? 'Working...' : `Generate 512x512 (quick ${QUICK_STEPS})`}</Text>
          </Pressable>
        ) : null}

        {showGuideButton('generateQuality') ? (
          <Pressable
            style={[styles.btn, busy ? styles.btnDisabled : null]}
            disabled={busy}
            onPress={async () => {
              // v2 quality path: keep stable settings.
              await runGenerate(QUALITY_STEPS, true, true);
              completeGuideStep('generateQuality');
            }}
          >
            <Text style={styles.btnText}>{busy ? 'Working...' : `Generate 512x512 (quality ${QUALITY_STEPS})`}</Text>
          </Pressable>
        ) : null}

        {showGuideButton('testNative') ? (
          <Pressable
            style={[styles.btn, busy ? styles.btnDisabled : null]}
            disabled={busy}
            onPress={async () => {
              setBusy(true);
              setErr('');
              setProvider('-');
              setOutput('-');
              setStatus('Running identity ONNX...');
              try {
                const mod = (NativeModules as any)?.OrtNative;
                if (!mod?.runIdentity) throw new Error('Native module OrtNative is not available (rebuild required).');
                const res = await mod.runIdentity([1, 2, 3]);
                setProvider(String(res?.provider ?? 'unknown'));
                const outArr = res?.output ?? [];
                setOutput(JSON.stringify(outArr));
                setStatus('OK');
              } catch (e: any) {
                setErr(e?.message || String(e));
                setStatus('Error');
              } finally {
                setBusy(false);
                completeGuideStep('testNative');
              }
            }}
          >
            <Text style={styles.btnText}>{busy ? 'Running...' : 'Test native ONNX (NNAPI -> CPU)'}</Text>
          </Pressable>
        ) : null}

        {showGuideButton('importOnnx') ? (
          <Pressable
            style={[styles.btn, busy ? styles.btnDisabled : null]}
            disabled={busy}
            onPress={async () => {
              setBusy(true);
              setErr('');
              setStatus('Picking ONNX files...');
              try {
                const res = await DocumentPicker.getDocumentAsync({
                  multiple: true,
                  type: ['application/octet-stream', '*/*'],
                  copyToCacheDirectory: true,
                });
                if (res.canceled) return;

                const destDir = `${FileSystem.documentDirectory}models/sd1/`;
                await FileSystem.makeDirectoryAsync(destDir, { intermediates: true });

                const saved: string[] = [];
                for (const asset of res.assets) {
                  const name = (asset.name || 'model.onnx').replace(/[^\w.\-]+/g, '_');
                  const to = `${destDir}${name}`;
                  await FileSystem.copyAsync({ from: asset.uri, to });
                  saved.push(to);
                }
                setImportedFiles((prev) => Array.from(new Set([...saved, ...prev])));
                setStatus(`Imported ${saved.length} file(s)`);
              } catch (e: any) {
                setErr(e?.message || String(e));
                setStatus('Error');
              } finally {
                setBusy(false);
                completeGuideStep('importOnnx');
              }
            }}
          >
            <Text style={styles.btnText}>{busy ? 'Working...' : 'Import ONNX files (offline)'}</Text>
          </Pressable>
        ) : null}

        {showGuideButton('inspectImported') ? (
          <Pressable
            style={[styles.btn, busy ? styles.btnDisabled : null]}
            disabled={busy}
            onPress={async () => {
              setBusy(true);
              setErr('');
              setProvider('-');
              setOutput('-');
              setStatus('Inspecting...');
              try {
                if (importedFiles.length === 0) throw new Error('No imported ONNX files yet. Tap "Import ONNX files (offline)" first.');
                const mod = (NativeModules as any)?.OrtNative;
                if (!mod?.inspectModel) throw new Error('Native module OrtNative.inspectModel is not available (rebuild required).');
                const target = importedFiles[0];
                const info = await mod.inspectModel(target);
                setProvider(String(info?.provider ?? 'unknown'));
                setOutput(JSON.stringify({ file: target.split('/').pop(), inputs: info?.inputs, outputs: info?.outputs }));
                setStatus('OK');
              } catch (e: any) {
                setErr(e?.message || String(e));
                setStatus('Error');
              } finally {
                setBusy(false);
                completeGuideStep('inspectImported');
              }
            }}
          >
            <Text style={styles.btnText}>Inspect first imported model</Text>
          </Pressable>
        ) : null}

        {showGuideButton('importPackZip') ? (
          <Pressable
            style={[styles.btn, busy ? styles.btnDisabled : null]}
            disabled={busy}
            onPress={async () => {
              setBusy(true);
              setErr('');
              setStatus('Picking zip...');
              try {
                const res = await DocumentPicker.getDocumentAsync({
                  multiple: false,
                  type: ['application/zip', 'application/octet-stream', '*/*'],
                  copyToCacheDirectory: false,
                });
                if (res.canceled) return;
                const asset = res.assets[0];
                const zipUri = asset.uri;

                const destDir = `${FileSystem.documentDirectory}packs/sd15-onnx/`;
                await FileSystem.makeDirectoryAsync(destDir, { intermediates: true });

                const mod = (NativeModules as any)?.OrtNative;
                if (!mod?.unzip) throw new Error('Native module OrtNative.unzip is not available (rebuild required).');
                if (mod?.statPath) {
                  const st = await mod.statPath(zipUri);
                  if (st && st.exists === false) throw new Error(`Zip not found: ${String(st.path || zipUri)}`);
                }
                setStatus('Unzipping... (this can take minutes)');
                const out = await mod.unzip(zipUri, destDir);
                {
                  const base = String(out?.destPath || destDir);
                  const normalized = base.startsWith('file://') ? base.slice('file://'.length) : base;
                  setPackDir(`${normalized.replace(/\/+$/, '')}/sd15-onnx`);
                }
                setPackFiles(Number(out?.files || 0));
                setStatus(`Unzipped ${Number(out?.files || 0)} file(s)`);
              } catch (e: any) {
                setErr(e?.message || String(e));
                setStatus('Error');
              } finally {
                setBusy(false);
                completeGuideStep('importPackZip');
              }
            }}
          >
            <Text style={styles.btnText}>{busy ? 'Working...' : 'Import SD15 pack (.zip)'}</Text>
          </Pressable>
        ) : null}

        {showGuideButton('inspectUnet') ? (
          <Pressable
            style={[styles.btn, busy ? styles.btnDisabled : null]}
            disabled={busy}
            onPress={async () => {
              setBusy(true);
              setErr('');
              setProvider('-');
              setOutput('-');
              setStatus('Inspecting UNet...');
              try {
                if (!packDir) throw new Error('Pack not set. Tap "Import SD15 pack (.zip)" or "Unzip SD15 pack from external" first.');
                const mod = (NativeModules as any)?.OrtNative;
                if (!mod?.inspectModel) throw new Error('Native module OrtNative.inspectModel is not available (rebuild required).');
                const info = await mod.inspectModel(packFileUri('unet.onnx'));
                setProvider(String(info?.provider ?? 'unknown'));
                setOutput(JSON.stringify({ file: 'unet.onnx', inputs: info?.inputs, outputs: info?.outputs }));
                setStatus('OK');
              } catch (e: any) {
                setErr(e?.message || String(e));
                setStatus('Error');
              } finally {
                setBusy(false);
                completeGuideStep('inspectUnet');
              }
            }}
          >
            <Text style={styles.btnText}>Inspect pack UNet</Text>
          </Pressable>
        ) : null}

        {showGuideButton('inspectText') ? (
          <Pressable
            style={[styles.btn, busy ? styles.btnDisabled : null]}
            disabled={busy}
            onPress={async () => {
              setBusy(true);
              setErr('');
              setProvider('-');
              setOutput('-');
              setStatus('Inspecting text_encoder (CPU)...');
              try {
                if (!packDir) throw new Error('Pack not set. Tap "Import SD15 pack (.zip)" or "Unzip SD15 pack from external" first.');
                const mod = (NativeModules as any)?.OrtNative;
                if (!mod?.inspectModelCpu) throw new Error('Native module OrtNative.inspectModelCpu is not available (rebuild required).');
                const info = await mod.inspectModelCpu(packFileUri('text_encoder.onnx'));
                setProvider(String(info?.provider ?? 'unknown'));
                setOutput(JSON.stringify({ file: 'text_encoder.onnx', inputs: info?.inputs, outputs: info?.outputs }));
                setStatus('OK');
              } catch (e: any) {
                setErr(e?.message || String(e));
                setStatus('Error');
              } finally {
                setBusy(false);
                completeGuideStep('inspectText');
              }
            }}
          >
            <Text style={styles.btnText}>Inspect pack text_encoder</Text>
          </Pressable>
        ) : null}

        {showGuideButton('inspectVae') ? (
          <Pressable
            style={[styles.btn, busy ? styles.btnDisabled : null]}
            disabled={busy}
            onPress={async () => {
              setBusy(true);
              setErr('');
              setProvider('-');
              setOutput('-');
              setStatus('Inspecting vae_decoder (CPU)...');
              try {
                if (!packDir) throw new Error('Pack not set. Tap "Import SD15 pack (.zip)" or "Unzip SD15 pack from external" first.');
                const mod = (NativeModules as any)?.OrtNative;
                if (!mod?.inspectModelCpu) throw new Error('Native module OrtNative.inspectModelCpu is not available (rebuild required).');
                const info = await mod.inspectModelCpu(packFileUri('vae_decoder.onnx'));
                setProvider(String(info?.provider ?? 'unknown'));
                setOutput(JSON.stringify({ file: 'vae_decoder.onnx', inputs: info?.inputs, outputs: info?.outputs }));
                setStatus('OK');
              } catch (e: any) {
                setErr(e?.message || String(e));
                setStatus('Error');
              } finally {
                setBusy(false);
                completeGuideStep('inspectVae');
              }
            }}
          >
            <Text style={styles.btnText}>Inspect pack vae_decoder</Text>
          </Pressable>
        ) : null}

        {showGuideButton('showExternalPath') ? (
          <Pressable
            style={[styles.btn, busy ? styles.btnDisabled : null]}
            disabled={busy}
            onPress={async () => {
              setBusy(true);
              setErr('');
              try {
                setStatus('Getting external path...');
                const mod = (NativeModules as any)?.OrtNative;
                if (!mod?.getExternalFilesDirPath) throw new Error('Native module OrtNative.getExternalFilesDirPath is not available (rebuild required).');
                const p = await mod.getExternalFilesDirPath();
                setExternalDir(String(p || ''));
                setStatus('OK');
              } catch (e: any) {
                setErr(e?.message || String(e));
                setStatus('Error');
              } finally {
                setBusy(false);
                completeGuideStep('showExternalPath');
              }
            }}
          >
            <Text style={styles.btnText}>{busy ? 'Working...' : 'Show app external pack path'}</Text>
          </Pressable>
        ) : null}

        {showGuideButton('unzipExternal') ? (
          <Pressable
            style={[styles.btn, busy ? styles.btnDisabled : null]}
            disabled={busy}
            onPress={async () => {
              setBusy(true);
              setErr('');
              try {
                if (!externalDir) throw new Error('External dir is not set. Tap "Show app external pack path" first.');
                const mod = (NativeModules as any)?.OrtNative;
                if (!mod?.unzip) throw new Error('Native module OrtNative.unzip is not available (rebuild required).');
                const extBase = externalDir.replace(/\/+$/, '');
                const zipCandidates = [
                  `file://${extBase}/sd15-onnx-pack.zip`,
                  'file:///storage/emulated/0/Android/data/com.anonymous.staticplaysd1mobile/files/sd15-onnx-pack.zip',
                  'file:///storage/emulated/0/Download/sd15-onnx-pack.zip',
                ];
                let zipUri = zipCandidates[0];
                if (mod?.statPath) {
                  let found = '';
                  for (const candidate of zipCandidates) {
                    const st = await mod.statPath(candidate);
                    if (st?.exists) {
                      found = candidate;
                      break;
                    }
                  }
                  if (!found) throw new Error(`Zip not found in any expected location: ${zipCandidates.join(' | ')}`);
                  zipUri = found;
                }
                const destDir = `file://${externalDir.replace(/\/+$/, '')}/sd15-onnx/`;
                setStatus('Unzipping... (this can take minutes)');
                const out = await mod.unzip(zipUri, destDir);
                {
                  const base = String(out?.destPath || destDir);
                  const normalized = base.startsWith('file://') ? base.slice('file://'.length) : base;
                  setPackDir(`${normalized.replace(/\/+$/, '')}/sd15-onnx`);
                }
                setPackFiles(Number(out?.files || 0));
                setStatus(`Unzipped ${Number(out?.files || 0)} file(s)`);
              } catch (e: any) {
                setErr(e?.message || String(e));
                setStatus('Error');
              } finally {
                setBusy(false);
                completeGuideStep('unzipExternal');
              }
            }}
          >
            <Text style={styles.btnText}>{busy ? 'Working...' : 'Unzip SD15 pack from external'}</Text>
          </Pressable>
        ) : null}

        {busy ? <ActivityIndicator style={{ marginTop: 6 }} /> : null}

        <View style={styles.card}>
          <Text style={styles.label}>Status</Text>
          <Text style={styles.value}>{status}</Text>
          <Text style={[styles.label, { marginTop: 10 }]}>Next</Text>
          <Text style={styles.value}>See docs/SD1_ON_DEVICE_PLAN.md</Text>
        </View>

        <View style={styles.card}>
          <Text style={styles.label}>Native provider</Text>
          <Text style={styles.value}>{provider}</Text>
          <Text style={[styles.label, { marginTop: 10 }]}>Native output</Text>
          <Text style={styles.value}>{output}</Text>
          {err ? (
            <>
              <Text style={[styles.label, { marginTop: 10, color: '#fbbf24' }]}>Error</Text>
              <Text selectable style={[styles.value, { color: '#fbbf24' }]}>
                {err}
              </Text>
            </>
          ) : null}
        </View>

        <View style={styles.card}>
          <Text style={styles.label}>Last export path</Text>
          <Text style={styles.value}>{lastExportPath || '-'}</Text>
        </View>

        <View style={styles.card}>
          <Text style={styles.label}>Imported files</Text>
          <Text style={styles.value}>{importedFiles.length ? importedFiles.map((p) => p.split('/').pop()).join(', ') : '-'}</Text>
        </View>

        <View style={styles.card}>
          <Text style={styles.label}>Pack dir</Text>
          <Text style={styles.value}>{packDir || '-'}</Text>
          <Text style={[styles.label, { marginTop: 10 }]}>Pack files</Text>
          <Text style={styles.value}>{packDir ? String(packFiles) : '-'}</Text>
          <Text style={[styles.label, { marginTop: 10 }]}>External dir</Text>
          <Text style={styles.value}>{externalDir || '-'}</Text>
        </View>
      </ScrollView>
      <StatusBar style="auto" />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#0b0f19' },
  container: { padding: 18, gap: 12 },
  title: { color: '#eaf0ff', fontSize: 22, fontWeight: '900' },
  sub: { color: '#8aa0c4', fontSize: 12 },
  btn: {
    backgroundColor: '#2563eb',
    paddingVertical: 12,
    paddingHorizontal: 14,
    borderRadius: 12,
    alignItems: 'center',
    marginTop: 6,
  },
  btnDisabled: { opacity: 0.5 },
  btnText: { color: 'white', fontWeight: '800' },
  card: {
    backgroundColor: 'rgba(0,0,0,0.25)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.09)',
    borderRadius: 12,
    padding: 14,
    marginTop: 8,
  },
  label: { color: '#9db2d6', fontSize: 12 },
  value: { color: '#eaf0ff', marginTop: 4 },
  input: {
    color: '#eaf0ff',
    marginTop: 8,
    padding: 10,
    borderRadius: 10,
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.10)',
    minHeight: 80,
    textAlignVertical: 'top',
  },
  row: {
    flexDirection: 'row',
    gap: 8,
    marginTop: 10,
  },
  smallBtn: {
    borderRadius: 10,
    paddingVertical: 8,
    paddingHorizontal: 10,
  },
  smallBtnPrimary: {
    backgroundColor: '#1d4ed8',
  },
  smallBtnGhost: {
    backgroundColor: '#1f2937',
    borderWidth: 1,
    borderColor: '#334155',
  },
  smallBtnText: {
    color: '#eaf0ff',
    fontWeight: '700',
    fontSize: 12,
  },
});
