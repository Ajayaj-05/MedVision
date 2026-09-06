import { useEffect, useRef, useState } from 'react'
import './App.css'

// Use the Vite dev proxy so the browser stays same-origin while the backend
// remains available at http://127.0.0.1:8000/predict.
const API_URL = '/api/predict'
const MAX_FILE_SIZE = 10 * 1024 * 1024
const ACCEPTED_TYPES = ['image/jpeg', 'image/png']

function Icon({ name, size = 20 }) {
  const paths = {
    shield: <><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z" /><path d="m9 12 2 2 4-4" /></>,
    upload: <><path d="M12 16V4" /><path d="m7 9 5-5 5 5" /><path d="M5 20h14" /></>,
    image: <><rect width="18" height="18" x="3" y="3" rx="2" /><circle cx="8.5" cy="8.5" r="1.5" /><path d="m21 15-5-5L5 21" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    download: <><path d="M12 3v12" /><path d="m7 10 5 5 5-5" /><path d="M5 21h14" /></>,
    refresh: <><path d="M20 11a8.1 8.1 0 0 0-15.5-2M4 5v4h4" /><path d="M4 13a8.1 8.1 0 0 0 15.5 2M20 19v-4h-4" /></>,
    close: <><path d="m6 6 12 12M18 6 6 18" /></>,
  }
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>
}

function formatBytes(bytes) { return `${(bytes / 1024 / 1024).toFixed(1)} MB` }

function imageSource(value) {
  return value?.startsWith('data:') ? value : `data:image/png;base64,${value}`
}

function loadImage(source) {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = reject
    image.src = source
  })
}

async function createGradCamVisualization(originalSource, heatmapValue) {
  const [original, heatmap] = await Promise.all([
    loadImage(originalSource),
    loadImage(imageSource(heatmapValue)),
  ])
  const width = original.naturalWidth || original.width
  const height = original.naturalHeight || original.height
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const context = canvas.getContext('2d')
  context.drawImage(original, 0, 0, width, height)

  const heatmapCanvas = document.createElement('canvas')
  heatmapCanvas.width = width
  heatmapCanvas.height = height
  const heatmapContext = heatmapCanvas.getContext('2d', { willReadFrequently: true })
  // The backend returns a small grayscale activation map (typically 4x4).
  // Interpolate it while scaling so the activation regions do not become
  // hard-edged blocks in the browser.
  heatmapContext.imageSmoothingEnabled = true
  heatmapContext.imageSmoothingQuality = 'high'
  // A blur proportional to the image size softens the remaining coarse
  // activation boundaries before they are converted into colors.
  const blurRadius = Math.max(12, Math.round(Math.min(width, height) * 0.04))
  heatmapContext.filter = `blur(${blurRadius}px)`
  heatmapContext.drawImage(heatmap, 0, 0, width, height)
  heatmapContext.filter = 'none'
  const pixels = heatmapContext.getImageData(0, 0, width, height)
  const overlayCanvas = document.createElement('canvas')
  overlayCanvas.width = width
  overlayCanvas.height = height
  const overlayContext = overlayCanvas.getContext('2d')
  const overlay = overlayContext.createImageData(width, height)

  for (let index = 0; index < pixels.data.length; index += 4) {
    const intensity = (pixels.data[index] + pixels.data[index + 1] + pixels.data[index + 2]) / (255 * 3)
    const value = Math.min(1, Math.max(0, intensity))
    // Keep weak activations nearly transparent while preserving strong ones.
    const activation = Math.min(1, Math.max(0, (value - 0.04) / 0.96))
    const alpha = Math.round(110 * Math.pow(activation, 1.15))

    // Cyan → green → yellow → orange → red is a familiar Grad-CAM scale.
    let red
    let green
    let blue
    if (value < 0.25) {
      const mix = value / 0.25
      red = 0
      green = Math.round(190 + 65 * mix)
      blue = Math.round(220 * (1 - mix))
    } else if (value < 0.5) {
      const mix = (value - 0.25) / 0.25
      red = Math.round(255 * mix)
      green = 255
      blue = 0
    } else if (value < 0.75) {
      const mix = (value - 0.5) / 0.25
      red = 240
      green = Math.round(255 * (1 - mix))
      blue = 0
    } else {
      const mix = (value - 0.75) / 0.25
      red = 240
      green = Math.round(105 * (1 - mix))
      blue = 0
    }
    overlay.data[index] = red
    overlay.data[index + 1] = green
    overlay.data[index + 2] = blue
    overlay.data[index + 3] = alpha
  }
  overlayContext.putImageData(overlay, 0, 0)
  context.globalCompositeOperation = 'source-over'
  context.drawImage(overlayCanvas, 0, 0, width, height)
  return canvas.toDataURL('image/png')
}

function App() {
  const inputRef = useRef(null)
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [dimensions, setDimensions] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [status, setStatus] = useState('upload')
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const [gradcamVisualization, setGradcamVisualization] = useState('')
  const [view, setView] = useState('original')
  const [analyzedAt, setAnalyzedAt] = useState(null)

  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview])

  useEffect(() => {
    let cancelled = false
    if (!result || !preview) return undefined
    createGradCamVisualization(preview, result.gradcam_image)
      .then((visualization) => { if (!cancelled) setGradcamVisualization(visualization) })
      .catch(() => { if (!cancelled) setGradcamVisualization('') })
    return () => { cancelled = true }
  }, [preview, result])

  const selectFile = (selectedFile) => {
    setError('')
    if (!selectedFile) return
    if (!ACCEPTED_TYPES.includes(selectedFile.type)) { setError('Please choose a JPG, JPEG, or PNG image.'); return }
    if (selectedFile.size > MAX_FILE_SIZE) { setError('That image is larger than 10 MB. Please choose a smaller file.'); return }
    const objectUrl = URL.createObjectURL(selectedFile)
    const image = new Image()
    image.onload = () => setDimensions({ width: image.naturalWidth, height: image.naturalHeight })
    image.src = objectUrl
    setFile(selectedFile); setPreview(objectUrl); setResult(null); setGradcamVisualization(''); setStatus('upload')
  }

  const clearFile = () => {
    if (preview) URL.revokeObjectURL(preview)
    setFile(null); setPreview(''); setDimensions(null); setGradcamVisualization(''); setError('')
    if (inputRef.current) inputRef.current.value = ''
  }

  const analyze = async () => {
    if (!file || status === 'processing') return
    setError(''); setStatus('processing')
    const formData = new FormData(); formData.append('file', file)
    try {
      const response = await fetch(API_URL, { method: 'POST', body: formData })
      if (!response.ok) throw new Error('service')
      const data = await response.json()
      if (!data.predicted_class || typeof data.confidence !== 'number' || typeof data.pneumonia_probability !== 'number' || !data.gradcam_image) throw new Error('response')
      setResult(data); setAnalyzedAt(new Date()); setView('original'); setStatus('results')
    } catch (requestError) {
      setStatus('upload')
      setError(requestError.message === 'Failed to fetch' ? 'Unable to reach MedVision. Please make sure the analysis service is running and try again.' : 'We could not complete the analysis. Please check the image and try again.')
    }
  }

  const downloadReport = () => {
    if (!result) return
    const gradcam = gradcamVisualization || preview
    const time = analyzedAt?.toLocaleString() || new Date().toLocaleString()
    const report = `<!doctype html><html><head><title>MedVision Analysis Report</title><style>body{font:15px Arial;color:#172b4d;max-width:760px;margin:40px auto}h1{font-size:30px;color:#123b65}h2{margin-top:28px;border-bottom:1px solid #d9e3ed;padding-bottom:8px}.meta{color:#63758a}.stat{display:inline-block;margin:10px 28px 10px 0;font-size:20px;font-weight:bold}img{max-width:100%;max-height:420px;object-fit:contain;background:#eef4f8;padding:8px}.disclaimer{background:#f3f7fa;padding:16px;margin-top:30px;color:#4d6175}</style></head><body><h1>MedVision</h1><p class="meta">Chest X-ray Analysis Assistant · ${time}</p><h2>Analysis summary</h2><div class="stat">Prediction: ${result.predicted_class}</div><div class="stat">Model confidence ${result.confidence.toFixed(1)}%</div><div class="stat">Pneumonia probability ${result.pneumonia_probability.toFixed(1)}%</div><h2>Original X-ray</h2><img src="${preview}" alt="Original uploaded chest X-ray"><h2>Grad-CAM explanation</h2><img src="${gradcam}" alt="Grad-CAM explanation"><div class="disclaimer"><b>Assistive tool only — not a medical diagnosis.</b></div></body></html>`
    // Keep the returned window reference so the report document can be filled.
    // noopener prevents that access in some browsers, leaving a blank tab.
    const reportWindow = window.open('', '_blank')
    if (!reportWindow) { setError('Please allow pop-ups to print your report.'); return }
    const reportDocument = reportWindow.document
    reportDocument.open()
    reportDocument.write(report)
    reportDocument.close()

    const images = Array.from(reportDocument.images)
    const imagesReady = Promise.all(images.map((image) => {
      if (image.complete) return Promise.resolve()
      return new Promise((resolve) => {
        image.onload = resolve
        image.onerror = resolve
      })
    }))
    imagesReady.then(() => {
      reportWindow.focus()
      setTimeout(() => {
        reportWindow.onafterprint = () => reportWindow.close()
        reportWindow.print()
      }, 100)
    })
  }

  const startOver = () => { clearFile(); setResult(null); setStatus('upload'); setAnalyzedAt(null) }
  const isPneumonia = result?.predicted_class === 'PNEUMONIA'

  return <div className="app-shell">
    <header className="topbar"><div className="brand"><div className="brand-mark"><Icon name="shield" size={23} /></div><div><strong>Med<span>Vision</span></strong><small>Chest X-ray Analysis Assistant</small></div></div><div className="trust-badge"><span className="status-dot" /> Assistive AI tool</div></header>
    <main>
      <section className="hero"><div className="eyebrow"><span /> CLINICAL IMAGING SUPPORT</div><h1>Understand the image.<br /><em>Support the next decision.</em></h1><p>MedVision offers an AI-assisted look at chest X-rays, with transparent visual explanations to support learning and research.</p></section>
      {status !== 'results' && <section className="workspace upload-workspace"><div className="section-heading"><div><span className="step-label">01 / UPLOAD</span><h2>Upload a chest X-ray</h2><p>Use a clear frontal chest X-ray in JPG, JPEG, or PNG format.</p></div><div className="file-limit">Max file size 10 MB</div></div><div className={`dropzone ${dragging ? 'is-dragging' : ''} ${file ? 'has-file' : ''}`} onDragOver={(e) => { e.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={(e) => { e.preventDefault(); setDragging(false); selectFile(e.dataTransfer.files[0]) }}>{!file ? <><div className="upload-icon"><Icon name="upload" size={25} /></div><h3>Drop your X-ray here</h3><p>or <button className="text-button" onClick={() => inputRef.current?.click()}>browse from your device</button></p><span className="format-note">JPG · JPEG · PNG</span></> : <div className="selected-file"><img src={preview} alt="Preview of selected chest X-ray" /><div className="file-details"><div className="file-type"><Icon name="image" size={16} /> Image ready</div><h3>{file.name}</h3><p>{dimensions ? `${dimensions.width} × ${dimensions.height} px` : 'Reading dimensions...'} · {formatBytes(file.size)}</p></div><button className="icon-button" onClick={clearFile} aria-label="Remove selected image"><Icon name="close" size={19} /></button></div>}<input ref={inputRef} type="file" accept=".jpg,.jpeg,.png,image/jpeg,image/png" onChange={(e) => selectFile(e.target.files[0])} hidden /></div>{error && <div className="error-message" role="alert">{error}</div>}{status === 'processing' && <div className="processing"><span className="spinner" /><div><strong>Analyzing X-ray, please wait...</strong><small>Securely processing the image and generating an explanation.</small></div></div>}<div className="action-row"><button className="primary-button" disabled={!file || status === 'processing'} onClick={analyze}>{status === 'processing' ? 'Analyzing...' : 'Analyze X-ray'} <span>→</span></button>{file && <button className="secondary-button" onClick={clearFile}>Clear selection</button>}</div></section>}
      {status === 'results' && result && <section className="workspace results-workspace"><div className="results-header"><div><span className="step-label">02 / RESULTS</span><h2>Analysis overview</h2><p>AI-assisted findings based on the uploaded image.</p></div><button className="secondary-button report-button" onClick={downloadReport}><Icon name="download" size={17} /> Download Report</button></div><div className="results-grid"><div className="viewer-card"><div className="viewer-toolbar"><span>Image viewer</span><div className="toggle" role="tablist"><button className={view === 'original' ? 'active' : ''} onClick={() => setView('original')}>Original X-ray</button><button className={view === 'gradcam' ? 'active' : ''} onClick={() => setView('gradcam')}>Grad-CAM</button></div></div><div className="image-stage"><img src={view === 'original' ? preview : (gradcamVisualization || preview)} alt={view === 'original' ? 'Original uploaded chest X-ray' : 'Grad-CAM visualization highlighting image regions that influenced the model prediction'} /></div><p className="viewer-note">{view === 'gradcam' ? 'Grad-CAM: regions that influenced the model prediction. It is an AI explanation, not a medical finding.' : `${file?.name || 'Uploaded X-ray'} · ${dimensions ? `${dimensions.width} × ${dimensions.height} px` : ''}`}</p></div><div className="insight-card"><div className={`result-banner ${isPneumonia ? 'positive' : 'normal'}`}><div className="result-icon"><Icon name={isPneumonia ? 'image' : 'check'} size={20} /></div><div><span>Prediction</span><h3>{isPneumonia ? 'Pneumonia Detected' : 'Normal'}</h3></div></div><div className="metric"><div><span>Model confidence</span><strong>{result.confidence.toFixed(1)}%</strong></div><div className="bar"><i style={{ width: `${Math.min(100, Math.max(0, result.confidence))}%` }} /></div></div><div className="metric secondary-metric"><div><span>Pneumonia probability</span><strong>{result.pneumonia_probability.toFixed(1)}%</strong></div><div className="bar muted"><i style={{ width: `${Math.min(100, Math.max(0, result.pneumonia_probability))}%` }} /></div></div><div className="result-meta"><span>Analyzed</span><strong>{analyzedAt?.toLocaleString()}</strong></div></div></div><button className="secondary-button another-button" onClick={startOver}><Icon name="refresh" size={16} /> Analyze Another X-ray</button></section>}
    </main><footer><span>MEDVISION <i>•</i> CHEST X-RAY ANALYSIS ASSISTANT</span><span className="footer-disclaimer">Assistive tool only — not a medical diagnosis.</span></footer>
  </div>
}

export default App
