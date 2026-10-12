

export default function Transparency() {
  return (
    <div className="w-full max-w-[1200px] mx-auto flex flex-col p-8 gap-8">
      <div className="text-center">
        <h1 className="font-headline-lg text-headline-lg text-on-surface tracking-tight mb-2">Algorithm Transparency</h1>
        <p className="font-body-lg text-body-lg text-slate-body max-w-2xl mx-auto">
          Deep dives into the models, weights, and architectural decisions powering CheckReality AI.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-surface-crisp p-6 rounded-2xl shadow-sm border border-outline/10">
          <div className="flex items-center gap-3 mb-4">
            <span className="material-symbols-outlined text-primary text-2xl">image</span>
            <h2 className="font-headline-sm text-headline-sm text-on-surface font-bold">Image Saliency Engine</h2>
          </div>
          <p className="text-slate-body mb-4 font-body-md">
            Built on top of <strong>EfficientNet-B0 and EfficientNet-B3</strong> architectures. We fine-tuned these convolutional networks to detect spatial anomalies, such as unnatural blending boundaries and specular reflection mismatches introduced by GANs and diffusion models.
          </p>
          <ul className="text-sm font-label-code text-secondary flex flex-col gap-2">
            <li>• Architecture: EfficientNet (B0 / B3)</li>
            <li>• Explainability: Grad-CAM Heatmaps</li>
            <li>• Base Weights: ImageNet Pretrained</li>
          </ul>
        </div>

        <div className="bg-surface-crisp p-6 rounded-2xl shadow-sm border border-outline/10">
          <div className="flex items-center gap-3 mb-4">
            <span className="material-symbols-outlined text-secondary text-2xl">graphic_eq</span>
            <h2 className="font-headline-sm text-headline-sm text-on-surface font-bold">Audio Biometric Engine</h2>
          </div>
          <p className="text-slate-body mb-4 font-body-md">
            Utilizes a <strong>ResNet-18</strong> architecture applied to Mel-Spectrograms (visual representations of audio frequencies). It isolates synthetic vocoder artifacts, unnatural phase discontinuities, and high-frequency noise typical of voice clones.
          </p>
          <ul className="text-sm font-label-code text-secondary flex flex-col gap-2">
            <li>• Architecture: ResNet-18 (2D CNN)</li>
            <li>• Pre-processing: 128-bin Mel-Spectrograms (16kHz)</li>
            <li>• Target: Synthetic Voice & Vocoder Artifacts</li>
          </ul>
        </div>

        <div className="bg-surface-crisp p-6 rounded-2xl shadow-sm border border-outline/10">
          <div className="flex items-center gap-3 mb-4">
            <span className="material-symbols-outlined text-cyan-energy text-2xl">videocam</span>
            <h2 className="font-headline-sm text-headline-sm text-on-surface font-bold">Video Temporal Engine</h2>
          </div>
          <p className="text-slate-body mb-4 font-body-md">
            Combines an <strong>EfficientNet</strong> backbone with a <strong>Long Short-Term Memory (LSTM)</strong> network. It analyzes frame-by-frame spatial features and temporal consistency across 16-frame sequences to detect unnatural motion, blink rates, and lip-sync issues.
          </p>
          <ul className="text-sm font-label-code text-secondary flex flex-col gap-2">
            <li>• Architecture: EfficientNet + LSTM (Hidden Dim: 256)</li>
            <li>• Sampling: 16 equidistant frames per sequence</li>
            <li>• Target: Temporal Inconsistency & Deepfakes</li>
          </ul>
        </div>

        <div className="bg-surface-crisp p-6 rounded-2xl shadow-sm border border-outline/10">
          <div className="flex items-center gap-3 mb-4">
            <span className="material-symbols-outlined text-primary text-2xl">document_scanner</span>
            <h2 className="font-headline-sm text-headline-sm text-on-surface font-bold">Text NLP Engine</h2>
          </div>
          <p className="text-slate-body mb-4 font-body-md">
            Powered by <strong>RoBERTa-base</strong>, a robust transformer model. We run two parallel pipelines: one trained to detect AI-generated phrasing (perplexity and burstiness), and another tuned on factual verification datasets to flag misinformation and coordinated narrative manipulation.
          </p>
          <ul className="text-sm font-label-code text-secondary flex flex-col gap-2">
            <li>• Architecture: RoBERTa-base (Transformers)</li>
            <li>• Explainability: SHAP Token Contribution Scores</li>
            <li>• Task: Sequence Classification (Misinformation / AI Generated)</li>
          </ul>
        </div>
      </div>
      
      <div className="bg-primary-container p-6 rounded-2xl mt-4">
        <h2 className="font-headline-sm text-headline-sm text-on-primary-container font-bold mb-2">Multimodal Fusion</h2>
        <p className="text-on-primary-container/80">
          When inputs contain multiple modalities (e.g. Video with Audio), our fusion engine combines probabilities using a confidence-weighted heuristic (Video: 40%, Audio: 30%, Image: 20%, Text: 10%) calibrated via Platt Scaling to minimize Expected Calibration Error.
        </p>
      </div>
    </div>
  );
}
