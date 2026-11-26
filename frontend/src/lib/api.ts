/**
 * API client for GWLab FastAPI backend service.
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface SystemStatus {
  status: string;
  version: string;
  mode: string;
  detector: string;
  duration_s: number;
  sampling_rate_hz: number;
  f_lower_hz: number;
  f_upper_hz: number;
  waveform_approximant: string;
  governance: string;
}

export interface ParameterSummaryData {
  median: number;
  p5: number;
  p95: number;
  ci90_width: number;
}

export interface Stage6CData {
  run_id: string;
  status: string;
  log_evidence: number;
  log_evidence_err: number;
  n_evaluations: number;
  guard_evaluations_completed?: number;
  wall_time_s: number;
  cpu_time_s: number;
  parameters: Record<string, ParameterSummaryData>;
  boundary_concentration_pct: number;
  artifact_path: string;
}

export interface LedgerData {
  cumulative_calls: number;
  cumulative_cpu_s: number;
  cumulative_wall_s: number;
  remaining_calls: number;
  remaining_cpu_s: number;
  remaining_wall_s: number;
  total_budget_calls: number;
  total_budget_cpu_s: number;
  total_budget_wall_s: number;
  consumption_pct: number;
  artifact_path: string;
  notice: string;
}

export interface SeedComparisonData {
  seed_1: number;
  seed_2: number;
  run_1_id: string;
  run_2_id: string;
  logz_1: number;
  logzerr_1: number;
  logz_2: number;
  logzerr_2: number;
  n_eval_1: number;
  n_eval_2: number;
  delta_logz: number;
  sigma_delta: number;
  z_ratio: number;
  z_ratio_formatted: string;
  qualification: string;
  artifact_path: string;
}

export interface ProvenanceItem {
  artifact: string;
  category: string;
  path: string;
  scope: string;
  status: string;
}

export interface OverviewBenchmarks {
  stage6c: Stage6CData;
  ledger: LedgerData;
  seed_comparison: SeedComparisonData;
  provenance_matrix: ProvenanceItem[];
}

export interface WaveformRequestParams {
  mc_msun: number;
  eta: number;
  distance_mpc: number;
  tc_s?: number;
  phic_rad?: number;
  duration_s?: number;
  sampling_rate_hz?: number;
  f_end_hz?: number;
}

export interface WaveformResponseData {
  frequencies: number[];
  amplitude: number[];
  unwrapped_phase: number[];
  wrapped_phase: number[];
  real: number[];
  imag: number[];
  m1_msun: number;
  m2_msun: number;
  m_total_msun: number;
  q: number;
  f_isco: number;
  f_term: number;
  amp_prefactor: number;
  n_active_bins: number;
}

export interface ChirpTimeResponseData {
  frequencies: number[];
  tau_newtonian: number[];
  tau_35pn: number[];
}

export interface NoiseAnalysisRequestParams {
  psd_model?: "aligo_representative" | "flat";
  f_min_hz?: number;
  f_max_hz?: number;
  sampling_rate_hz?: number;
  duration_obs_s?: number;
  segment_duration_s?: number;
  overlap_fraction?: number;
  tukey_alpha?: number;
  seed?: number;
}

export interface NoiseAnalysisMetadata {
  psd_model: string;
  psd_model_label: string;
  psd_provenance: string;
  f_min_hz: number;
  f_max_hz: number;
  sampling_rate_hz: number;
  duration_obs_s: number;
  segment_duration_s: number;
  overlap_fraction: number;
  tukey_alpha: number;
  seed: number;
  n_segments: number;
  kappa_overlap: number;
  nu_eff: number;
  nu_eff_qualification: string;
  delta_f_hz: number;
  n_active_bins: number;
  noise_mean: number;
  noise_std: number;
  units: {
    frequency: string;
    psd: string;
    asd: string;
    strain: string;
    time: string;
  };
}

export interface NoiseAnalysisResponseData {
  frequencies: number[];
  psd_analytic: number[];
  psd_welch: number[];
  asd_analytic: number[];
  asd_welch: number[];
  time_preview_s: number[];
  strain_preview: number[];
  metadata: NoiseAnalysisMetadata;
}

export interface MatchedFilterRequestParams {
  mc_msun?: number;
  eta?: number;
  distance_mpc?: number;
  tc_s?: number;
  phic_rad?: number;
  delta_mc_msun?: number;
  psd_model?: "aligo_representative" | "flat";
  f_min_hz?: number;
  f_max_hz?: number;
  seed?: number;
}

export interface MatchedFilterMetadata {
  optimal_snr: number;
  sigma_h: number;
  recovered_peak_snr: number;
  recovered_peak_time_s: number;
  recovered_peak_phase_rad: number;
  time_offset_s: number;
  injected_tc_s: number;
  injected_mc_msun: number;
  template_mc_msun: number;
  delta_mc_msun: number;
  eta: number;
  distance_mpc: number;
  f_isco_inj_hz: number;
  f_isco_tmpl_hz: number;
  f_min_hz: number;
  f_max_hz: number;
  n_active_bins: number;
  delta_f_hz: number;
  duration_s: number;
  sampling_rate_hz: number;
  psd_model: string;
  psd_model_label: string;
  seed: number;
  units: {
    snr: string;
    time: string;
    frequency: string;
    fourier_strain: string;
  };
}

export interface MatchedFilterResponseData {
  snr_time_s: number[];
  snr_abs: number[];
  snr_real: number[];
  zoom_time_s: number[];
  zoom_abs: number[];
  zoom_real: number[];
  frequencies: number[];
  data_amp: number[];
  injection_amp: number[];
  template_amp: number[];
  metadata: MatchedFilterMetadata;
}

export type PosteriorParamKey = "mc" | "eta" | "tc" | "phic";

export interface PosteriorMarginal1D {
  key: PosteriorParamKey;
  label: string;
  symbol: string;
  unit: string;
  prior_bounds: [number, number];
  injected: number;
  support_range: [number, number];
  bin_width: number;
  bin_edges: number[];
  bin_centers: number[];
  density: number[];
  summary: {
    median: number;
    p5: number;
    p95: number;
    ci90_width: number;
    weighted_mean: number;
    weighted_std: number;
  };
}

export interface PosteriorJoint2D {
  param_x: PosteriorParamKey;
  param_y: PosteriorParamKey;
  x_label: string;
  y_label: string;
  x_unit: string;
  y_unit: string;
  x_centers: number[];
  y_centers: number[];
  x_edges: number[];
  y_edges: number[];
  grid_normalized: number[][];
  max_bin_mass: number;
  injected_x: number;
  injected_y: number;
}

export interface PosteriorMetadata {
  run_id: string;
  status: string;
  realization: string;
  likelihood_model: string;
  prior_model: string;
  sampler: string;
  seed: number;
  n_samples: number;
  n_dims: number;
  n_evaluations: number;
  guard_evaluations_completed: number;
  log_evidence: number;
  log_evidence_err: number;
  kish_n_eff: number;
  eta_top1_pct: number;
  weight_sum: number;
  between_seed_delta_logz: number;
  artifact_npz: string;
  artifact_json: string;
  binning_note: string;
  qualification: string;
}

export interface PosteriorExplorerResponseData {
  parameter_order: PosteriorParamKey[];
  marginals_1d: Record<PosteriorParamKey, PosteriorMarginal1D>;
  joint_2d: PosteriorJoint2D;
  metadata: PosteriorMetadata;
}

export interface ExperimentRunSummary {
  run_id: string;
  status: string;
  model: string;
  realization: string;
  sampler_seed: number;
  nlive: number | null;
  log_evidence: number;
  log_evidence_err: number;
  n_evaluations: number | null;
  n_total_samples: number | null;
  n_eff: number | null;
  wall_time_s: number | null;
  cpu_time_s: number | null;
  mc_median: number | null;
  eta_median: number | null;
  tc_median: number | null;
  phic_r_bar: number | null;
  artifact_path: string;
}

export type CampaignStageStatus =
  | "Completed / Historical"
  | "Unexecuted"
  | "Proposed"
  | "Frozen / Specified";

export interface CampaignStageItem {
  stage_id: string;
  title: string;
  status: CampaignStageStatus;
  model: string;
  realization: string;
  summary: string;
  artifact_path: string;
}

export interface ExperimentsExplorerResponseData {
  multi_seed_comparison: {
    runs: ExperimentRunSummary[];
    delta_logz: number;
    sigma_delta: number;
    discrepancy_ratio: number | null;
    qualification: string;
  };
  stage6c_anchor: {
    run_id: string;
    status: string;
    log_evidence: number;
    log_evidence_err: number;
    n_calls: number | null;
    n_samples: number | null;
    artifact_path: string;
  };
  ledger: {
    cumulative_calls: number | null;
    remaining_calls: number | null;
    run_b_authorized: boolean;
    artifact_path: string;
  };
  campaign_stages: CampaignStageItem[];
  limitations: string[];
}

export async function fetchSystemStatus(): Promise<SystemStatus> {
  const res = await fetch(`${API_BASE_URL}/api/status`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Failed to fetch system status: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchOverviewBenchmarks(): Promise<OverviewBenchmarks> {
  const res = await fetch(`${API_BASE_URL}/api/benchmarks/overview`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Failed to fetch overview benchmarks: ${res.statusText}`);
  }
  return res.json();
}

export async function generateWaveform(params: WaveformRequestParams): Promise<WaveformResponseData> {
  const res = await fetch(`${API_BASE_URL}/api/waveforms/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    throw new Error(`Waveform generation failed: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchChirpTime(params: {
  mc_msun: number;
  eta: number;
  f_isco?: number;
}): Promise<ChirpTimeResponseData> {
  const res = await fetch(`${API_BASE_URL}/api/waveforms/chirp-time`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    throw new Error(`Chirp time calculation failed: ${res.statusText}`);
  }
  return res.json();
}

export async function analyzeNoiseAndPsd(
  params: NoiseAnalysisRequestParams
): Promise<NoiseAnalysisResponseData> {
  const res = await fetch(`${API_BASE_URL}/api/noise/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const errBody = await res.json();
      if (errBody?.detail) detail = errBody.detail;
    } catch {
      // ignore json parse failure
    }
    throw new Error(`Noise & PSD analysis failed: ${detail}`);
  }
  return res.json();
}

export async function evaluateMatchedFilter(
  params: MatchedFilterRequestParams
): Promise<MatchedFilterResponseData> {
  const res = await fetch(`${API_BASE_URL}/api/matched-filter/evaluate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const errBody = await res.json();
      if (errBody?.detail) detail = errBody.detail;
    } catch {
      // ignore json parse failure
    }
    throw new Error(`Matched filter evaluation failed: ${detail}`);
  }
  return res.json();
}

export async function fetchInferencePosterior(params: {
  param_x?: PosteriorParamKey;
  param_y?: PosteriorParamKey;
  bins_1d?: number;
  bins_2d?: number;
} = {}): Promise<PosteriorExplorerResponseData> {
  const qs = new URLSearchParams();
  if (params.param_x) qs.set("param_x", params.param_x);
  if (params.param_y) qs.set("param_y", params.param_y);
  if (params.bins_1d) qs.set("bins_1d", String(params.bins_1d));
  if (params.bins_2d) qs.set("bins_2d", String(params.bins_2d));
  const res = await fetch(`${API_BASE_URL}/api/inference/posterior?${qs.toString()}`, {
    cache: "no-store",
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const errBody = await res.json();
      if (errBody?.detail) detail = errBody.detail;
    } catch {
      // ignore json parse failure
    }
    throw new Error(`Failed to load posterior archive: ${detail}`);
  }
  return res.json();
}

export async function fetchExperimentsExplorer(): Promise<ExperimentsExplorerResponseData> {
  const res = await fetch(`${API_BASE_URL}/api/experiments/explorer`, {
    cache: "no-store",
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const errBody = await res.json();
      if (errBody?.detail) detail = errBody.detail;
    } catch {
      // ignore json parse failure
    }
    throw new Error(`Failed to load experiments archive: ${detail}`);
  }
  return res.json();
}
