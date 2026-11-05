// Thin fetch wrapper around the JSON API.
async function req(method, url, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(url, opts);
  if (res.status === 204) return null;
  let data = null;
  try { data = await res.json(); } catch (_) { /* no body */ }
  if (!res.ok) {
    let msg = (data && data.detail) || res.statusText;
    if (Array.isArray(msg)) msg = msg.map((e) => e.msg).join("; ");
    const err = new Error(msg);
    err.status = res.status;
    throw err;
  }
  return data;
}

const qs = (o) =>
  Object.entries(o).filter(([, v]) => v !== undefined && v !== null && v !== "")
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`).join("&");

export const api = {
  bootstrap: () => req("GET", "/api/bootstrap"),
  search: (q) => req("GET", `/api/search?${qs({ q })}`),
  nearestGrid: (lat, lon) => req("GET", `/api/nearest-grid?${qs({ lat, lon })}`),
  nearestStation: (lat, lon) => req("GET", `/api/nearest-station?${qs({ lat, lon })}`),
  gridPoints: (b) => req("GET", `/api/grid-points?${qs(b)}`),
  turbineSpecs: (name) => req("GET", `/api/turbine-specs?${qs({ name })}`),
  layout: (body) => req("POST", "/api/layout", body),

  scenarios: () => req("GET", "/api/scenarios"),
  createScenario: (name, kind) => req("POST", "/api/scenarios", { name, kind }),
  getScenario: (id) => req("GET", `/api/scenarios/${id}`),
  updateScenario: (id, patch) => req("PUT", `/api/scenarios/${id}`, patch),
  deleteScenario: (id) => req("DELETE", `/api/scenarios/${id}`),
  duplicateScenario: (id) => req("POST", `/api/scenarios/${id}/duplicate`),
  run: (id) => req("POST", `/api/scenarios/${id}/run`),
  job: (id) => req("GET", `/api/jobs/${id}`),

  results: () => req("GET", "/api/results"),
  result: (id) => req("GET", `/api/results/${id}`),
  figure: (id, name, params = {}) => req("GET", `/api/results/${id}/figure/${name}?${qs(params)}`),
  table: (id, page, pageSize = 50) => req("GET", `/api/results/${id}/table?${qs({ page, page_size: pageSize })}`),
  deleteResult: (id) => req("DELETE", `/api/results/${id}`),
  mlArchitectures: () => req("GET", "/api/ml/architectures"),
  packageImport: () => req("POST", "/api/package/import"),
  mlModels: () => req("GET", "/api/ml/models"),
  mlModel: (id) => req("GET", `/api/ml/models/${id}`),
  mlTrain: (body) => req("POST", "/api/ml/train", body),
  mlFinetune: (id, body) => req("POST", `/api/ml/models/${id}/finetune`, body),
  mlDelete: (id) => req("DELETE", `/api/ml/models/${id}`),
  mlFigure: (id, name) => req("GET", `/api/ml/models/${id}/figure/${name}`),
  compareReal: (body) => req("POST", "/api/compare/real", body),
  compare: (ids) => req("POST", "/api/compare", { result_ids: ids }),
};

// Poll a background job until it finishes; `onUpdate(job)` is called on every poll.
export async function waitForJob(job, onUpdate) {
  const t0 = Date.now();
  while (job.status === "queued" || job.status === "running") {
    onUpdate && onUpdate(job, Math.round((Date.now() - t0) / 1000));
    await new Promise((r) => setTimeout(r, 700));
    job = await api.job(job.id);
  }
  if (job.status === "error") throw new Error(job.error);
  return job;
}
