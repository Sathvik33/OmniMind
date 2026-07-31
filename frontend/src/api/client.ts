const API_BASE = import.meta.env.VITE_BACKEND_URL ?? "http://127.0.0.1:8000";

export type JobStatus = {
  job_id: number;
  status: string;
  artifact_id?: number;
  failed_stage?: string | null;
  error_message?: string | null;
};

export type UploadResult = {
  job_id: number;
  artifact_id?: number;
  filename?: string;
  modality?: string;
};

export async function uploadFile(file: File): Promise<UploadResult> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || `Upload failed (${res.status})`);
  }
  return res.json();
}

export async function getJobStatus(jobId: number): Promise<JobStatus> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}`);
  if (!res.ok) throw new Error(`Job status failed (${res.status})`);
  return res.json();
}

export async function streamQuery(
  query: string,
  onToken: (chunk: string) => void,
  signal?: AbortSignal,
  artifactIds?: number[],
): Promise<string> {
  const res = await fetch(`${API_BASE}/query-stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      artifact_ids: artifactIds?.length ? artifactIds : undefined,
    }),
    signal,
  });
  if (!res.ok || !res.body) {
    throw new Error(`Query failed (${res.status})`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let full = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value, { stream: true });
    full += chunk;
    onToken(chunk);
  }

  return full;
}

export function cleanStreamText(text: string, opts?: { final?: boolean }): string {
  const lines = text.split("\n");
  const kept: string[] = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const s = line.trim();
    const isLast = i === lines.length - 1;
    // Keep incomplete trailing metadata lines out of the live view
    if (!opts?.final && isLast && (/^\[Retrieved:/.test(s) || /^\[Response confidence:/.test(s))) {
      continue;
    }
    if (/^\[Retrieved:.*\]$/.test(s)) continue;
    if (/^\[Response confidence:.*\]$/.test(s)) continue;
    if (s.startsWith("⚠️") || s.startsWith("ℹ️") || s.startsWith("❌")) continue;
    kept.push(line);
  }

  let stripped = kept.join("\n");

  // Light markdown → plain text for any residual LLM markup
  stripped = stripped
    .replace(/```[\w+-]*\n?([\s\S]*?)```/g, "$1")
    .replace(/!\[([^\]]*)\]\([^)]+\)/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/(\*\*|__)(.*?)\1/g, "$2")
    .replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/^\s*[-*+]\s+/gm, "• ")
    .replace(/\n{3,}/g, "\n\n");

  return opts?.final ? stripped.trim() : stripped.replace(/^\n+/, "");
}
