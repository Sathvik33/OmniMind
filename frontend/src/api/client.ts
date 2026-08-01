const API_BASE = import.meta.env.VITE_BACKEND_URL ?? "http://127.0.0.1:8000";
const TOKEN_KEY = "aegis_token";

export type User = {
  id: number;
  email: string;
  username: string;
};

export type AuthResponse = {
  access_token: string;
  token_type: string;
  user: User;
};

export type ChatSummary = {
  id: number;
  title: string | null;
  created_at?: string | null;
  updated_at?: string | null;
};

export type ChatDocument = {
  name: string;
  ext?: string;
  status: string;
  statusLabel?: string;
  artifact_id?: number;
  sizeLabel?: string;
};

export type ChatMessageApi = {
  id: number;
  role: string;
  content: string;
  created_at?: string | null;
  document?: ChatDocument | null;
};

export type ChatArtifact = {
  id: number;
  filename: string;
  modality: string;
  status: string;
};

export type ChatDetail = {
  id: number;
  title: string | null;
  messages: ChatMessageApi[];
  artifacts: ChatArtifact[];
};

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
  session_id?: number;
};

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string | null) {
  if (token) localStorage.setItem(TOKEN_KEY, token);
  else localStorage.removeItem(TOKEN_KEY);
}

async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers || {});
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init.body && !(init.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const res = await fetch(`${API_BASE}${path}`, { ...init, headers });
  if (res.status === 401) {
    setToken(null);
  }
  return res;
}

async function readError(res: Response): Promise<string> {
  try {
    const data = await res.json();
    if (typeof data.detail === "string") return data.detail;
    if (Array.isArray(data.detail)) {
      return data.detail.map((d: { msg?: string }) => d.msg || JSON.stringify(d)).join("; ");
    }
    return JSON.stringify(data);
  } catch {
    return (await res.text()) || `Request failed (${res.status})`;
  }
}

export async function signup(email: string, password: string): Promise<AuthResponse> {
  const res = await apiFetch("/auth/signup", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function login(email: string, password: string): Promise<AuthResponse> {
  const res = await apiFetch("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function fetchMe(): Promise<User> {
  const res = await apiFetch("/auth/me");
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function listChats(): Promise<ChatSummary[]> {
  const res = await apiFetch("/chats");
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function createChat(title?: string): Promise<ChatSummary> {
  const res = await apiFetch("/chats", {
    method: "POST",
    body: JSON.stringify({ title: title ?? null }),
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function getChat(chatId: number): Promise<ChatDetail> {
  const res = await apiFetch(`/chats/${chatId}`);
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function deleteChat(chatId: number): Promise<void> {
  const res = await apiFetch(`/chats/${chatId}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await readError(res));
}

export async function uploadFile(file: File, sessionId: number): Promise<UploadResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("session_id", String(sessionId));
  const res = await apiFetch("/upload", { method: "POST", body: form });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function getJobStatus(jobId: number): Promise<JobStatus> {
  const res = await apiFetch(`/jobs/${jobId}`);
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function streamQuery(
  query: string,
  sessionId: number,
  onToken: (chunk: string) => void,
  signal?: AbortSignal,
): Promise<string> {
  const res = await apiFetch("/query-stream", {
    method: "POST",
    body: JSON.stringify({ query, session_id: sessionId }),
    signal,
  });
  if (!res.ok || !res.body) {
    throw new Error(await readError(res));
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
    if (!opts?.final && isLast && (/^\[Retrieved:/.test(s) || /^\[Response confidence:/.test(s))) {
      continue;
    }
    if (/^\[Retrieved:.*\]$/.test(s)) continue;
    if (/^\[Response confidence:.*\]$/.test(s)) continue;
    if (s.startsWith("⚠️") || s.startsWith("ℹ️") || s.startsWith("❌")) continue;
    kept.push(line);
  }

  let stripped = kept.join("\n");
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
