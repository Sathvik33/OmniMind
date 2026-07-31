import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Composer } from "./Composer";
import { MessageList, type ChatMessage, type DocStatus } from "./MessageList";
import {
  cleanStreamText,
  getJobStatus,
  streamQuery,
  uploadFile,
} from "../api/client";
import "./Workspace.css";

type Props = {
  onBack: () => void;
};

function uid() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function fileExt(name: string): string {
  const parts = name.split(".");
  return parts.length > 1 ? parts.pop()!.slice(0, 4).toUpperCase() : "FILE";
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function updateDocMessage(
  messages: ChatMessage[],
  docId: string,
  patch: Partial<NonNullable<ChatMessage["document"]>>,
): ChatMessage[] {
  return messages.map((m) =>
    m.id === docId && m.document
      ? { ...m, document: { ...m.document, ...patch } }
      : m,
  );
}

export function Workspace({ onBack }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [jobId, setJobId] = useState<number | null>(null);
  const [pendingArtifactId, setPendingArtifactId] = useState<number | null>(null);
  const [pendingDocMsgId, setPendingDocMsgId] = useState<string | null>(null);
  const [readyArtifactIds, setReadyArtifactIds] = useState<number[]>([]);
  const [jobLabel, setJobLabel] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const scrollerRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const el = scrollerRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, streaming, jobLabel]);

  useEffect(() => {
    if (jobId == null) return;
    let cancelled = false;
    let polls = 0;
    const maxPolls = 240;

    const tick = async () => {
      try {
        const status = await getJobStatus(jobId);
        if (cancelled) return;
        polls += 1;
        const s = (status.status || "").toLowerCase();
        const human = s.replaceAll("_", " ");
        setJobLabel(`Still working · ${human}`);
        if (pendingDocMsgId) {
          setMessages((prev) =>
            updateDocMessage(prev, pendingDocMsgId, {
              status: "processing" as DocStatus,
              statusLabel: `Preparing · ${human}`,
            }),
          );
        }
        if (s === "completed") {
          const readyId = status.artifact_id ?? pendingArtifactId;
          if (readyId != null) {
            setReadyArtifactIds((prev) =>
              prev.includes(readyId) ? prev : [...prev, readyId],
            );
          }
          if (pendingDocMsgId) {
            setMessages((prev) =>
              updateDocMessage(prev, pendingDocMsgId, {
                status: "ready",
                statusLabel: "Ready — ask away",
              }),
            );
          }
          setJobId(null);
          setPendingArtifactId(null);
          setPendingDocMsgId(null);
          setJobLabel(null);
          setError(null);
          return;
        }
        if (s === "failed" || s === "dead_letter") {
          if (pendingDocMsgId) {
            setMessages((prev) =>
              updateDocMessage(prev, pendingDocMsgId, {
                status: "failed",
                statusLabel: status.error_message || "Processing failed",
              }),
            );
          }
          setJobId(null);
          setPendingArtifactId(null);
          setPendingDocMsgId(null);
          setJobLabel(null);
          setError(status.error_message || "Ingestion failed");
          return;
        }
        if (polls >= maxPolls) {
          if (pendingDocMsgId) {
            setMessages((prev) =>
              updateDocMessage(prev, pendingDocMsgId, {
                status: "failed",
                statusLabel: "Taking too long",
              }),
            );
          }
          setJobId(null);
          setPendingArtifactId(null);
          setPendingDocMsgId(null);
          setJobLabel(null);
          setError("This is taking too long. Check the worker and try again.");
          return;
        }
        window.setTimeout(tick, 1500);
      } catch (e) {
        if (!cancelled) {
          if (pendingDocMsgId) {
            setMessages((prev) =>
              updateDocMessage(prev, pendingDocMsgId, {
                status: "failed",
                statusLabel: "Status check failed",
              }),
            );
          }
          setJobId(null);
          setPendingArtifactId(null);
          setPendingDocMsgId(null);
          setJobLabel(null);
          setError(e instanceof Error ? e.message : "Could not poll job status");
        }
      }
    };
    tick();
    return () => {
      cancelled = true;
    };
  }, [jobId, pendingArtifactId, pendingDocMsgId]);

  const onUpload = async (file: File) => {
    setError(null);
    const docId = uid();
    setMessages((prev) => [
      ...prev,
      {
        id: docId,
        role: "document",
        content: file.name,
        document: {
          name: file.name,
          sizeLabel: formatBytes(file.size),
          ext: fileExt(file.name),
          status: "uploading",
          statusLabel: "Uploading…",
        },
      },
    ]);
    setUploading(true);
    setPendingDocMsgId(docId);
    setJobLabel(`Uploading ${file.name}`);
    try {
      const result = await uploadFile(file);
      setPendingArtifactId(result.artifact_id ?? null);
      setJobId(result.job_id);
      setMessages((prev) =>
        updateDocMessage(prev, docId, {
          status: "processing",
          statusLabel: "Queued — preparing for search…",
        }),
      );
      setJobLabel("Queued — you’ll be able to ask once it’s ready");
    } catch (e) {
      setMessages((prev) =>
        updateDocMessage(prev, docId, {
          status: "failed",
          statusLabel: e instanceof Error ? e.message : "Upload failed",
        }),
      );
      setJobLabel(null);
      setPendingArtifactId(null);
      setPendingDocMsgId(null);
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const onSend = async (text: string) => {
    if (jobId != null || uploading) {
      setError("Wait until the file is ready before asking.");
      return;
    }
    if (!readyArtifactIds.length) {
      setError("Upload a file and wait until it’s ready before asking.");
      return;
    }

    setError(null);
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    const userMsg: ChatMessage = { id: uid(), role: "user", content: text };
    const assistantId = uid();
    setMessages((prev) => [
      ...prev,
      userMsg,
      { id: assistantId, role: "assistant", content: "" },
    ]);
    setStreaming(true);

    let raw = "";
    try {
      await streamQuery(
        text,
        (chunk) => {
          raw += chunk;
          const cleaned = cleanStreamText(raw);
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, content: cleaned } : m)),
          );
        },
        controller.signal,
        readyArtifactIds,
      );
      const finalText = cleanStreamText(raw, { final: true }) || "No response received.";
      setMessages((prev) =>
        prev.map((m) => (m.id === assistantId ? { ...m, content: finalText } : m)),
      );
    } catch (e) {
      if ((e as Error).name === "AbortError") return;
      const msg = e instanceof Error ? e.message : "Query failed";
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? { ...m, content: m.content || `Couldn’t reach Aegis: ${msg}` }
            : m,
        ),
      );
      setError(msg);
    } finally {
      setStreaming(false);
    }
  };

  const ingesting = uploading || jobId != null;
  const canAsk = !streaming && !ingesting && readyArtifactIds.length > 0;

  return (
    <motion.section
      className="workspace"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
    >
      <header className="workspace__top">
        <button type="button" className="ghost-btn" onClick={onBack}>
          Back
        </button>
        <div className="workspace__brand">
          <img src="/aegis.svg" alt="" width={28} height={28} />
          <div>
            <strong>Aegis</strong>
            <span>
              {readyArtifactIds.length
                ? `${readyArtifactIds.length} file${readyArtifactIds.length === 1 ? "" : "s"} ready`
                : "Your files, your answers"}
            </span>
          </div>
        </div>
        <button
          type="button"
          className="ghost-btn"
          onClick={() => {
            abortRef.current?.abort();
            setMessages([]);
            setReadyArtifactIds([]);
            setPendingDocMsgId(null);
            setError(null);
          }}
        >
          Clear
        </button>
      </header>

      <div className="workspace__panel">
        <div className="workspace__scroll" ref={scrollerRef}>
          <MessageList messages={messages} streaming={streaming} />
        </div>

        {error ? <div className="workspace__error">{error}</div> : null}

        <Composer
          disabled={!canAsk}
          uploading={ingesting}
          jobLabel={jobLabel}
          onSend={onSend}
          onUpload={onUpload}
        />
      </div>
    </motion.section>
  );
}
