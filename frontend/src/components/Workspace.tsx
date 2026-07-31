import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Composer } from "./Composer";
import { MessageList, type ChatMessage } from "./MessageList";
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

export function Workspace({ onBack }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [jobId, setJobId] = useState<number | null>(null);
  const [pendingArtifactId, setPendingArtifactId] = useState<number | null>(null);
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
    const maxPolls = 240; // ~6 minutes at 1.5s

    const tick = async () => {
      try {
        const status = await getJobStatus(jobId);
        if (cancelled) return;
        polls += 1;
        const s = (status.status || "").toLowerCase();
        setJobLabel(`Embedding · ${s.replaceAll("_", " ")} — ask after this finishes`);
        if (s === "completed") {
          const readyId = status.artifact_id ?? pendingArtifactId;
          if (readyId != null) {
            setReadyArtifactIds((prev) =>
              prev.includes(readyId) ? prev : [...prev, readyId],
            );
          }
          setJobId(null);
          setPendingArtifactId(null);
          setJobLabel(null);
          setError(null);
          return;
        }
        if (s === "failed" || s === "dead_letter") {
          setJobId(null);
          setPendingArtifactId(null);
          setJobLabel(null);
          setError(status.error_message || "Ingestion failed");
          return;
        }
        if (polls >= maxPolls) {
          setJobId(null);
          setPendingArtifactId(null);
          setJobLabel(null);
          setError("Ingestion is taking too long. Check the Celery worker and retry.");
          return;
        }
        window.setTimeout(tick, 1500);
      } catch (e) {
        if (!cancelled) {
          setJobId(null);
          setPendingArtifactId(null);
          setJobLabel(null);
          setError(e instanceof Error ? e.message : "Could not poll job status");
        }
      }
    };
    tick();
    return () => {
      cancelled = true;
    };
  }, [jobId, pendingArtifactId]);

  const onUpload = async (file: File) => {
    setError(null);
    setUploading(true);
    setJobLabel(`Uploading ${file.name}`);
    try {
      const result = await uploadFile(file);
      setPendingArtifactId(result.artifact_id ?? null);
      setJobId(result.job_id);
      setJobLabel("Queued for embedding — retrieval locked until done");
    } catch (e) {
      setJobLabel(null);
      setPendingArtifactId(null);
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const onSend = async (text: string) => {
    if (jobId != null || uploading) {
      setError("Wait until embedding finishes before asking questions.");
      return;
    }
    if (!readyArtifactIds.length) {
      setError("Upload a file and wait for embedding to finish before asking.");
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
      const finalText = cleanStreamText(raw) || "No response received.";
      setMessages((prev) =>
        prev.map((m) => (m.id === assistantId ? { ...m, content: finalText } : m)),
      );
    } catch (e) {
      if ((e as Error).name === "AbortError") return;
      const msg = e instanceof Error ? e.message : "Query failed";
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? { ...m, content: m.content || `Could not reach AEGIS backend: ${msg}` }
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
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
    >
      <header className="workspace__top">
        <button type="button" className="ghost-btn" onClick={onBack}>
          Back
        </button>
        <div className="workspace__brand">
          <img src="/aegis.svg" alt="" width={28} height={28} />
          <div>
            <strong>AEGIS</strong>
            <span>
              {readyArtifactIds.length
                ? `Answering from ${readyArtifactIds.length} ready upload${readyArtifactIds.length === 1 ? "" : "s"}`
                : "Multimodal RAG workspace"}
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
            setError(null);
          }}
        >
          Clear chat
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
