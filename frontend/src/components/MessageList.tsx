import { AnimatePresence, motion } from "framer-motion";
import "./MessageList.css";

export type DocStatus = "uploading" | "processing" | "ready" | "failed";

export type ChatMessage = {
  id: string;
  role: "user" | "assistant" | "document";
  content: string;
  document?: {
    name: string;
    sizeLabel?: string;
    ext: string;
    status: DocStatus;
    statusLabel?: string;
  };
};

type Props = {
  messages: ChatMessage[];
  streaming?: boolean;
};

function formatStatus(status: DocStatus, label?: string): string {
  if (label) return label;
  switch (status) {
    case "uploading":
      return "Uploading…";
    case "processing":
      return "Preparing for search…";
    case "ready":
      return "Ready to ask about";
    case "failed":
      return "Couldn’t process";
  }
}

export function MessageList({ messages, streaming }: Props) {
  if (messages.length === 0) {
    return (
      <motion.div
        className="empty-chat"
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45 }}
      >
        <p className="empty-chat__title">Drop something in</p>
        <p className="empty-chat__text">
          Upload a document, slide deck, image, or video. It’ll show up here,
          then you can ask about it.
        </p>
      </motion.div>
    );
  }

  return (
    <div className="message-list">
      <AnimatePresence initial={false}>
        {messages.map((msg, i) => {
          if (msg.role === "document" && msg.document) {
            const doc = msg.document;
            return (
              <motion.article
                key={msg.id}
                className="msg msg--document"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
              >
                <span className="msg__who">Uploaded</span>
                <div className="doc-card">
                  <div className="doc-card__icon" aria-hidden>
                    {doc.ext}
                  </div>
                  <div className="doc-card__meta">
                    <span className="doc-card__name" title={doc.name}>
                      {doc.name}
                    </span>
                    {doc.sizeLabel ? (
                      <span className="doc-card__sub">{doc.sizeLabel}</span>
                    ) : null}
                    <span className={`doc-card__status doc-card__status--${doc.status}`}>
                      {formatStatus(doc.status, doc.statusLabel)}
                    </span>
                  </div>
                </div>
              </motion.article>
            );
          }

          return (
            <motion.article
              key={msg.id}
              className={`msg msg--${msg.role}`}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{
                duration: 0.3,
                delay: Math.min(i * 0.02, 0.12),
                ease: [0.22, 1, 0.36, 1],
              }}
            >
              <span className="msg__who">{msg.role === "user" ? "You" : "Aegis"}</span>
              <div className="msg__body">
                {msg.content}
                {streaming && i === messages.length - 1 && msg.role === "assistant" ? (
                  <span className="caret" />
                ) : null}
              </div>
            </motion.article>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
