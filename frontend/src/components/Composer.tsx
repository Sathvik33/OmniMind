import { useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import "./Composer.css";

type Props = {
  disabled?: boolean;
  uploading?: boolean;
  jobLabel?: string | null;
  onSend: (text: string) => void;
  onUpload: (file: File) => void;
};

const ACCEPT = ".pdf,.txt,.docx,.pptx,.xlsx,.png,.jpg,.jpeg,.mp4,.mov,.avi";

export function Composer({ disabled, uploading, jobLabel, onSend, onUpload }: Props) {
  const [text, setText] = useState("");
  const [showUpload, setShowUpload] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const submit = () => {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setText("");
  };

  return (
    <div className="composer">
      <AnimatePresence>
        {jobLabel ? (
          <motion.div
            className="composer__status"
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 4 }}
          >
            <span className="pulse-dot" />
            {jobLabel}
          </motion.div>
        ) : null}
      </AnimatePresence>

      <AnimatePresence>
        {showUpload ? (
          <motion.div
            className="upload-tray"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
          >
            <button
              type="button"
              className="upload-drop"
              disabled={uploading}
              onClick={() => fileRef.current?.click()}
            >
              <strong>{uploading ? "Uploading…" : "Choose a file"}</strong>
              <span>PDF, Word, slides, images, or video</span>
            </button>
            <input
              ref={fileRef}
              className="sr-only"
              type="file"
              accept={ACCEPT}
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) {
                  onUpload(file);
                  setShowUpload(false);
                  e.target.value = "";
                }
              }}
            />
          </motion.div>
        ) : null}
      </AnimatePresence>

      <div className="composer__bar">
        <button
          type="button"
          className={`icon-btn ${showUpload ? "icon-btn--active" : ""}`}
          aria-label="Attach file"
          disabled={uploading}
          onClick={() => setShowUpload((v) => !v)}
        >
          +
        </button>
        <textarea
          rows={1}
          value={text}
          disabled={disabled || uploading}
          placeholder={
            uploading
              ? "Hang on while the file is prepared…"
              : disabled
                ? "Upload a file first, then ask…"
                : "Ask about what you uploaded…"
          }
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
        />
        <button
          type="button"
          className="send-btn"
          disabled={disabled || uploading || !text.trim()}
          onClick={submit}
        >
          Send
        </button>
      </div>
    </div>
  );
}
