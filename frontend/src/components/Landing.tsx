import { motion } from "framer-motion";
import "./Landing.css";

type Props = {
  onEnter: () => void;
};

export function Landing({ onEnter }: Props) {
  return (
    <section className="landing">
      <motion.header
        className="landing__brand"
        initial={{ opacity: 0, y: -12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
      >
        <img src="/aegis.svg" alt="" width={36} height={36} />
        <span>AEGIS</span>
      </motion.header>

      <div className="landing__stage">
        <div className="landing__copy">
          <motion.h1
            initial={{ opacity: 0, y: 28 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.85, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
          >
            AEGIS
          </motion.h1>
          <motion.p
            className="landing__lede"
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.25, ease: [0.22, 1, 0.36, 1] }}
          >
            Ground every answer in your documents, images, and video — hybrid
            retrieval with guardrails.
          </motion.p>
          <motion.div
            className="landing__cta"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.75, delay: 0.4, ease: [0.22, 1, 0.36, 1] }}
          >
            <button type="button" className="btn-primary" onClick={onEnter}>
              Open workspace
            </button>
            <span className="landing__hint">Upload · Ask · Stream</span>
          </motion.div>
        </div>

        <motion.div
          className="landing__visual"
          initial={{ opacity: 0, scale: 0.96 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 1, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
          aria-hidden
        >
          <div className="visual-plane">
            <motion.div
              className="visual-ring"
              animate={{ rotate: 360 }}
              transition={{ duration: 48, repeat: Infinity, ease: "linear" }}
            />
            <motion.div
              className="visual-core"
              animate={{ scale: [1, 1.04, 1] }}
              transition={{ duration: 5.5, repeat: Infinity, ease: "easeInOut" }}
            />
            <div className="visual-shards">
              <span />
              <span />
              <span />
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
