import { motion } from "framer-motion";
import "./AmbientBg.css";

export function AmbientBg() {
  return (
    <div className="ambient" aria-hidden>
      <div className="ambient__base" />
      <div className="ambient__grid" />
      <motion.div
        className="ambient__orb ambient__orb--teal"
        animate={{ x: [0, 40, -20, 0], y: [0, -30, 20, 0], scale: [1, 1.08, 0.96, 1] }}
        transition={{ duration: 18, repeat: Infinity, ease: "easeInOut" }}
      />
      <motion.div
        className="ambient__orb ambient__orb--sand"
        animate={{ x: [0, -50, 25, 0], y: [0, 35, -15, 0], scale: [1, 0.94, 1.06, 1] }}
        transition={{ duration: 22, repeat: Infinity, ease: "easeInOut" }}
      />
      <motion.div
        className="ambient__beam"
        animate={{ opacity: [0.15, 0.35, 0.2, 0.15], rotate: [0, 4, -3, 0] }}
        transition={{ duration: 14, repeat: Infinity, ease: "easeInOut" }}
      />
      <div className="ambient__noise" />
    </div>
  );
}
