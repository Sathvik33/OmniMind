import "./AmbientBg.css";

export function AmbientBg() {
  return (
    <div className="ambient" aria-hidden>
      <div className="ambient__base" />
      <div className="ambient__wash" />
      <div className="ambient__grain" />
    </div>
  );
}
