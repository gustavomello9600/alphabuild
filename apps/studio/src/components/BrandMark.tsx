import { useAppStore } from "../store/useAppStore";
import p1Studio from "../assets/brand/p1_studio_neon_squircle.svg";
import p2Prism from "../assets/brand/p2_prism_brand_mark.svg";
import p3Cad from "../assets/brand/p3_structural_cad.svg";
import p4aDark from "../assets/brand/p4a_minimal_dark.svg";
import p4bLight from "../assets/brand/p4b_minimal_light.svg";
import p5Layered from "../assets/brand/p5_layered_depth.svg";

export type BrandVariant = "studio" | "prism" | "cad" | "minimal" | "layered";

interface BrandMarkProps {
  variant?: BrandVariant;
  size?: number;
  alt?: string;
  className?: string;
}

/**
 * Logo choice by context:
 * - studio: neon squircle for the main dark shell/welcome state (light uses p4b)
 * - prism: high-contrast app icon / compact controls
 * - cad: technical/structural surfaces such as empty compare slots
 * - minimal: small inline marks and light-theme chrome
 * - layered: studio co-design surfaces
 */
export function BrandMark({
  variant = "studio",
  size = 28,
  alt = "AlphaBuilder",
  className = "",
}: BrandMarkProps) {
  const theme = useAppStore((state) => state.theme);
  const light = theme === "tcc-light";
  const source =
    variant === "prism"
      ? p2Prism
      : variant === "cad"
        ? light
          ? p2Prism
          : p3Cad
        : variant === "layered"
          ? light
            ? p2Prism
            : p5Layered
          : variant === "minimal"
            ? light
              ? p4bLight
              : p4aDark
            : light
              ? p4bLight
              : p1Studio;
  return (
    <img
      className={`brand-mark ${className}`.trim()}
      src={source}
      width={size}
      height={size}
      alt={alt}
      draggable={false}
      decoding="async"
    />
  );
}
