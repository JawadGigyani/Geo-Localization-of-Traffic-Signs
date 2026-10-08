"use client";

import { useEffect, useMemo, useRef, useState } from "react";
// aliased: maplibre exports `Map`, which would shadow the global Map constructor
import maplibregl, { Map as MapLibreMap, GeoJSONSource } from "maplibre-gl";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

// The basemap serves Noto, not MapLibre's default Open Sans / Arial Unicode.
// Leaving text-font unset makes glyph loading 404, which aborts tile layout for
// the whole source -- the circle layers vanish along with the labels.
const MAP_FONT = ["Noto Sans Bold"];
const MAP_STYLE = "https://tiles.openfreemap.org/styles/positron";

const UVM_GREEN = "#00693e";
const UVM_GOLD = "#ffb81c";
const INK = "#0a0a0a";

type Sign = {
  id: string;
  sign_type: string;
  confidence: number;
  track_id?: number;
  image_crop_url?: string;
  full_frame_url?: string;
  source_sequence?: string;
  gps_source?: string;
  sign_lat?: number | null;
  sign_lon?: number | null;
  gt_sign_id?: string | null;
  camera_lat?: number | null;
  camera_lon?: number | null;
  observations?: number | null;
  error_m?: number | null;
  lng: number;
  lat: number;
};

// Anything that is not a measured fix must be visibly labelled, so a demo point
// is never mistaken for a real detection.
const GPS_LABELS: Record<string, { text: string; real: boolean }> = {
  // projected from the widest view (bearing + range), not triangulated
  geo_projected: { text: "Projected sign position (single view)", real: true },
  frames_meta: { text: "Camera GPS (dataset)", real: true },
  xml: { text: "Camera GPS (XML)", real: true },
  gps_sidecar: { text: "Camera GPS (sidecar)", real: true },
  depth_projected: { text: "Depth-projected position", real: true },
  synthetic_demo: { text: "Synthetic placeholder", real: false },
  unknown: { text: "Unknown source", real: false },
};

function haversineMeters(lat1: number, lon1: number, lat2: number, lon2: number) {
  const toRad = (d: number) => (d * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return 6371000 * 2 * Math.asin(Math.min(1, Math.sqrt(a)));
}

function median(values: number[]) {
  if (!values.length) return null;
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

export default function HomePage() {
  const mapRef = useRef<HTMLDivElement | null>(null);
  const mapObj = useRef<MapLibreMap | null>(null);
  const [signs, setSigns] = useState<Sign[]>([]);
  const [selected, setSelected] = useState<Sign | null>(null);
  const [panelOpen, setPanelOpen] = useState(true);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");

  const stats = useMemo(() => {
    const errors = signs
      .map((s) => s.error_m)
      .filter((e): e is number => typeof e === "number");
    const byType = new Map<string, number>();
    signs.forEach((s) => byType.set(s.sign_type, (byType.get(s.sign_type) ?? 0) + 1));
    return {
      total: signs.length,
      sequences: new Set(signs.map((s) => s.source_sequence)).size,
      types: [...byType.entries()].sort((a, b) => b[1] - a[1]),
      medianError: median(errors),
      scored: errors.length,
      placeholder: signs.filter((s) => !GPS_LABELS[s.gps_source ?? "unknown"]?.real).length,
    };
  }, [signs]);

  useEffect(() => {
    if (!mapRef.current || mapObj.current) return;

    const map = new maplibregl.Map({
      container: mapRef.current,
      style: MAP_STYLE,
      center: [-72.6, 44.5],
      zoom: 7.5,
    });
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-left");
    map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-right");
    mapObj.current = map;

    map.on("load", () => {
      map.addSource("signs", {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
        cluster: true,
        clusterMaxZoom: 15,
        clusterRadius: 45,
      });

      map.addLayer({
        id: "clusters",
        type: "circle",
        source: "signs",
        filter: ["has", "point_count"],
        paint: {
          "circle-color": UVM_GREEN,
          "circle-radius": ["step", ["get", "point_count"], 16, 10, 21, 40, 27],
          "circle-stroke-width": 3,
          "circle-stroke-color": INK,
        },
      });

      map.addLayer({
        id: "cluster-count",
        type: "symbol",
        source: "signs",
        filter: ["has", "point_count"],
        layout: {
          "text-field": "{point_count_abbreviated}",
          "text-font": MAP_FONT,
          "text-size": 13,
        },
        paint: { "text-color": "#ffffff" },
      });

      map.addLayer({
        id: "unclustered",
        type: "circle",
        source: "signs",
        filter: ["!", ["has", "point_count"]],
        paint: {
          // gold = real GPS fix, grey = synthetic/unknown placeholder
          "circle-color": [
            "match",
            ["get", "gps_source"],
            "synthetic_demo", "#9ca3af",
            "unknown", "#9ca3af",
            UVM_GOLD,
          ],
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 8, 5, 14, 8, 18, 12],
          "circle-stroke-width": 2.5,
          "circle-stroke-color": INK,
        },
      });

      map.on("click", "clusters", (e) => {
        const f = map.queryRenderedFeatures(e.point, { layers: ["clusters"] })[0];
        if (!f || f.geometry.type !== "Point") return;
        const center = f.geometry.coordinates as [number, number];
        const src = map.getSource("signs") as GeoJSONSource;
        src
          .getClusterExpansionZoom(f.properties?.cluster_id as number)
          .then((zoom) => map.easeTo({ center, zoom }));
      });

      map.on("click", "unclustered", (e) => {
        const f = e.features?.[0];
        if (!f || f.geometry.type !== "Point") return;
        const p = f.properties || {};
        setSelected({
          id: String(p.id),
          sign_type: String(p.sign_type),
          confidence: Number(p.confidence),
          track_id: p.track_id != null ? Number(p.track_id) : undefined,
          image_crop_url: p.image_crop_url || undefined,
          full_frame_url: p.full_frame_url || undefined,
          source_sequence: p.source_sequence || undefined,
          gps_source: p.gps_source || "unknown",
          sign_lat: p.sign_lat ?? null,
          sign_lon: p.sign_lon ?? null,
          gt_sign_id: p.gt_sign_id ?? null,
          camera_lat: p.camera_lat ?? null,
          camera_lon: p.camera_lon ?? null,
          observations: p.observations ?? null,
          error_m: p.error_m ?? null,
          lng: (f.geometry.coordinates as number[])[0],
          lat: (f.geometry.coordinates as number[])[1],
        });
        setPanelOpen(true);
      });

      ["clusters", "unclustered"].forEach((id) => {
        map.on("mouseenter", id, () => (map.getCanvas().style.cursor = "pointer"));
        map.on("mouseleave", id, () => (map.getCanvas().style.cursor = ""));
      });

      // First load sweeps all of Vermont and zooms to whatever exists, so the
      // map never opens empty just because the default centre is somewhere the
      // survey did not drive.
      let fitted = false;

      const loadSigns = async (sweepAll = false) => {
        const b = map.getBounds();
        const bbox = sweepAll
          ? "-74.0,42.0,-71.0,45.6"
          : `${b.getWest()},${b.getSouth()},${b.getEast()},${b.getNorth()}`;
        try {
          const res = await fetch(`${API_BASE}/signs?bbox=${encodeURIComponent(bbox)}`);
          if (!res.ok) {
            setStatus("error");
            return;
          }
          const data = await res.json();
          const rows: Sign[] = data.signs || [];
          setSigns(rows);
          setStatus("ready");

          (map.getSource("signs") as GeoJSONSource).setData({
            type: "FeatureCollection",
            features: rows.map((s) => ({
              type: "Feature" as const,
              geometry: { type: "Point" as const, coordinates: [s.lng, s.lat] },
              properties: { ...s },
            })),
          });

          if (sweepAll && !fitted && rows.length) {
            fitted = true;
            const lons = rows.map((s) => s.lng);
            const lats = rows.map((s) => s.lat);
            // Padding must leave room for the side panel but never exceed the
            // viewport: fitBounds refuses to fit when padding is wider than the
            // map, which is easy to hit on a narrow window.
            const { width, height } = map.getCanvas().getBoundingClientRect();
            const right = Math.min(400, Math.max(20, width * 0.3));
            const side = Math.min(50, width * 0.05);
            const vert = Math.min(70, height * 0.08);
            map.fitBounds(
              [
                [Math.min(...lons), Math.min(...lats)],
                [Math.max(...lons), Math.max(...lats)],
              ],
              { padding: { top: vert, bottom: vert, left: side, right }, maxZoom: 16, duration: 0 }
            );
          }
        } catch {
          setStatus("error");
        }
      };

      loadSigns(true);
      map.on("moveend", () => window.setTimeout(() => loadSigns(false), 250));
    });

    return () => {
      map.remove();
      mapObj.current = null;
    };
  }, []);

  const flyToSign = (s: Sign) => {
    setSelected(s);
    mapObj.current?.easeTo({ center: [s.lng, s.lat], zoom: 17, duration: 600 });
  };

  const gps = GPS_LABELS[selected?.gps_source ?? "unknown"];

  return (
    <main className="relative h-screen w-screen overflow-hidden bg-white text-black">
      {/*
        The map must be sized with h-full/w-full, NOT `absolute inset-0`.
        maplibre-gl.css declares `.maplibregl-map { position: relative }` with the
        same specificity as Tailwind's `.absolute` and loads after it, so the
        element falls back to static flow and collapses to zero height.
      */}
      <div ref={mapRef} data-testid="map" className="h-full w-full" />

      {/* header */}
      <header className="pointer-events-none absolute inset-x-0 top-0 z-20 flex items-start justify-between gap-4 p-4">
        <div className="nb pointer-events-auto bg-white px-4 py-2.5">
          <h1 className="text-lg font-bold uppercase leading-none tracking-tight">
            Traffic Sign <span style={{ color: UVM_GREEN }}>Inventory</span>
          </h1>
          <p className="mt-1 text-[11px] font-semibold uppercase tracking-widest text-neutral-600">
            ARTSv2 · YOLO11 · PostGIS
          </p>
        </div>

        <div className="pointer-events-auto flex items-center gap-2">
          <span
            className={`nb-thin px-3 py-2 text-xs font-bold uppercase tracking-wide ${
              status === "error" ? "bg-red-400" : "bg-white"
            }`}
          >
            {status === "loading" && "Loading…"}
            {status === "error" && "API unreachable"}
            {status === "ready" && `${stats.total} signs in view`}
          </span>
          {!panelOpen && (
            <button
              type="button"
              onClick={() => setPanelOpen(true)}
              className="nb-thin nb-press px-3 py-2 text-xs font-bold uppercase tracking-wide text-white"
              style={{ background: UVM_GREEN }}
            >
              Details
            </button>
          )}
        </div>
      </header>

      {/* legend */}
      <div className="nb absolute bottom-5 left-4 z-20 bg-white px-3.5 py-3">
        <p className="mb-2 text-[11px] font-bold uppercase tracking-widest">Map key</p>
        <div className="flex items-center gap-2 text-xs font-semibold">
          <span
            className="inline-block h-3.5 w-3.5 rounded-full border-2 border-black"
            style={{ background: UVM_GOLD }}
          />
          Individual sign
        </div>
        <div className="mt-1.5 flex items-center gap-2 text-xs font-semibold">
          <span
            className="inline-block h-4 w-4 rounded-full border-2 border-black"
            style={{ background: UVM_GREEN }}
          />
          Cluster — click to expand
        </div>
      </div>

      {/* side panel */}
      <aside
        data-testid="panel"
        className={`absolute right-0 top-0 z-30 flex h-full w-full max-w-sm flex-col border-l-[3px] border-black bg-white transition-transform duration-150 ${
          panelOpen ? "translate-x-0" : "translate-x-full"
        }`}
      >
        <div
          className="flex items-center justify-between border-b-[3px] border-black px-4 py-3"
          style={{ background: UVM_GREEN }}
        >
          <h2 className="text-sm font-bold uppercase tracking-widest text-white">
            {selected ? "Sign detail" : "Inventory"}
          </h2>
          <div className="flex items-center gap-2">
            {selected && (
              <button
                type="button"
                onClick={() => setSelected(null)}
                className="border-2 border-black bg-white px-2 py-0.5 text-[11px] font-bold uppercase"
              >
                Back
              </button>
            )}
            <button
              type="button"
              onClick={() => setPanelOpen(false)}
              aria-label="Hide panel"
              data-testid="panel-close"
              className="border-2 border-black bg-white px-2 py-0.5 text-[11px] font-bold leading-tight"
            >
              ✕
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto">
          {!selected ? (
            <div className="space-y-6 p-4">
              <div className="grid grid-cols-2 gap-3">
                <Stat label="Signs in view" value={String(stats.total)} accent />
                <Stat label="Sequences" value={String(stats.sequences)} />
                <Stat
                  label="Median error"
                  value={stats.medianError != null ? `${stats.medianError.toFixed(1)} m` : "—"}
                  hint={stats.scored ? `${stats.scored} scored` : undefined}
                  accent
                />
                <Stat label="Sign types" value={String(stats.types.length)} />
              </div>

              {stats.placeholder > 0 && (
                <p className="nb-thin bg-red-300 px-3 py-2 text-xs font-bold uppercase">
                  {stats.placeholder} placeholder coordinate
                  {stats.placeholder === 1 ? "" : "s"} — not measured
                </p>
              )}

              <section>
                <h3 className="mb-2.5 text-[11px] font-bold uppercase tracking-widest">
                  Most common types
                </h3>
                <ul className="space-y-2">
                  {stats.types.slice(0, 10).map(([type, count]) => (
                    <li key={type}>
                      <div className="flex items-center justify-between text-sm font-bold">
                        <span className="font-mono">{type}</span>
                        <span className="tabular-nums">{count}</span>
                      </div>
                      <div className="mt-1 h-3 border-2 border-black bg-white">
                        <div
                          className="h-full"
                          style={{
                            width: `${(count / stats.types[0][1]) * 100}%`,
                            background: UVM_GREEN,
                          }}
                        />
                      </div>
                    </li>
                  ))}
                </ul>
              </section>

              <section>
                <h3 className="mb-2.5 text-[11px] font-bold uppercase tracking-widest">
                  Detections
                </h3>
                <ul className="space-y-2">
                  {signs.slice(0, 30).map((s) => (
                    <li key={s.id}>
                      <button
                        type="button"
                        onClick={() => flyToSign(s)}
                        data-testid="detection-item"
                        className="nb-thin nb-press flex w-full items-center gap-3 bg-white p-2 text-left"
                      >
                        {s.image_crop_url ? (
                          // eslint-disable-next-line @next/next/no-img-element
                          <img
                            src={s.image_crop_url}
                            alt=""
                            className="h-10 w-10 shrink-0 border-2 border-black object-cover"
                          />
                        ) : (
                          <span className="h-10 w-10 shrink-0 border-2 border-black bg-neutral-200" />
                        )}
                        <span className="min-w-0 flex-1">
                          <span className="block font-mono text-sm font-bold">{s.sign_type}</span>
                          <span className="block truncate text-[11px] font-semibold uppercase tracking-wide text-neutral-600">
                            {s.observations ?? 1} view{(s.observations ?? 1) === 1 ? "" : "s"}
                            {s.error_m != null && ` · ${s.error_m.toFixed(1)} m`}
                          </span>
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              </section>
            </div>
          ) : (
            <div className="space-y-4 p-4">
              {selected.image_crop_url ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={selected.image_crop_url}
                  alt={`Crop of sign ${selected.sign_type}`}
                  className="nb w-full bg-neutral-100 object-contain"
                  style={{ maxHeight: 210 }}
                />
              ) : (
                <div className="nb flex h-28 items-center justify-center bg-neutral-100 text-xs font-bold uppercase">
                  No crop stored
                </div>
              )}

              <div>
                <p className="font-mono text-3xl font-bold" style={{ color: UVM_GREEN }}>
                  {selected.sign_type}
                </p>
                <p className="mt-1 text-[11px] font-semibold uppercase tracking-wide text-neutral-600">
                  {selected.source_sequence}
                  {selected.track_id != null && ` · track ${selected.track_id}`}
                </p>
              </div>

              <div>
                <div className="mb-1 flex items-baseline justify-between text-[11px] font-bold uppercase tracking-widest">
                  <span>Confidence</span>
                  <span className="tabular-nums">{(selected.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="h-4 border-2 border-black bg-white">
                  <div
                    className="h-full"
                    style={{
                      width: `${Math.min(100, selected.confidence * 100)}%`,
                      background: UVM_GOLD,
                    }}
                  />
                </div>
              </div>

              <dl className="nb-thin space-y-1.5 bg-white p-3 text-sm">
                <Row label="Position" value={`${selected.lat.toFixed(5)}, ${selected.lng.toFixed(5)}`} />
                <Row
                  label="Observed in"
                  value={`${selected.observations ?? 1} frame${(selected.observations ?? 1) === 1 ? "" : "s"}`}
                />
                {selected.camera_lat != null && selected.camera_lon != null && (
                  <Row
                    label="Range"
                    value={`${haversineMeters(
                      selected.camera_lat,
                      selected.camera_lon,
                      selected.lat,
                      selected.lng
                    ).toFixed(0)} m from camera`}
                  />
                )}
              </dl>

              <p
                className={`nb-thin px-3 py-2 text-[11px] font-bold uppercase tracking-wide ${
                  gps?.real ? "bg-white" : "bg-red-300"
                }`}
              >
                {gps?.text ?? selected.gps_source}
              </p>

              {selected.sign_lat != null && selected.sign_lon != null && (
                <section className="nb p-3.5" style={{ background: UVM_GREEN }}>
                  <h3 className="mb-1 text-[11px] font-bold uppercase tracking-widest text-white/80">
                    Error vs ground truth
                  </h3>
                  <p className="text-4xl font-bold tabular-nums text-white">
                    {(
                      selected.error_m ??
                      haversineMeters(selected.lat, selected.lng, selected.sign_lat, selected.sign_lon)
                    ).toFixed(1)}
                    <span className="ml-1 text-lg font-semibold text-white/70">m</span>
                  </p>
                  {selected.camera_lat != null && selected.camera_lon != null && (
                    <p className="mt-1.5 text-xs font-semibold text-white/85">
                      Camera pin would be{" "}
                      {haversineMeters(
                        selected.camera_lat,
                        selected.camera_lon,
                        selected.sign_lat,
                        selected.sign_lon
                      ).toFixed(1)}{" "}
                      m off
                    </p>
                  )}
                </section>
              )}

              {selected.full_frame_url && (
                <a
                  href={selected.full_frame_url}
                  target="_blank"
                  rel="noreferrer"
                  className="nb-thin nb-press inline-block bg-white px-3 py-2 text-xs font-bold uppercase"
                >
                  Open full frame
                </a>
              )}
            </div>
          )}
        </div>
      </aside>
    </main>
  );
}

function Stat({
  label,
  value,
  hint,
  accent,
}: {
  label: string;
  value: string;
  hint?: string;
  accent?: boolean;
}) {
  return (
    <div className="nb-thin bg-white p-2.5">
      <p className="text-[10px] font-bold uppercase tracking-widest text-neutral-600">{label}</p>
      <p
        className="mt-0.5 text-2xl font-bold tabular-nums"
        style={accent ? { color: UVM_GREEN } : undefined}
      >
        {value}
      </p>
      {hint && <p className="text-[10px] font-semibold uppercase text-neutral-500">{hint}</p>}
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <dt className="shrink-0 text-[10px] font-bold uppercase tracking-widest text-neutral-600">
        {label}
      </dt>
      <dd className="text-right font-mono text-[11px] font-semibold">{value}</dd>
    </div>
  );
}
