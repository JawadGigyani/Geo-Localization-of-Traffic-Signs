-- Traffic Sign Inventory - Supabase / PostGIS schema
-- Run in: Supabase Dashboard -> SQL Editor
-- Then create Storage buckets: sign-crops, sign-frames (public read for the demo)
--
-- Safe to re-run. On a database created by an older version of this file, the
-- ALTER TABLE statements below will add the new columns in place.

create extension if not exists postgis;

create table if not exists traffic_signs (
  id uuid primary key default gen_random_uuid(),
  sign_type text not null,
  confidence real not null,
  track_id integer,
  capture_timestamp timestamptz,
  image_crop_url text,
  full_frame_url text,
  source_sequence text,
  -- the map pin: the estimated SIGN position (gps_source 'geo_projected'), or
  -- the camera's GPS for rows written by the tracker baseline
  geom geography(Point, 4326) not null,
  -- where that coordinate came from. 'none' rows are never inserted; the UI
  -- shows a warning badge for anything that is not a real fix.
  gps_source text not null default 'unknown',
  -- where the camera was when the sign was captured (geom is the SIGN)
  camera_lat double precision,
  camera_lon double precision,
  -- how many frames this physical sign was seen in
  observations integer default 1,
  -- ARTSv2 ground truth for the geo-localization error metric
  sign_lat double precision,
  sign_lon double precision,
  gt_sign_id text,
  created_at timestamptz default now()
);

-- Idempotent upgrades for an existing table
alter table traffic_signs add column if not exists gps_source text not null default 'unknown';
alter table traffic_signs add column if not exists gt_sign_id text;
alter table traffic_signs add column if not exists sign_lat double precision;
alter table traffic_signs add column if not exists sign_lon double precision;
alter table traffic_signs add column if not exists camera_lat double precision;
alter table traffic_signs add column if not exists camera_lon double precision;
alter table traffic_signs add column if not exists observations integer default 1;

-- Recreate the constraint so 'geo_projected' is accepted. A plain
-- "add column if not exists" cannot alter an existing check.
alter table traffic_signs drop constraint if exists traffic_signs_gps_source_check;
alter table traffic_signs add constraint traffic_signs_gps_source_check
  check (gps_source in ('frames_meta', 'xml', 'gps_sidecar', 'geo_projected',
                        'depth_projected', 'synthetic_demo', 'unknown'));

create index if not exists traffic_signs_geom_idx on traffic_signs using gist (geom);
create index if not exists traffic_signs_type_idx on traffic_signs (sign_type);
create index if not exists traffic_signs_seq_idx  on traffic_signs (source_sequence);

-- Row Level Security: the backend uses the service_role key (which bypasses
-- RLS), so the anon key gets read-only access and nothing else.
alter table traffic_signs enable row level security;

drop policy if exists traffic_signs_read on traffic_signs;
create policy traffic_signs_read on traffic_signs
  for select to anon, authenticated using (true);

-- Drop any older versions of the two functions first. Postgres cannot "replace"
-- a function whose argument list or return type changed -- it would create a
-- second overload and make RPC calls ambiguous.
do $$
declare r record;
begin
  for r in
    select oid::regprocedure as sig
    from pg_proc
    where proname in ('insert_traffic_sign', 'signs_in_bbox')
  loop
    execute 'drop function if exists ' || r.sig || ' cascade';
  end loop;
end $$;

-- Insert helper (keeps geography encoding out of the Python client)
create or replace function insert_traffic_sign(
  p_sign_type text,
  p_confidence real,
  p_track_id integer,
  p_lat double precision,
  p_lng double precision,
  p_gps_source text default 'unknown',
  p_image_crop_url text default null,
  p_full_frame_url text default null,
  p_source_sequence text default null,
  p_capture_timestamp timestamptz default null,
  p_sign_lat double precision default null,
  p_sign_lon double precision default null,
  p_gt_sign_id text default null,
  p_camera_lat double precision default null,
  p_camera_lon double precision default null,
  p_observations integer default 1
)
returns uuid
language plpgsql
as $$
declare
  new_id uuid;
begin
  insert into traffic_signs (
    sign_type, confidence, track_id, capture_timestamp,
    image_crop_url, full_frame_url, source_sequence, geom, gps_source,
    sign_lat, sign_lon, gt_sign_id, camera_lat, camera_lon, observations
  ) values (
    p_sign_type,
    p_confidence,
    p_track_id,
    coalesce(p_capture_timestamp, now()),
    p_image_crop_url,
    p_full_frame_url,
    p_source_sequence,
    ST_SetSRID(ST_MakePoint(p_lng, p_lat), 4326)::geography,
    coalesce(p_gps_source, 'unknown'),
    p_sign_lat,
    p_sign_lon,
    p_gt_sign_id,
    p_camera_lat,
    p_camera_lon,
    coalesce(p_observations, 1)
  )
  returning id into new_id;
  return new_id;
end;
$$;

-- Bounding-box query for the map (lng/lat degrees, EPSG:4326)
create or replace function signs_in_bbox(
  min_lng double precision,
  min_lat double precision,
  max_lng double precision,
  max_lat double precision
)
returns table (
  id uuid,
  sign_type text,
  confidence real,
  track_id integer,
  capture_timestamp timestamptz,
  image_crop_url text,
  full_frame_url text,
  source_sequence text,
  gps_source text,
  lng double precision,
  lat double precision,
  sign_lat double precision,
  sign_lon double precision,
  gt_sign_id text,
  camera_lat double precision,
  camera_lon double precision,
  observations integer,
  error_m double precision,
  created_at timestamptz
)
language sql
stable
as $$
  select
    t.id,
    t.sign_type,
    t.confidence,
    t.track_id,
    t.capture_timestamp,
    t.image_crop_url,
    t.full_frame_url,
    t.source_sequence,
    t.gps_source,
    ST_X(t.geom::geometry) as lng,
    ST_Y(t.geom::geometry) as lat,
    t.sign_lat,
    t.sign_lon,
    t.gt_sign_id,
    t.camera_lat,
    t.camera_lon,
    t.observations,
    case when t.sign_lat is not null then
      ST_Distance(t.geom, ST_SetSRID(ST_MakePoint(t.sign_lon, t.sign_lat), 4326)::geography)
    end as error_m,
    t.created_at
  from traffic_signs t
  where ST_Intersects(
    t.geom,
    ST_MakeEnvelope(min_lng, min_lat, max_lng, max_lat, 4326)::geography
  );
$$;

-- Geo-localization error: predicted map pin vs ARTSv2 ground-truth sign GPS.
-- This is the headline metric for the geospatial half of the project.
create or replace view geo_error as
  select
    id,
    sign_type,
    source_sequence,
    gps_source,
    gt_sign_id,
    ST_Distance(
      geom,
      ST_SetSRID(ST_MakePoint(sign_lon, sign_lat), 4326)::geography
    ) as error_m
  from traffic_signs
  where sign_lat is not null and sign_lon is not null;

-- Storage buckets (Dashboard -> Storage):
--   1. New bucket: sign-crops  (public)
--   2. New bucket: sign-frames (public)
