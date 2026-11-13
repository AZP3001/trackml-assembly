// lidarcheck.mjs — every sensor reading the simulation produces, checked
// against a brute-force raycast over ALL of the track's walls.
//
// The sim raycasts against a per-gate wall bucket. If that bucket misses a
// wall a living car can see (it used to, wherever another pass of the track
// ran alongside), the car's lidar reads open road through a barrier. Run on
// the tracks that fold back on or cross themselves, where that showed up.
//
//   node tools/lidarcheck.mjs [track indices...]
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
const ctx = { Math, Array, JSON }; vm.createContext(ctx);
vm.runInContext(readFileSync('tracks.js', 'utf8') + '\nthis.getDefaultTracks = getDefaultTracks;', ctx);
const defs = ctx.getDefaultTracks((id, name, path, width, startPos, startAngle, zones = []) => ({ id, name, path, width, startPos, startAngle, zones }), 1200, 900);
const mod = new WebAssembly.Module(readFileSync('wasm/sim.wasm'));
const POP = 150, H = 4;
let failures = 0;
for (const ti of (process.argv.length > 2 ? process.argv.slice(2) : ['9', '10', '22']).map(Number)) {
  const def = defs[ti];
  const w = new WebAssembly.Instance(mod, {}).exports;
  const f32 = (p, n) => new Float32Array(w.memory.buffer, p, n);
  const pIn = f32(w.path_in_ptr(), def.path.length * 4);
  def.path.forEach((p, i) => { pIn[i*4]=p.x; pIn[i*4+1]=p.y; pIn[i*4+2]=p.type==='corner'?1:0; pIn[i*4+3]=p.radius!==undefined?p.radius:60; });
  w.set_config(10, 0.05, 0.02, 0.05, 750, 2, 0, H, 0); w.set_mutation(0.15, 0.5, 0.05);
  const hs = def.startPos?1:0, ha = def.startAngle!=null?1:0;
  w.track_build(w.path_in_ptr(), def.path.length, def.width, hs, hs?def.startPos.x:0, hs?def.startPos.y:0, ha, ha?def.startAngle:0, w.zone_in_ptr(), 0, 0, 0);
  const nW = w.track_wall_count();
  const wallStride = 5; // guess: x1,y1,x2,y2,seg
  const walls = new Float32Array(w.memory.buffer, w.track_walls_ptr(), nW * wallStride);
  const L = w.sensor_len();
  w.pop_init(POP, 0, H, 1234); w.pop_randomize_brains(); w.pop_reset();
  let checked = 0, bad = 0, worst = 0, ex = null, badLate = 0; const hist = new Array(10).fill(0);
  for (let gen = 0; gen < 8; gen++) {
    for (let step = 0; step < 4000; step++) {
      w.run(1); 
      if (step % 7 === 0) {
        w.write_render(); const r = f32(w.render_ptr(), POP * 18);
        for (let c = 0; c < POP; c++) {
          const o = c * 18; if (r[o+1]) continue;
          const x = r[o+2], y = r[o+3], a = r[o+4];
          for (let k = 0; k < 7; k++) {
            const ang = a + (k - 3) * Math.PI / 6, dx = Math.cos(ang) * L, dy = Math.sin(ang) * L;
            let minT = 1;
            for (let j = 0; j < nW; j++) {
              const x1 = walls[j*wallStride], y1 = walls[j*wallStride+1], ex_ = walls[j*wallStride+2]-x1, ey_ = walls[j*wallStride+3]-y1;
              const den = dx*ey_ - dy*ex_; if (Math.abs(den) < 1e-9) continue;
              const t = ((x1-x)*ey_ - (y1-y)*ex_) / den, u = ((x1-x)*dy - (y1-y)*dx) / den;
              if (t >= 0 && t <= 1 && u >= 0 && u <= 1 && t < minT) minT = t;
            }
            const want = 1 - minT, got = r[o+11+k]; checked++;
            const d = Math.abs(want - got);
            if (d > 0.02) { bad++; hist[Math.min(9,Math.floor(d*10))]++; if (step>0) badLate++; if (d > worst) { worst = d; ex = { gen, step, c, k, x, y, a, want, got, all: Array.from(r.slice(o+11,o+18)).map(v=>+v.toFixed(2)), sx: w.track_start_x(), sy: w.track_start_y() }; } }
          }
        }
      }
      if (w.all_crashed() === 1) break;
    }
    w.write_fitness(); const st = w.fitness_stride(); const fb = f32(w.fitness_ptr(), POP*st), ev = f32(w.ev_fitness_ptr(), POP);
    for (let i = 0; i < POP; i++) ev[i] = fb[i*st];
    w.evolve(15, 0, 0); w.pop_reset();
  }
  if (bad) failures++;
  console.log(`${bad ? 'FAIL' : 'ok  '}  ${def.name}: ${checked} rays, ${bad} disagree with brute force${ex ? ' — worst ' + JSON.stringify(ex) : ''}`);
}
console.log(failures ? `\n${failures} track(s) with wrong sensor readings` : '\nevery sensor reading matches a raycast over all walls.');
process.exitCode = failures ? 1 : 0;
