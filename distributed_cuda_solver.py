import os
import sys
import json
import math
import time
import hashlib
import subprocess
import numpy as np
from mpi4py import MPI
from numba import cuda, float64

# Compile-time constants for 2D shared-memory tiling (16x16 interior + 1-cell halo apron)
TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2  # 18
GLOBAL_NX = 512
GLOBAL_NY = 512
LOCAL_NX = 256
LOCAL_NY = 256
STEPS = 400
LAMBDA_DAMP = 0.15

def get_gpu_uuid() -> str:
    """Query active NVIDIA GPU UUID via nvidia-smi."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        encoding="utf-8",
    )
    return out.strip().splitlines()[0].strip()

def derive_student_seed(github_user: str, gpu_uuid: str) -> int:
    """Derive deterministic 64-bit integer seed bound to student GitHub handle and GPU UUID."""
    digest = hashlib.sha256(f"{github_user.strip()}::{gpu_uuid}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)

# ==============================================================================
# TASK 1: Distributed Cartesian Topology & Hardware Context Initialization
# ==============================================================================
def init_distributed_context(github_user: str):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Expected exactly 4 MPI ranks (2x2 grid), got {size}.")

    # Bind process to GPU 0 on Google Colab
    cuda.select_device(0)
    gpu_uuid = get_gpu_uuid() if rank == 0 else None
    gpu_uuid = comm.bcast(gpu_uuid, root=0)
    instance_seed = derive_student_seed(github_user, gpu_uuid)

    # Create non-periodic 2x2 Cartesian communicator
    cart_comm = comm.Create_cart(dims=[2, 2], periods=[False, False], reorder=False)
    coords = cart_comm.Get_coords(rank)  # (row_rank, col_rank)
    nbr_up, nbr_down = cart_comm.Shift(direction=0, disp=1)
    nbr_left, nbr_right = cart_comm.Shift(direction=1, disp=1)

    neighbors = {
        "up": nbr_up,
        "down": nbr_down,
        "left": nbr_left,
        "right": nbr_right,
    }
    return cart_comm, rank, coords, neighbors, gpu_uuid, instance_seed

# ==============================================================================
# TASK 2: Seeded Heterogeneous Subdomain Initialization
# ==============================================================================
def initialize_subdomains(coords, neighbors, instance_seed):
    row_rank, col_rank = coords
    # Global grid spacing (512 interior cells + 2 outer Dirichlet boundary layers -> 513 intervals)
    dx = 1.0 / (GLOBAL_NX + 1)
    dy = 1.0 / (GLOBAL_NY + 1)

    # Local 258x258 arrays (0 and 257 are ghost cells; 1..256 are owned interior)
    u_host = np.zeros((LOCAL_NY + 2, LOCAL_NX + 2), dtype=np.float64)
    kappa_host = np.zeros((LOCAL_NY + 2, LOCAL_NX + 2), dtype=np.float64)
    source_host = np.zeros((LOCAL_NY + 2, LOCAL_NX + 2), dtype=np.float64)

    # Deterministic pulse parameters from INSTANCE_SEED
    rng = np.random.default_rng(instance_seed)
    pulses = []
    for _ in range(4):
        px = rng.uniform(0.2, 0.8)
        py = rng.uniform(0.2, 0.8)
        amp = rng.uniform(0.6, 1.2)
        sigma = rng.uniform(0.04, 0.08)
        pulses.append((px, py, amp, sigma))

    # Global index offsets for owned interior (1..256 maps to global 1..512)
    g_row_offset = row_rank * LOCAL_NY
    g_col_offset = col_rank * LOCAL_NX

    for li in range(LOCAL_NY + 2):
        gi = g_row_offset + li  # 0..513
        y = gi * dy
        for lj in range(LOCAL_NX + 2):
            gj = g_col_offset + lj  # 0..513
            x = gj * dx

            # Heterogeneous diffusivity k(x,y) in [0.08, 0.24]
            kappa_host[li, lj] = 0.16 + 0.08 * math.sin(2.0 * math.pi * x) * math.cos(2.0 * math.pi * y)

            # Localized source term S(x,y)
            r_src2 = (x - 0.5) ** 2 + (y - 0.5) ** 2
            source_host[li, lj] = 0.25 * math.exp(-r_src2 / (2.0 * 0.07 ** 2))

            # Initial Gaussian pulses
            val = 0.0
            for px, py, amp, sigma in pulses:
                r2 = (x - px) ** 2 + (y - py) ** 2
                val += amp * math.exp(-r2 / (2.0 * sigma ** 2))
            u_host[li, lj] = val

    # Enforce homogeneous Dirichlet boundary condition (u = 0.0) on outer physical walls
    if neighbors["up"] == MPI.PROC_NULL:
        u_host[0, :] = 0.0
    if neighbors["down"] == MPI.PROC_NULL:
        u_host[LOCAL_NY + 1, :] = 0.0
    if neighbors["left"] == MPI.PROC_NULL:
        u_host[:, 0] = 0.0
    if neighbors["right"] == MPI.PROC_NULL:
        u_host[:, LOCAL_NX + 1] = 0.0

    # Safe explicit CFL time step: max(kappa) <= 0.24 -> dt <= dx^2 / (4 * 0.24)
    kappa_max_local = float(np.max(kappa_host[1:LOCAL_NY + 1, 1:LOCAL_NX + 1]))
    dt = 0.8 * (dx * dx) / (4.0 * 0.24)
    return u_host, kappa_host, source_host, dx, dy, dt, kappa_max_local

# ==============================================================================
# TASK 3: GPU Halo Pack/Unpack Kernels & Pinned MPI Exchange
# ==============================================================================
@cuda.jit
def pack_halos_kernel(u, send_up, send_down, send_left, send_right, ny, nx):
    idx = cuda.grid(1)
    if idx < nx:
        # Top interior row (i = 1) and bottom interior row (i = ny)
        send_up[idx] = u[1, idx + 1]
        send_down[idx] = u[ny, idx + 1]
    if idx < ny:
        # Left interior col (j = 1) and right interior col (j = nx)
        send_left[idx] = u[idx + 1, 1]
        send_right[idx] = u[idx + 1, nx]

@cuda.jit
def unpack_halos_kernel(u, recv_up, recv_down, recv_left, recv_right,
                        has_up, has_down, has_left, has_right, ny, nx):
    idx = cuda.grid(1)
    if idx < nx:
        u[0, idx + 1] = recv_up[idx] if has_up else 0.0
        u[ny + 1, idx + 1] = recv_down[idx] if has_down else 0.0
    if idx < ny:
        u[idx + 1, 0] = recv_left[idx] if has_left else 0.0
        u[idx + 1, nx + 1] = recv_right[idx] if has_right else 0.0

def post_halo_mpi_exchange(cart_comm, neighbors, h_send, h_recv):
    reqs = []
    # Tag scheme: 10=up->down, 11=down->up, 12=left->right, 13=right->left
    if neighbors["up"] != MPI.PROC_NULL:
        reqs.append(cart_comm.Irecv(h_recv["up"], source=neighbors["up"], tag=11))
    if neighbors["down"] != MPI.PROC_NULL:
        reqs.append(cart_comm.Irecv(h_recv["down"], source=neighbors["down"], tag=10))
    if neighbors["left"] != MPI.PROC_NULL:
        reqs.append(cart_comm.Irecv(h_recv["left"], source=neighbors["left"], tag=13))
    if neighbors["right"] != MPI.PROC_NULL:
        reqs.append(cart_comm.Irecv(h_recv["right"], source=neighbors["right"], tag=12))

    if neighbors["up"] != MPI.PROC_NULL:
        reqs.append(cart_comm.Isend(h_send["up"], dest=neighbors["up"], tag=10))
    if neighbors["down"] != MPI.PROC_NULL:
        reqs.append(cart_comm.Isend(h_send["down"], dest=neighbors["down"], tag=11))
    if neighbors["left"] != MPI.PROC_NULL:
        reqs.append(cart_comm.Isend(h_send["left"], dest=neighbors["left"], tag=12))
    if neighbors["right"] != MPI.PROC_NULL:
        reqs.append(cart_comm.Isend(h_send["right"], dest=neighbors["right"], tag=13))
    return reqs

# ==============================================================================
# TASK 4: 2D Shared-Memory Tiled Stencil & Dual-Stream Orchestrator
# ==============================================================================
@cuda.jit
def tiled_stencil_kernel(u, u_next, kappa, source, inv_dx2, inv_dy2, dt, lam,
                         r_start, r_end, c_start, c_end):
    smem = cuda.shared.array((SHARED_DIM, SHARED_DIM), dtype=float64)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    j = c_start + cuda.blockIdx.x * TILE_DIM + tx
    i = r_start + cuda.blockIdx.y * TILE_DIM + ty

    sx = tx + 1
    sy = ty + 1

    # Load center cell into shared memory
    if i <= r_end and j <= c_end:
        smem[sy, sx] = u[i, j]
    else:
        smem[sy, sx] = 0.0

    # Load 1-cell halo apron around the 16x16 block
    if ty == 0 and i <= r_end and j <= c_end:
        smem[0, sx] = u[i - 1, j]
    if ty == TILE_DIM - 1 and i <= r_end and j <= c_end:
        smem[SHARED_DIM - 1, sx] = u[i + 1, j]
    if tx == 0 and i <= r_end and j <= c_end:
        smem[sy, 0] = u[i, j - 1]
    if tx == TILE_DIM - 1 and i <= r_end and j <= c_end:
        smem[sy, SHARED_DIM - 1] = u[i, j + 1]

    # Handle out-of-bounds apron edge for partial edge blocks near r_end / c_end
    if i == r_end and ty < TILE_DIM - 1 and j <= c_end:
        smem[sy + 1, sx] = u[i + 1, j]
    if j == c_end and tx < TILE_DIM - 1 and i <= r_end:
        smem[sy, sx + 1] = u[i, j + 1]

    cuda.syncthreads()

    if i <= r_end and j <= c_end:
        u_c = smem[sy, sx]
        u_n = smem[sy - 1, sx]
        u_s = smem[sy + 1, sx]
        u_w = smem[sy, sx - 1]
        u_e = smem[sy, sx + 1]

        k_c = kappa[i, j]
        k_n = 0.5 * (kappa[i - 1, j] + k_c)
        k_s = 0.5 * (kappa[i + 1, j] + k_c)
        k_w = 0.5 * (kappa[i, j - 1] + k_c)
        k_e = 0.5 * (kappa[i, j + 1] + k_c)

        flux_y = (k_s * (u_s - u_c) - k_n * (u_c - u_n)) * inv_dy2
        flux_x = (k_e * (u_e - u_c) - k_w * (u_c - u_w)) * inv_dx2
        reaction = -lam * (u_c * u_c * u_c) + source[i, j]

        u_next[i, j] = u_c + dt * (flux_x + flux_y + reaction)

def launch_stencil_box(d_u, d_u_next, d_kappa, d_source, inv_dx2, inv_dy2, dt,
                       r_start, r_end, c_start, c_end, stream):
    rows = r_end - r_start + 1
    cols = c_end - c_start + 1
    if rows <= 0 or cols <= 0:
        return
    grid = ((cols + TILE_DIM - 1) // TILE_DIM, (rows + TILE_DIM - 1) // TILE_DIM)
    block = (TILE_DIM, TILE_DIM)
    tiled_stencil_kernel[grid, block, stream](
        d_u, d_u_next, d_kappa, d_source, inv_dx2, inv_dy2, dt, LAMBDA_DAMP,
        r_start, r_end, c_start, c_end
    )

def step_overlapped(cart_comm, neighbors, buffers, streams, params):
    d_u, d_u_next, d_kappa, d_source, d_send, d_recv, h_send, h_recv = buffers
    stream_interior, stream_boundary = streams
    inv_dx2, inv_dy2, dt = params

    has_up = neighbors["up"] != MPI.PROC_NULL
    has_down = neighbors["down"] != MPI.PROC_NULL
    has_left = neighbors["left"] != MPI.PROC_NULL
    has_right = neighbors["right"] != MPI.PROC_NULL

    # Stage A: Pack halos & Async D2H on stream_boundary
    threads_1d = 128
    blocks_1d = (max(LOCAL_NX, LOCAL_NY) + threads_1d - 1) // threads_1d
    pack_halos_kernel[blocks_1d, threads_1d, stream_boundary](
        d_u, d_send["up"], d_send["down"], d_send["left"], d_send["right"], LOCAL_NY, LOCAL_NX
    )
    for k in ("up", "down", "left", "right"):
        d_send[k].copy_to_host(h_send[k], stream=stream_boundary)

    # Stage B: Sync boundary D2H, post non-blocking MPI, compute deep interior [2..255, 2..255]
    stream_boundary.synchronize()
    reqs = post_halo_mpi_exchange(cart_comm, neighbors, h_send, h_recv)

    launch_stencil_box(
        d_u, d_u_next, d_kappa, d_source, inv_dx2, inv_dy2, dt,
        2, LOCAL_NY - 1, 2, LOCAL_NX - 1, stream_interior
    )

    # Stage C: Waitall MPI, Async H2D, unpack halos, compute 4 outer perimeter rings on stream_boundary
    MPI.Request.Waitall(reqs)
    for k in ("up", "down", "left", "right"):
        d_recv[k].copy_to_device(h_recv[k], stream=stream_boundary)

    unpack_halos_kernel[blocks_1d, threads_1d, stream_boundary](
        d_u, d_recv["up"], d_recv["down"], d_recv["left"], d_recv["right"],
        has_up, has_down, has_left, has_right, LOCAL_NY, LOCAL_NX
    )

    # Top ring (row 1), Bottom ring (row 256), Left ring (rows 2..255, col 1), Right ring (rows 2..255, col 256)
    launch_stencil_box(d_u, d_u_next, d_kappa, d_source, inv_dx2, inv_dy2, dt, 1, 1, 1, LOCAL_NX, stream_boundary)
    launch_stencil_box(d_u, d_u_next, d_kappa, d_source, inv_dx2, inv_dy2, dt, LOCAL_NY, LOCAL_NY, 1, LOCAL_NX, stream_boundary)
    launch_stencil_box(d_u, d_u_next, d_kappa, d_source, inv_dx2, inv_dy2, dt, 2, LOCAL_NY - 1, 1, 1, stream_boundary)
    launch_stencil_box(d_u, d_u_next, d_kappa, d_source, inv_dx2, inv_dy2, dt, 2, LOCAL_NY - 1, LOCAL_NX, LOCAL_NX, stream_boundary)

    stream_interior.synchronize()
    stream_boundary.synchronize()

def step_serialized(cart_comm, neighbors, buffers, stream_default, params):
    d_u, d_u_next, d_kappa, d_source, d_send, d_recv, h_send, h_recv = buffers
    inv_dx2, inv_dy2, dt = params

    has_up = neighbors["up"] != MPI.PROC_NULL
    has_down = neighbors["down"] != MPI.PROC_NULL
    has_left = neighbors["left"] != MPI.PROC_NULL
    has_right = neighbors["right"] != MPI.PROC_NULL

    threads_1d = 128
    blocks_1d = (max(LOCAL_NX, LOCAL_NY) + threads_1d - 1) // threads_1d
    pack_halos_kernel[blocks_1d, threads_1d, stream_default](
        d_u, d_send["up"], d_send["down"], d_send["left"], d_send["right"], LOCAL_NY, LOCAL_NX
    )
    for k in ("up", "down", "left", "right"):
        d_send[k].copy_to_host(h_send[k], stream=stream_default)
    stream_default.synchronize()

    reqs = post_halo_mpi_exchange(cart_comm, neighbors, h_send, h_recv)
    MPI.Request.Waitall(reqs)

    for k in ("up", "down", "left", "right"):
        d_recv[k].copy_to_device(h_recv[k], stream=stream_default)
    unpack_halos_kernel[blocks_1d, threads_1d, stream_default](
        d_u, d_recv["up"], d_recv["down"], d_recv["left"], d_recv["right"],
        has_up, has_down, has_left, has_right, LOCAL_NY, LOCAL_NX
    )

    # Full subdomain [1..256, 1..256] after halo exchange completes
    launch_stencil_box(
        d_u, d_u_next, d_kappa, d_source, inv_dx2, inv_dy2, dt,
        1, LOCAL_NY, 1, LOCAL_NX, stream_default
    )
    stream_default.synchronize()

# ==============================================================================
# TASK 5: Distributed Reductions, PTX Audit & Comparative Benchmark
# ==============================================================================
def compute_global_diagnostics(cart_comm, d_u, d_u_prev, dx, dy, dt):
    u_curr = d_u.copy_to_host()[1:LOCAL_NY + 1, 1:LOCAL_NX + 1]
    u_old = d_u_prev.copy_to_host()[1:LOCAL_NY + 1, 1:LOCAL_NX + 1]

    local_energy = float(np.sum(u_curr) * dx * dy)
    diff = (u_curr - u_old) / dt
    local_res_sq = float(np.sum(diff * diff) * dx * dy)

    global_energy = cart_comm.allreduce(local_energy, op=MPI.SUM)
    global_res_sq = cart_comm.allreduce(local_res_sq, op=MPI.SUM)
    return global_energy, math.sqrt(global_res_sq)

def run_solver(github_user: str):
    cart_comm, rank, coords, neighbors, gpu_uuid, instance_seed = init_distributed_context(github_user)
    u_init, kappa_host, source_host, dx, dy, dt, kappa_max_local = initialize_subdomains(
        coords, neighbors, instance_seed
    )

    kappa_max_global = cart_comm.allreduce(kappa_max_local, op=MPI.MAX)
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)
    params = (inv_dx2, inv_dy2, dt)

    # Allocate device & pinned host staging buffers
    d_kappa = cuda.to_device(kappa_host)
    d_source = cuda.to_device(source_host)

    def make_buffers():
        d_u = cuda.to_device(u_init.copy())
        d_u_next = cuda.to_device(u_init.copy())
        d_send = {k: cuda.device_array(LOCAL_NX, dtype=np.float64) for k in ("up", "down", "left", "right")}
        d_recv = {k: cuda.device_array(LOCAL_NX, dtype=np.float64) for k in ("up", "down", "left", "right")}
        h_send = {k: cuda.pinned_array(LOCAL_NX, dtype=np.float64) for k in ("up", "down", "left", "right")}
        h_recv = {k: cuda.pinned_array(LOCAL_NX, dtype=np.float64) for k in ("up", "down", "left", "right")}
        return [d_u, d_u_next, d_kappa, d_source, d_send, d_recv, h_send, h_recv]

    stream_interior = cuda.stream()
    stream_boundary = cuda.stream()

    # Warmup JIT compilation
    warm_bufs = make_buffers()
    step_overlapped(cart_comm, neighbors, warm_bufs, (stream_interior, stream_boundary), params)

    # --- Mode 1: Serialized Execution ---
    buf_ser = make_buffers()
    cart_comm.Barrier()
    ev_ser_start = cuda.event()
    ev_ser_end = cuda.event()
    ev_ser_start.record(stream_boundary)
    t0_ser = time.perf_counter()

    for s in range(1, STEPS + 1):
        step_serialized(cart_comm, neighbors, buf_ser, stream_boundary, params)
        buf_ser[0], buf_ser[1] = buf_ser[1], buf_ser[0]

    ev_ser_end.record(stream_boundary)
    ev_ser_end.synchronize()
    cart_comm.Barrier()
    ser_wall_ms = (time.perf_counter() - t0_ser) * 1000.0
    ser_cuda_ms = cuda.event_elapsed_time(ev_ser_start, ev_ser_end)

    # --- Mode 2: Dual-Stream Overlapped Execution + Energy Trace ---
    buf_ovl = make_buffers()
    energy_trace = {}

    e0, r0 = compute_global_diagnostics(cart_comm, buf_ovl[0], buf_ovl[0], dx, dy, dt)
    energy_trace[0] = {"energy": e0, "l2_residual": r0}

    cart_comm.Barrier()
    ev_ovl_start = cuda.event()
    ev_ovl_end = cuda.event()
    ev_ovl_start.record(stream_boundary)
    t0_ovl = time.perf_counter()

    for s in range(1, STEPS + 1):
        step_overlapped(cart_comm, neighbors, buf_ovl, (stream_interior, stream_boundary), params)
        buf_ovl[0], buf_ovl[1] = buf_ovl[1], buf_ovl[0]
        if s % 100 == 0:
            en, res = compute_global_diagnostics(cart_comm, buf_ovl[0], buf_ovl[1], dx, dy, dt)
            energy_trace[s] = {"energy": en, "l2_residual": res}

    ev_ovl_end.record(stream_boundary)
    ev_ovl_end.synchronize()
    cart_comm.Barrier()
    ovl_wall_ms = (time.perf_counter() - t0_ovl) * 1000.0
    ovl_cuda_ms = cuda.event_elapsed_time(ev_ovl_start, ev_ovl_end)

    # Parity check between serialized and overlapped final states
    u_ser_final = buf_ser[0].copy_to_host()[1:LOCAL_NY + 1, 1:LOCAL_NX + 1]
    u_ovl_final = buf_ovl[0].copy_to_host()[1:LOCAL_NY + 1, 1:LOCAL_NX + 1]
    local_max_diff = float(np.max(np.abs(u_ser_final - u_ovl_final)))
    parity_max_diff = cart_comm.allreduce(local_max_diff, op=MPI.MAX)

    # Aggregate timings across ranks
    max_ser_wall_ms = cart_comm.allreduce(ser_wall_ms, op=MPI.MAX)
    max_ovl_wall_ms = cart_comm.allreduce(ovl_wall_ms, op=MPI.MAX)
    max_ser_cuda_ms = cart_comm.allreduce(ser_cuda_ms, op=MPI.MAX)
    max_ovl_cuda_ms = cart_comm.allreduce(ovl_cuda_ms, op=MPI.MAX)

    if rank == 0:
        ptx_dict = tiled_stencil_kernel.inspect_asm()
        ptx_code = next(iter(ptx_dict.values()))
        st_shared_count = ptx_code.count("st.shared.f64")
        ld_shared_count = ptx_code.count("ld.shared.f64")
        bar_sync_count = ptx_code.count("bar.sync")
        ptx_audit_passed = (st_shared_count > 0) and (ld_shared_count > 0) and (bar_sync_count > 0)

        dt_max_cfl = (dx * dx) / (4.0 * kappa_max_global)

        print("=== DISTRIBUTED MPI + CUDA SOLVER TELEMETRY ===")
        print(f"GitHub User      : {github_user}")
        print(f"GPU UUID         : {gpu_uuid}")
        print(f"INSTANCE_SEED    : {instance_seed}")
        print(f"MPI Topology     : 4 ranks (2x2 non-periodic Cartesian)")
        print(f"dx = dy          : {dx:.8e}")
        print(f"kappa_max        : {kappa_max_global:.8f}")
        print(f"dt_used          : {dt:.8e}")
        print(f"dt_max_cfl       : {dt_max_cfl:.8e}")
        print(f"PTX Counts       : st.shared.f64={st_shared_count}, ld.shared.f64={ld_shared_count}, bar.sync={bar_sync_count}")
        print(f"PTX Audit Passed : {ptx_audit_passed}")
        print(f"Serialized Wall  : {max_ser_wall_ms:.3f} ms (CUDA Event: {max_ser_cuda_ms:.3f} ms)")
        print(f"Overlapped Wall  : {max_ovl_wall_ms:.3f} ms (CUDA Event: {max_ovl_cuda_ms:.3f} ms)")
        print(f"Parity Max Diff  : {parity_max_diff:.3e}")
        print("\n--- Energy & L2 Residual Trace ---")
        for step_k, vals in energy_trace.items():
            print(f"Step {step_k:3d} | Global Energy E = {vals['energy']:.10f} | L2 Residual = {vals['l2_residual']:.6e}")

        manifest = {
            "github_username": github_user,
            "gpu_uuid": gpu_uuid,
            "instance_seed": instance_seed,
            "mpi_ranks": 4,
            "grid_dims": [2, 2],
            "dx": dx,
            "kappa_max": kappa_max_global,
            "dt": dt,
            "dt_max_cfl": dt_max_cfl,
            "ptx_counts": {
                "st.shared.f64": st_shared_count,
                "ld.shared.f64": ld_shared_count,
                "bar.sync": bar_sync_count,
            },
            "ptx_audit_passed": ptx_audit_passed,
            "timings_ms": {
                "serialized_ms": round(max_ser_wall_ms, 3),
                "overlapped_ms": round(max_ovl_wall_ms, 3),
                "serialized_cuda_event_ms": round(max_ser_cuda_ms, 3),
                "overlapped_cuda_event_ms": round(max_ovl_cuda_ms, 3),
            },
            "parity_max_diff": parity_max_diff,
            "energy_trace": {str(k): v for k, v in energy_trace.items()},
        }
        with open("submission_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print("\nSaved submission_manifest.json successfully.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Usage: python3 distributed_cuda_solver.py <GITHUB_USERNAME>")
    run_solver(sys.argv[1])
