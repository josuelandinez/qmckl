#include "qmckl.h"

// Fix C/C++ interoperability for the C99 restrict keyword
#define restrict
extern "C" {
#include "qmckl_context_private_type.h"
}
#undef restrict

#include <Kokkos_Core.hpp>
#include <KokkosBlas3_gemm.hpp>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <atomic>
#include <type_traits>
#include <mutex>
#include <algorithm>

#ifdef HAVE_KTUNE
#include <KTune/KTune.hpp>
#define KOKKOS_OR_KTUNE_PARALLEL_FOR KTune::parallel_for
#else
#define KOKKOS_OR_KTUNE_PARALLEL_FOR Kokkos::parallel_for
#endif

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
using DeviceLayout = Kokkos::LayoutLeft;
#else
using DeviceLayout = Kokkos::LayoutRight;
#endif

#if defined(KOKKOS_ENABLE_CUDA)
using ExecSpace   = Kokkos::Cuda;
using DeviceSpace = Kokkos::CudaSpace;
#elif defined(KOKKOS_ENABLE_HIP)
using ExecSpace   = Kokkos::HIP;
using DeviceSpace = Kokkos::HIPSpace;
#elif defined(KOKKOS_ENABLE_SYCL)
using ExecSpace   = Kokkos::Experimental::SYCL;
using DeviceSpace = Kokkos::Experimental::SYCLDeviceUSMSpace;
#else
using ExecSpace   = Kokkos::DefaultExecutionSpace;
using DeviceSpace = Kokkos::DefaultExecutionSpace::memory_space;
#endif

using HostExecSpace = Kokkos::DefaultHostExecutionSpace;
using HostSpace     = Kokkos::DefaultHostExecutionSpace::memory_space;

// Compile-time check to determine if the execution space has direct access to host memory
constexpr bool is_host_accessible = Kokkos::SpaceAccessibility<ExecSpace, HostSpace>::accessible;

// -------------------------------------------------------------------------------------------------
// Device state
//
// AO_VGL / MO_VGL are represented internally as a single "stacked" 2D matrix of shape
// (5*point_num, X) rather than as five independent (point_num, X) arrays or a single
// 3D (point_num, 5, X) array. Row block c*point_num .. (c+1)*point_num-1 holds component
// c (value, dX, dY, dZ, Laplacian). This is mathematically exact (GEMM acts row-wise,
// so stacking rows of A and re-using the same B for all of them is identical to running
// the GEMM once per row-block) and it lets the whole VGL transform for MOs be computed
// with a SINGLE GEMM call instead of five, on both CPU and GPU.
// -------------------------------------------------------------------------------------------------

struct KokkosDeviceState {
    bool basis_initialized = false;
    bool mo_initialized = false;
    int32_t lmax_global = 0;

    Kokkos::View<double*, DeviceSpace> d_coord;
    Kokkos::View<double*, DeviceSpace> d_nucl_coord;
    Kokkos::View<int64_t*, DeviceSpace> d_nucleus_index;
    Kokkos::View<int64_t*, DeviceSpace> d_nucleus_shell_num;
    Kokkos::View<double*, DeviceSpace> d_nucleus_range;
    Kokkos::View<int32_t*, DeviceSpace> d_shell_ang_mom;
    Kokkos::View<int64_t*, DeviceSpace> d_shell_prim_idx;
    Kokkos::View<int64_t*, DeviceSpace> d_shell_prim_num;
    Kokkos::View<int64_t*, DeviceSpace> d_shell_to_nucl;
    Kokkos::View<double*, DeviceSpace> d_exponent;
    Kokkos::View<double*, DeviceSpace> d_coef_normalized;
    Kokkos::View<double*, DeviceSpace> d_ao_factor;
    Kokkos::View<int64_t*, DeviceSpace> d_ao_index;

    // Plain (non-VGL) AO/MO value buffers - unaffected by this change, still single 2D matrices.
    Kokkos::View<double**, DeviceLayout, DeviceSpace> d_ao_value;
    Kokkos::View<double**, DeviceLayout, DeviceSpace> d_mo_value;

    // Stacked VGL buffers: (5*point_num, ao_num) and (5*point_num, mo_num).
    Kokkos::View<double**, DeviceLayout, DeviceSpace> d_ao_vgl_stacked;
    Kokkos::View<double**, DeviceLayout, DeviceSpace> d_mo_vgl_stacked;

    Kokkos::View<double**, DeviceLayout, DeviceSpace> d_mo_coef;

    Kokkos::View<double**, Kokkos::LayoutRight, HostSpace> h_mo_coef_raw;
    Kokkos::View<double**, DeviceLayout, HostSpace> h_mo_coef_mirror;
    Kokkos::View<double**, DeviceLayout, HostSpace> h_ao_val_mirror;
    Kokkos::View<double**, DeviceLayout, HostSpace> h_mo_val_mirror;

    // Host-side staging mirrors for the stacked VGL buffers, cached across calls so
    // repeated Monte Carlo steps do not pay reallocation cost every time.
    Kokkos::View<double**, DeviceLayout, HostSpace> h_ao_vgl_stacked_mirror;
    Kokkos::View<double**, DeviceLayout, HostSpace> h_mo_vgl_stacked_mirror;

    // Scratch for the CPU sparse mo_value/mo_vgl kernels' compaction pass (see
    // qmckl_compute_mo_basis_mo_value_kokkos / _mo_vgl_kokkos). Cached here and resized
    // only when point_num/ao_num change, exactly like the mirrors above: allocating
    // these fresh on every one of the ~10 calls per Monte Carlo step (as an earlier
    // version did, using function-local Views) measurably regressed performance -
    // first-touch page faults on freshly (re)allocated memory landed inside the
    // profiled kernel time on every single call. Caching them here means the
    // allocation (and its first-touch cost) is paid once, not once per call. Contents
    // are pure scratch (fully overwritten before being read on every call), so
    // qmckl_kokkos_state_copy() intentionally does not duplicate them - a copied
    // context simply allocates its own on first use, exactly like a fresh context.
    Kokkos::View<int64_t**, Kokkos::LayoutRight, HostSpace> h_mo_value_idx_scratch;
    Kokkos::View<double**, Kokkos::LayoutRight, HostSpace> h_mo_value_av_scratch;
    Kokkos::View<int64_t**, Kokkos::LayoutRight, HostSpace> h_mo_vgl_idx_scratch;
    Kokkos::View<double**, Kokkos::LayoutRight, HostSpace> h_mo_vgl_av1_scratch;
    Kokkos::View<double**, Kokkos::LayoutRight, HostSpace> h_mo_vgl_av2_scratch;
    Kokkos::View<double**, Kokkos::LayoutRight, HostSpace> h_mo_vgl_av3_scratch;
    Kokkos::View<double**, Kokkos::LayoutRight, HostSpace> h_mo_vgl_av4_scratch;
    Kokkos::View<double**, Kokkos::LayoutRight, HostSpace> h_mo_vgl_av5_scratch;

    void reset() {
        basis_initialized = false;
        mo_initialized = false;
        lmax_global = 0;

        d_coord = Kokkos::View<double*, DeviceSpace>();
        d_nucl_coord = Kokkos::View<double*, DeviceSpace>();
        d_nucleus_index = Kokkos::View<int64_t*, DeviceSpace>();
        d_nucleus_shell_num = Kokkos::View<int64_t*, DeviceSpace>();
        d_nucleus_range = Kokkos::View<double*, DeviceSpace>();
        d_shell_ang_mom = Kokkos::View<int32_t*, DeviceSpace>();
        d_shell_prim_idx = Kokkos::View<int64_t*, DeviceSpace>();
        d_shell_prim_num = Kokkos::View<int64_t*, DeviceSpace>();
        d_shell_to_nucl = Kokkos::View<int64_t*, DeviceSpace>();
        d_exponent = Kokkos::View<double*, DeviceSpace>();
        d_coef_normalized = Kokkos::View<double*, DeviceSpace>();
        d_ao_factor = Kokkos::View<double*, DeviceSpace>();
        d_ao_index = Kokkos::View<int64_t*, DeviceSpace>();

        d_ao_value = Kokkos::View<double**, DeviceLayout, DeviceSpace>();
        d_mo_value = Kokkos::View<double**, DeviceLayout, DeviceSpace>();

        d_ao_vgl_stacked = Kokkos::View<double**, DeviceLayout, DeviceSpace>();
        d_mo_vgl_stacked = Kokkos::View<double**, DeviceLayout, DeviceSpace>();

        d_mo_coef = Kokkos::View<double**, DeviceLayout, DeviceSpace>();

        h_mo_coef_raw = Kokkos::View<double**, Kokkos::LayoutRight, HostSpace>();
        h_mo_coef_mirror = Kokkos::View<double**, DeviceLayout, HostSpace>();
        h_ao_val_mirror = Kokkos::View<double**, DeviceLayout, HostSpace>();
        h_mo_val_mirror = Kokkos::View<double**, DeviceLayout, HostSpace>();

        h_ao_vgl_stacked_mirror = Kokkos::View<double**, DeviceLayout, HostSpace>();
        h_mo_vgl_stacked_mirror = Kokkos::View<double**, DeviceLayout, HostSpace>();

        h_mo_value_idx_scratch = Kokkos::View<int64_t**, Kokkos::LayoutRight, HostSpace>();
        h_mo_value_av_scratch = Kokkos::View<double**, Kokkos::LayoutRight, HostSpace>();
        h_mo_vgl_idx_scratch = Kokkos::View<int64_t**, Kokkos::LayoutRight, HostSpace>();
        h_mo_vgl_av1_scratch = Kokkos::View<double**, Kokkos::LayoutRight, HostSpace>();
        h_mo_vgl_av2_scratch = Kokkos::View<double**, Kokkos::LayoutRight, HostSpace>();
        h_mo_vgl_av3_scratch = Kokkos::View<double**, Kokkos::LayoutRight, HostSpace>();
        h_mo_vgl_av4_scratch = Kokkos::View<double**, Kokkos::LayoutRight, HostSpace>();
        h_mo_vgl_av5_scratch = Kokkos::View<double**, Kokkos::LayoutRight, HostSpace>();
    }
};

static std::vector<KokkosDeviceState*> g_active_device_states;
static std::mutex g_state_mutex;

static void register_state(KokkosDeviceState* s) {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    g_active_device_states.push_back(s);
}

static void unregister_state(KokkosDeviceState* s) {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    auto it = std::find(g_active_device_states.begin(), g_active_device_states.end(), s);
    if (it != g_active_device_states.end()) {
        g_active_device_states.erase(it);
    }
}

static void reset_all_states() {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    for (auto* s : g_active_device_states) {
        if (s) s->reset();
    }
}

static inline bool should_bypass_ao_sync() {
    static const char* sync_env = std::getenv("QMCKL_KOKKOS_SYNC_AO");
    if (sync_env != nullptr) {
        if (std::strcmp(sync_env, "1") == 0 || std::strcmp(sync_env, "ON") == 0 || std::strcmp(sync_env, "true") == 0) return false;
        if (std::strcmp(sync_env, "0") == 0 || std::strcmp(sync_env, "OFF") == 0 || std::strcmp(sync_env, "false") == 0) return true;
    }
    static const char* bypass_env = std::getenv("QMCKL_KOKKOS_BYPASS_AO_SYNC");
    if (bypass_env != nullptr) return !(std::strcmp(bypass_env, "0") == 0 || std::strcmp(bypass_env, "OFF") == 0 || std::strcmp(bypass_env, "false") == 0);
    return false;
}

// -------------------------------------------------------------------------------------------------
// 1D Kernels (Preserved strictly for OpenMP host-side L1 cache localization)
// -------------------------------------------------------------------------------------------------

template <int LMAX, typename CoordViewType, typename ViewType>
void kokkos_ao_gaussian_kernel_coalesced(
    const int64_t point_num, const int64_t ao_num, const int64_t shell_num, const int64_t nucl_num,
    const KokkosDeviceState& state, CoordViewType d_coord, ViewType d_ao_value)
{
    auto d_nucl_coord      = state.d_nucl_coord;
    auto d_nucleus_range   = state.d_nucleus_range;
    auto d_shell_ang_mom   = state.d_shell_ang_mom;
    auto d_shell_prim_idx  = state.d_shell_prim_idx;
    auto d_shell_prim_num  = state.d_shell_prim_num;
    auto d_shell_to_nucl   = state.d_shell_to_nucl;
    auto d_exponent        = state.d_exponent;
    auto d_coef_normalized = state.d_coef_normalized;
    auto d_ao_factor       = state.d_ao_factor;
    auto d_ao_index        = state.d_ao_index;

    Kokkos::parallel_for("AO_Gaussian_Value_1D_Coalesced",
        Kokkos::RangePolicy<ExecSpace>(0, point_num),
        KOKKOS_LAMBDA(const int64_t ipoint) {
            constexpr double cutoff = 36.043653389117154;
            constexpr double shift = 1.e-20;

            const double e_x = d_coord(ipoint);
            const double e_y = d_coord(ipoint + point_num);
            const double e_z = d_coord(ipoint + 2 * point_num);

            for (int64_t ishell = 0; ishell < shell_num; ++ishell) {
                const int32_t l = d_shell_ang_mom(ishell);
                if (l > LMAX) continue;

                const int64_t inucl = d_shell_to_nucl(ishell);
                const double x = e_x - d_nucl_coord(inucl);
                const double y = e_y - d_nucl_coord(inucl + nucl_num);
                const double z = e_z - d_nucl_coord(inucl + 2 * nucl_num);
                const double r2 = x*x + y*y + z*z;

                if (r2 > cutoff * d_nucleus_range(inucl)) {
                    int64_t ao_offset = d_ao_index(ishell);
                    int deg = (l == 0) ? 1 : (l == 1) ? 3 : (l == 2) ? 6 : (l == 3) ? 10 : (l == 4) ? 15 : 21;
                    for (int k = 0; k < deg && ao_offset + k < ao_num; ++k) {
                        d_ao_value(ipoint, ao_offset + k) = 0.0;
                    }
                    continue;
                }

                double radial_sum = 0.0;
                const int64_t iprim_start = d_shell_prim_idx(ishell);
                const int64_t prim_count = d_shell_prim_num(ishell);

                for (int64_t p = 0; p < prim_count; ++p) {
                    const int64_t iprim = iprim_start + p;
                    const double alpha = d_exponent(iprim);
                    const double ar2 = alpha * r2;
                    if (ar2 <= cutoff) {
                        radial_sum += d_coef_normalized(iprim) * exp(-ar2);
                    }
                }

                const int64_t ao_offset = d_ao_index(ishell);

                if (radial_sum == 0.0) {
                    int deg = (l == 0) ? 1 : (l == 1) ? 3 : (l == 2) ? 6 : (l == 3) ? 10 : (l == 4) ? 15 : 21;
                    for (int k = 0; k < deg && ao_offset + k < ao_num; ++k) {
                        d_ao_value(ipoint, ao_offset + k) = 0.0;
                    }
                    continue;
                }

                double pows_x[6], pows_y[6], pows_z[6];
                pows_x[0] = 1.0; pows_y[0] = 1.0; pows_z[0] = 1.0;
                for (int i = 1; i <= LMAX; ++i) {
                    pows_x[i] = pows_x[i-1] * x;
                    pows_y[i] = pows_y[i-1] * y;
                    pows_z[i] = pows_z[i-1] * z;
                }

                int il = 0;
                if (l == 0) {
                    if (ao_offset < ao_num) d_ao_value(ipoint, ao_offset) = radial_sum * d_ao_factor(ao_offset);
                } else if (l == 1) {
                    const double vals[3] = {x + shift, y + shift, z + shift};
                    for (int i = 0; i < 3; ++i) {
                        if (ao_offset + il < ao_num) d_ao_value(ipoint, ao_offset + il) = vals[i] * radial_sum * d_ao_factor(ao_offset + il);
                        il++;
                    }
                } else if (l == 2) {
                    const double vals[6] = {x*x + shift, x*y + shift, x*z + shift, y*y + shift, y*z + shift, z*z + shift};
                    for (int i = 0; i < 6; ++i) {
                        if (ao_offset + il < ao_num) d_ao_value(ipoint, ao_offset + il) = vals[i] * radial_sum * d_ao_factor(ao_offset + il);
                        il++;
                    }
                } else {
                    for (int a = l; a >= 0; --a) {
                        for (int b = l - a; b >= 0; --b) {
                            const int c = l - a - b;
                            const double val = pows_x[a] * pows_y[b] * pows_z[c] + shift;
                            if (ao_offset + il < ao_num) d_ao_value(ipoint, ao_offset + il) = val * radial_sum * d_ao_factor(ao_offset + il);
                            il++;
                        }
                    }
                }
            }
    });
}

template <int LMAX, typename CoordViewType, typename ViewType2D>
void kokkos_ao_vgl_gaussian_kernel_coalesced(
    const int64_t point_num, const int64_t ao_num, const int64_t shell_num, const int64_t nucl_num,
    const KokkosDeviceState& state, CoordViewType d_coord,
    ViewType2D v_ao_c0, ViewType2D v_ao_c1, ViewType2D v_ao_c2, ViewType2D v_ao_c3, ViewType2D v_ao_c4)
{
    auto d_nucl_coord      = state.d_nucl_coord;
    auto d_nucleus_range   = state.d_nucleus_range;
    auto d_shell_ang_mom   = state.d_shell_ang_mom;
    auto d_shell_prim_idx  = state.d_shell_prim_idx;
    auto d_shell_prim_num  = state.d_shell_prim_num;
    auto d_shell_to_nucl   = state.d_shell_to_nucl;
    auto d_exponent        = state.d_exponent;
    auto d_coef_normalized = state.d_coef_normalized;
    auto d_ao_factor       = state.d_ao_factor;
    auto d_ao_index        = state.d_ao_index;

    Kokkos::parallel_for("AO_Gaussian_VGL_1D_Coalesced",
        Kokkos::RangePolicy<ExecSpace>(0, point_num),
        KOKKOS_LAMBDA(const int64_t ipoint) {
            constexpr double cutoff = 36.043653389117154;
            constexpr double shift = 1.e-20;

            const double e_x = d_coord(ipoint);
            const double e_y = d_coord(ipoint + point_num);
            const double e_z = d_coord(ipoint + 2 * point_num);

            for (int64_t ishell = 0; ishell < shell_num; ++ishell) {
                const int32_t l = d_shell_ang_mom(ishell);
                if (l > LMAX) continue;

                const int64_t inucl = d_shell_to_nucl(ishell);
                const double x = e_x - d_nucl_coord(inucl);
                const double y = e_y - d_nucl_coord(inucl + nucl_num);
                const double z = e_z - d_nucl_coord(inucl + 2 * nucl_num);
                const double r2 = x*x + y*y + z*z;

                const int64_t ao_offset = d_ao_index(ishell);
                const int deg = (l == 0) ? 1 : (l == 1) ? 3 : (l == 2) ? 6 : (l == 3) ? 10 : (l == 4) ? 15 : 21;

                if (r2 > cutoff * d_nucleus_range(inucl)) {
                    for (int k = 0; k < deg && ao_offset + k < ao_num; ++k) {
                        v_ao_c0(ipoint, ao_offset + k) = 0.0;
                        v_ao_c1(ipoint, ao_offset + k) = 0.0;
                        v_ao_c2(ipoint, ao_offset + k) = 0.0;
                        v_ao_c3(ipoint, ao_offset + k) = 0.0;
                        v_ao_c4(ipoint, ao_offset + k) = 0.0;
                    }
                    continue;
                }

                double s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0, s5 = 0.0;
                const int64_t iprim_start = d_shell_prim_idx(ishell);
                const int64_t prim_count = d_shell_prim_num(ishell);

                for (int64_t p = 0; p < prim_count; ++p) {
                    const int64_t iprim = iprim_start + p;
                    const double alpha = d_exponent(iprim);
                    const double ar2_val = alpha * r2;

                    if (ar2_val <= cutoff) {
                        const double exp_val = d_coef_normalized(iprim) * exp(-ar2_val);
                        const double f = -2.0 * alpha * exp_val;

                        s1 += exp_val;
                        s2 += f * x; s3 += f * y; s4 += f * z;
                        s5 += f * (3.0 - 2.0 * ar2_val);
                    }
                }

                if (s1 == 0.0) {
                    for (int k = 0; k < deg && ao_offset + k < ao_num; ++k) {
                        v_ao_c0(ipoint, ao_offset + k) = 0.0;
                        v_ao_c1(ipoint, ao_offset + k) = 0.0;
                        v_ao_c2(ipoint, ao_offset + k) = 0.0;
                        v_ao_c3(ipoint, ao_offset + k) = 0.0;
                        v_ao_c4(ipoint, ao_offset + k) = 0.0;
                    }
                    continue;
                }

                double px[8], py[8], pz[8];
                px[0]=1.0; px[1]=1.0; px[2]=1.0;
                py[0]=1.0; py[1]=1.0; py[2]=1.0;
                pz[0]=1.0; pz[1]=1.0; pz[2]=1.0;

                for (int i = 1; i <= LMAX; ++i) {
                    px[i+2] = px[i+1] * x;
                    py[i+2] = py[i+1] * y;
                    pz[i+2] = pz[i+1] * z;
                }

                int il = 0;
                if (l == 0) {
                    const double p1 = 1.0, p2 = 0.0, p3 = 0.0, p4 = 0.0, p5 = 0.0;
                    if (ao_offset < ao_num) {
                        const double fact = d_ao_factor(ao_offset);
                        v_ao_c0(ipoint, ao_offset) = p1 * s1 * fact;
                        v_ao_c1(ipoint, ao_offset) = (p2 * s1 + p1 * s2) * fact;
                        v_ao_c2(ipoint, ao_offset) = (p3 * s1 + p1 * s3) * fact;
                        v_ao_c3(ipoint, ao_offset) = (p4 * s1 + p1 * s4) * fact;
                        v_ao_c4(ipoint, ao_offset) = (p5 * s1 + p1 * s5 + 2.0 * (p2 * s2 + p3 * s3 + p4 * s4)) * fact;
                    }
                } else if (l == 1) {
                    const double p1[3] = {x+shift, y+shift, z+shift};
                    const double p2[3] = {1.0, 0.0, 0.0};
                    const double p3[3] = {0.0, 1.0, 0.0};
                    const double p4[3] = {0.0, 0.0, 1.0};
                    const double p5[3] = {0.0, 0.0, 0.0};
                    for (int i = 0; i < 3; ++i) {
                        if (ao_offset + il < ao_num) {
                            const double fact = d_ao_factor(ao_offset + il);
                            v_ao_c0(ipoint, ao_offset + il) = p1[i] * s1 * fact;
                            v_ao_c1(ipoint, ao_offset + il) = (p2[i] * s1 + p1[i] * s2) * fact;
                            v_ao_c2(ipoint, ao_offset + il) = (p3[i] * s1 + p1[i] * s3) * fact;
                            v_ao_c3(ipoint, ao_offset + il) = (p4[i] * s1 + p1[i] * s4) * fact;
                            v_ao_c4(ipoint, ao_offset + il) = (p5[i] * s1 + p1[i] * s5 + 2.0 * (p2[i] * s2 + p3[i] * s3 + p4[i] * s4)) * fact;
                        }
                        il++;
                    }
                } else if (l == 2) {
                    const double p1[6] = {x*x+shift, x*y+shift, x*z+shift, y*y+shift, y*z+shift, z*z+shift};
                    const double p2[6] = {2.0*x, y, z, 0.0, 0.0, 0.0};
                    const double p3[6] = {0.0, x, 0.0, 2.0*y, z, 0.0};
                    const double p4[6] = {0.0, 0.0, x, 0.0, y, 2.0*z};
                    const double p5[6] = {2.0, 0.0, 0.0, 2.0, 0.0, 2.0};
                    for (int i = 0; i < 6; ++i) {
                        if (ao_offset + il < ao_num) {
                            const double fact = d_ao_factor(ao_offset + il);
                            v_ao_c0(ipoint, ao_offset + il) = p1[i] * s1 * fact;
                            v_ao_c1(ipoint, ao_offset + il) = (p2[i] * s1 + p1[i] * s2) * fact;
                            v_ao_c2(ipoint, ao_offset + il) = (p3[i] * s1 + p1[i] * s3) * fact;
                            v_ao_c3(ipoint, ao_offset + il) = (p4[i] * s1 + p1[i] * s4) * fact;
                            v_ao_c4(ipoint, ao_offset + il) = (p5[i] * s1 + p1[i] * s5 + 2.0 * (p2[i] * s2 + p3[i] * s3 + p4[i] * s4)) * fact;
                        }
                        il++;
                    }
                } else {
                    for (int a = l; a >= 0; --a) {
                        for (int b = l - a; b >= 0; --b) {
                            const int c = l - a - b;
                            const double da = (double)a; const double db = (double)b; const double dc = (double)c;

                            const double xy = px[a+2] * py[b+2];
                            const double yz = py[b+2] * pz[c+2];
                            const double xz = px[a+2] * pz[c+2];

                            const double p1 = xy * pz[c+2] + shift;
                            const double dxy = xy * dc; const double dxz = xz * db; const double dyz = yz * da;

                            const double p2 = px[a+1] * dyz;
                            const double p3 = py[b+1] * dxz;
                            const double p4 = pz[c+1] * dxy;
                            const double p5 = (da-1.0) * px[a] * dyz + (db-1.0) * py[b] * dxz + (dc-1.0) * pz[c] * dxy;

                            if (ao_offset + il < ao_num) {
                                const double fact = d_ao_factor(ao_offset + il);
                                v_ao_c0(ipoint, ao_offset + il) = p1 * s1 * fact;
                                v_ao_c1(ipoint, ao_offset + il) = (p2 * s1 + p1 * s2) * fact;
                                v_ao_c2(ipoint, ao_offset + il) = (p3 * s1 + p1 * s3) * fact;
                                v_ao_c3(ipoint, ao_offset + il) = (p4 * s1 + p1 * s4) * fact;
                                v_ao_c4(ipoint, ao_offset + il) = (p5 * s1 + p1 * s5 + 2.0 * (p2 * s2 + p3 * s3 + p4 * s4)) * fact;
                            }
                            il++;
                        }
                    }
                }
            }
    });
}

// -------------------------------------------------------------------------------------------------
// 2D MDRange Kernels (GPU Acceleration: Parallelized across points & shells for maximum occupancy)
// -------------------------------------------------------------------------------------------------

template <int LMAX, typename CoordViewType, typename ViewType>
void kokkos_ao_gaussian_kernel_2d(
    const int64_t point_num, const int64_t ao_num, const int64_t shell_num, const int64_t nucl_num,
    const KokkosDeviceState& state, CoordViewType d_coord, ViewType d_ao_value)
{
    auto d_nucl_coord      = state.d_nucl_coord;
    auto d_nucleus_range   = state.d_nucleus_range;
    auto d_shell_ang_mom   = state.d_shell_ang_mom;
    auto d_shell_prim_idx  = state.d_shell_prim_idx;
    auto d_shell_prim_num  = state.d_shell_prim_num;
    auto d_shell_to_nucl   = state.d_shell_to_nucl;
    auto d_exponent        = state.d_exponent;
    auto d_coef_normalized = state.d_coef_normalized;
    auto d_ao_factor       = state.d_ao_factor;
    auto d_ao_index        = state.d_ao_index;

    // Tile size intentionally NOT specified here (previously hardcoded {32, 4}, never
    // empirically validated for this problem shape / GPU): KOKKOS_OR_KTUNE_PARALLEL_FOR
    // resolves to KTune::parallel_for when HAVE_KTUNE is defined, which searches and
    // caches tile sizes for MDRangePolicy - exactly the parameter/policy pairing KTune
    // is documented to optimize (unlike the CPU RangePolicy case, where its only lever
    // is chunk size). When HAVE_KTUNE is not defined, this falls back to plain
    // Kokkos::parallel_for with Kokkos's own default MDRangePolicy tiling heuristic.
    using Policy2D = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>;
    Policy2D policy({0, 0}, {point_num, shell_num});

    KOKKOS_OR_KTUNE_PARALLEL_FOR("AO_Gaussian_Value_2D", policy,
        KOKKOS_LAMBDA(const int64_t ipoint, const int64_t ishell) {
            constexpr double cutoff = 36.043653389117154;
            constexpr double shift = 1.e-20;

            const double e_x = d_coord(ipoint);
            const double e_y = d_coord(ipoint + point_num);
            const double e_z = d_coord(ipoint + 2 * point_num);

            const int32_t l = d_shell_ang_mom(ishell);
            if (l > LMAX) return;

            const int64_t inucl = d_shell_to_nucl(ishell);
            const double x = e_x - d_nucl_coord(inucl);
            const double y = e_y - d_nucl_coord(inucl + nucl_num);
            const double z = e_z - d_nucl_coord(inucl + 2 * nucl_num);
            const double r2 = x*x + y*y + z*z;

            const int64_t ao_offset = d_ao_index(ishell);
            const int deg = (l == 0) ? 1 : (l == 1) ? 3 : (l == 2) ? 6 : (l == 3) ? 10 : (l == 4) ? 15 : 21;

            if (r2 > cutoff * d_nucleus_range(inucl)) {
                for (int k = 0; k < deg && ao_offset + k < ao_num; ++k) {
                    d_ao_value(ipoint, ao_offset + k) = 0.0;
                }
                return;
            }

            double radial_sum = 0.0;
            const int64_t iprim_start = d_shell_prim_idx(ishell);
            const int64_t prim_count = d_shell_prim_num(ishell);

            for (int64_t p = 0; p < prim_count; ++p) {
                const int64_t iprim = iprim_start + p;
                const double alpha = d_exponent(iprim);
                const double ar2 = alpha * r2;
                if (ar2 <= cutoff) {
                    radial_sum += d_coef_normalized(iprim) * exp(-ar2);
                }
            }

            if (radial_sum == 0.0) {
                for (int k = 0; k < deg && ao_offset + k < ao_num; ++k) {
                    d_ao_value(ipoint, ao_offset + k) = 0.0;
                }
                return;
            }

            double pows_x[6], pows_y[6], pows_z[6];
            pows_x[0] = 1.0; pows_y[0] = 1.0; pows_z[0] = 1.0;
            for (int i = 1; i <= LMAX; ++i) {
                pows_x[i] = pows_x[i-1] * x;
                pows_y[i] = pows_y[i-1] * y;
                pows_z[i] = pows_z[i-1] * z;
            }

            int il = 0;
            if (l == 0) {
                if (ao_offset < ao_num) d_ao_value(ipoint, ao_offset) = radial_sum * d_ao_factor(ao_offset);
            } else if (l == 1) {
                const double vals[3] = {x + shift, y + shift, z + shift};
                for (int i = 0; i < 3; ++i) {
                    if (ao_offset + il < ao_num) d_ao_value(ipoint, ao_offset + il) = vals[i] * radial_sum * d_ao_factor(ao_offset + il);
                    il++;
                }
            } else if (l == 2) {
                const double vals[6] = {x*x + shift, x*y + shift, x*z + shift, y*y + shift, y*z + shift, z*z + shift};
                for (int i = 0; i < 6; ++i) {
                    if (ao_offset + il < ao_num) d_ao_value(ipoint, ao_offset + il) = vals[i] * radial_sum * d_ao_factor(ao_offset + il);
                    il++;
                }
            } else {
                for (int a = l; a >= 0; --a) {
                    for (int b = l - a; b >= 0; --b) {
                        const int c = l - a - b;
                        const double val = pows_x[a] * pows_y[b] * pows_z[c] + shift;
                        if (ao_offset + il < ao_num) d_ao_value(ipoint, ao_offset + il) = val * radial_sum * d_ao_factor(ao_offset + il);
                        il++;
                    }
                }
            }
    });
}

template <int LMAX, typename CoordViewType, typename ViewType2D>
void kokkos_ao_vgl_gaussian_kernel_2d(
    const int64_t point_num, const int64_t ao_num, const int64_t shell_num, const int64_t nucl_num,
    const KokkosDeviceState& state, CoordViewType d_coord,
    ViewType2D v_ao_c0, ViewType2D v_ao_c1, ViewType2D v_ao_c2, ViewType2D v_ao_c3, ViewType2D v_ao_c4)
{
    auto d_nucl_coord      = state.d_nucl_coord;
    auto d_nucleus_range   = state.d_nucleus_range;
    auto d_shell_ang_mom   = state.d_shell_ang_mom;
    auto d_shell_prim_idx  = state.d_shell_prim_idx;
    auto d_shell_prim_num  = state.d_shell_prim_num;
    auto d_shell_to_nucl   = state.d_shell_to_nucl;
    auto d_exponent        = state.d_exponent;
    auto d_coef_normalized = state.d_coef_normalized;
    auto d_ao_factor       = state.d_ao_factor;
    auto d_ao_index        = state.d_ao_index;

    // See the identical note in kokkos_ao_gaussian_kernel_2d above: tile size
    // intentionally not specified so KOKKOS_OR_KTUNE_PARALLEL_FOR (KTune::parallel_for
    // when HAVE_KTUNE is defined) can search and cache one for this policy/shape.
    using Policy2D = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>;
    Policy2D policy({0, 0}, {point_num, shell_num});

    KOKKOS_OR_KTUNE_PARALLEL_FOR("AO_Gaussian_VGL_2D", policy,
        KOKKOS_LAMBDA(const int64_t ipoint, const int64_t ishell) {
            constexpr double cutoff = 36.043653389117154;
            constexpr double shift = 1.e-20;

            const double e_x = d_coord(ipoint);
            const double e_y = d_coord(ipoint + point_num);
            const double e_z = d_coord(ipoint + 2 * point_num);

            const int32_t l = d_shell_ang_mom(ishell);
            if (l > LMAX) return;

            const int64_t inucl = d_shell_to_nucl(ishell);
            const double x = e_x - d_nucl_coord(inucl);
            const double y = e_y - d_nucl_coord(inucl + nucl_num);
            const double z = e_z - d_nucl_coord(inucl + 2 * nucl_num);
            const double r2 = x*x + y*y + z*z;

            const int64_t ao_offset = d_ao_index(ishell);
            const int deg = (l == 0) ? 1 : (l == 1) ? 3 : (l == 2) ? 6 : (l == 3) ? 10 : (l == 4) ? 15 : 21;

            if (r2 > cutoff * d_nucleus_range(inucl)) {
                for (int k = 0; k < deg && ao_offset + k < ao_num; ++k) {
                    v_ao_c0(ipoint, ao_offset + k) = 0.0;
                    v_ao_c1(ipoint, ao_offset + k) = 0.0;
                    v_ao_c2(ipoint, ao_offset + k) = 0.0;
                    v_ao_c3(ipoint, ao_offset + k) = 0.0;
                    v_ao_c4(ipoint, ao_offset + k) = 0.0;
                }
                return;
            }

            double s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0, s5 = 0.0;
            const int64_t iprim_start = d_shell_prim_idx(ishell);
            const int64_t prim_count = d_shell_prim_num(ishell);

            for (int64_t p = 0; p < prim_count; ++p) {
                const int64_t iprim = iprim_start + p;
                const double alpha = d_exponent(iprim);
                const double ar2_val = alpha * r2;

                if (ar2_val <= cutoff) {
                    const double exp_val = d_coef_normalized(iprim) * exp(-ar2_val);
                    const double f = -2.0 * alpha * exp_val;

                    s1 += exp_val;
                    s2 += f * x; s3 += f * y; s4 += f * z;
                    s5 += f * (3.0 - 2.0 * ar2_val);
                }
            }

            if (s1 == 0.0) {
                for (int k = 0; k < deg && ao_offset + k < ao_num; ++k) {
                    v_ao_c0(ipoint, ao_offset + k) = 0.0;
                    v_ao_c1(ipoint, ao_offset + k) = 0.0;
                    v_ao_c2(ipoint, ao_offset + k) = 0.0;
                    v_ao_c3(ipoint, ao_offset + k) = 0.0;
                    v_ao_c4(ipoint, ao_offset + k) = 0.0;
                }
                return;
            }

            double px[8], py[8], pz[8];
            px[0]=1.0; px[1]=1.0; px[2]=1.0;
            py[0]=1.0; py[1]=1.0; py[2]=1.0;
            pz[0]=1.0; pz[1]=1.0; pz[2]=1.0;

            for (int i = 1; i <= LMAX; ++i) {
                px[i+2] = px[i+1] * x;
                py[i+2] = py[i+1] * y;
                pz[i+2] = pz[i+1] * z;
            }

            int il = 0;
            if (l == 0) {
                const double p1 = 1.0, p2 = 0.0, p3 = 0.0, p4 = 0.0, p5 = 0.0;
                if (ao_offset < ao_num) {
                    const double fact = d_ao_factor(ao_offset);
                    v_ao_c0(ipoint, ao_offset) = p1 * s1 * fact;
                    v_ao_c1(ipoint, ao_offset) = (p2 * s1 + p1 * s2) * fact;
                    v_ao_c2(ipoint, ao_offset) = (p3 * s1 + p1 * s3) * fact;
                    v_ao_c3(ipoint, ao_offset) = (p4 * s1 + p1 * s4) * fact;
                    v_ao_c4(ipoint, ao_offset) = (p5 * s1 + p1 * s5 + 2.0 * (p2 * s2 + p3 * s3 + p4 * s4)) * fact;
                }
            } else if (l == 1) {
                const double p1[3] = {x+shift, y+shift, z+shift};
                const double p2[3] = {1.0, 0.0, 0.0};
                const double p3[3] = {0.0, 1.0, 0.0};
                const double p4[3] = {0.0, 0.0, 1.0};
                const double p5[3] = {0.0, 0.0, 0.0};
                for (int i = 0; i < 3; ++i) {
                    if (ao_offset + il < ao_num) {
                        const double fact = d_ao_factor(ao_offset + il);
                        v_ao_c0(ipoint, ao_offset + il) = p1[i] * s1 * fact;
                        v_ao_c1(ipoint, ao_offset + il) = (p2[i] * s1 + p1[i] * s2) * fact;
                        v_ao_c2(ipoint, ao_offset + il) = (p3[i] * s1 + p1[i] * s3) * fact;
                        v_ao_c3(ipoint, ao_offset + il) = (p4[i] * s1 + p1[i] * s4) * fact;
                        v_ao_c4(ipoint, ao_offset + il) = (p5[i] * s1 + p1[i] * s5 + 2.0 * (p2[i] * s2 + p3[i] * s3 + p4[i] * s4)) * fact;
                    }
                    il++;
                }
            } else if (l == 2) {
                const double p1[6] = {x*x+shift, x*y+shift, x*z+shift, y*y+shift, y*z+shift, z*z+shift};
                const double p2[6] = {2.0*x, y, z, 0.0, 0.0, 0.0};
                const double p3[6] = {0.0, x, 0.0, 2.0*y, z, 0.0};
                const double p4[6] = {0.0, 0.0, x, 0.0, y, 2.0*z};
                const double p5[6] = {2.0, 0.0, 0.0, 2.0, 0.0, 2.0};
                for (int i = 0; i < 6; ++i) {
                    if (ao_offset + il < ao_num) {
                        const double fact = d_ao_factor(ao_offset + il);
                        v_ao_c0(ipoint, ao_offset + il) = p1[i] * s1 * fact;
                        v_ao_c1(ipoint, ao_offset + il) = (p2[i] * s1 + p1[i] * s2) * fact;
                        v_ao_c2(ipoint, ao_offset + il) = (p3[i] * s1 + p1[i] * s3) * fact;
                        v_ao_c3(ipoint, ao_offset + il) = (p4[i] * s1 + p1[i] * s4) * fact;
                        v_ao_c4(ipoint, ao_offset + il) = (p5[i] * s1 + p1[i] * s5 + 2.0 * (p2[i] * s2 + p3[i] * s3 + p4[i] * s4)) * fact;
                    }
                    il++;
                }
            } else {
                for (int a = l; a >= 0; --a) {
                    for (int b = l - a; b >= 0; --b) {
                        const int c = l - a - b;
                        const double da = (double)a; const double db = (double)b; const double dc = (double)c;

                        const double xy = px[a+2] * py[b+2];
                        const double yz = py[b+2] * pz[c+2];
                        const double xz = px[a+2] * pz[c+2];

                        const double p1 = xy * pz[c+2] + shift;
                        const double dxy = xy * dc; const double dxz = xz * db; const double dyz = yz * da;

                        const double p2 = px[a+1] * dyz;
                        const double p3 = py[b+1] * dxz;
                        const double p4 = pz[c+1] * dxy;
                        const double p5 = (da-1.0) * px[a] * dyz + (db-1.0) * py[b] * dxz + (dc-1.0) * pz[c] * dxy;

                        if (ao_offset + il < ao_num) {
                            const double fact = d_ao_factor(ao_offset + il);
                            v_ao_c0(ipoint, ao_offset + il) = p1 * s1 * fact;
                            v_ao_c1(ipoint, ao_offset + il) = (p2 * s1 + p1 * s2) * fact;
                            v_ao_c2(ipoint, ao_offset + il) = (p3 * s1 + p1 * s3) * fact;
                            v_ao_c3(ipoint, ao_offset + il) = (p4 * s1 + p1 * s4) * fact;
                            v_ao_c4(ipoint, ao_offset + il) = (p5 * s1 + p1 * s5 + 2.0 * (p2 * s2 + p3 * s3 + p4 * s4)) * fact;
                        }
                        il++;
                    }
                }
            }
    });
}

extern "C" {

void* qmckl_kokkos_state_create() {
    KokkosDeviceState* s = new KokkosDeviceState();
    register_state(s);
    return static_cast<void*>(s);
}

void qmckl_kokkos_state_destroy(void* state_ptr) {
    if (state_ptr) {
        KokkosDeviceState* s = static_cast<KokkosDeviceState*>(state_ptr);
        // Unregister BEFORE resetting: once removed from the registry, a concurrent
        // process-wide reset_all_states() (triggered by Kokkos finalize on ref-count
        // reaching zero) can no longer touch this state's Views, avoiding a double
        // teardown race between an explicit context_destroy and process shutdown.
        unregister_state(s);
        s->reset();
        delete s;
    }
}

void* qmckl_kokkos_state_copy(const void* src_ptr) {
    if (!src_ptr) return nullptr;

    const KokkosDeviceState* src = static_cast<const KokkosDeviceState*>(src_ptr);
    KokkosDeviceState* dest = new KokkosDeviceState();
    register_state(dest);

    dest->basis_initialized = src->basis_initialized;
    dest->mo_initialized    = src->mo_initialized;
    dest->lmax_global       = src->lmax_global;

    auto duplicate_view = [](auto& dst_view, const auto& src_view, const std::string& label) {
        if (src_view.extent(0) == 0) return;
        using ViewType = typename std::remove_reference<decltype(dst_view)>::type;
        if constexpr (ViewType::rank == 1) {
            dst_view = ViewType(label, src_view.extent(0));
        } else if constexpr (ViewType::rank == 2) {
            dst_view = ViewType(label, src_view.extent(0), src_view.extent(1));
        } else if constexpr (ViewType::rank == 3) {
            dst_view = ViewType(label, src_view.extent(0), src_view.extent(1), src_view.extent(2));
        }
        Kokkos::deep_copy(dst_view, src_view);
    };

    duplicate_view(dest->d_coord, src->d_coord, "Coordinates");
    duplicate_view(dest->d_nucl_coord, src->d_nucl_coord, "nucl_coord");
    duplicate_view(dest->d_nucleus_index, src->d_nucleus_index, "nucl_idx");
    duplicate_view(dest->d_nucleus_shell_num, src->d_nucleus_shell_num, "nucl_shell_num");
    duplicate_view(dest->d_nucleus_range, src->d_nucleus_range, "nucl_range");
    duplicate_view(dest->d_shell_ang_mom, src->d_shell_ang_mom, "shell_ang_mom");
    duplicate_view(dest->d_shell_prim_idx, src->d_shell_prim_idx, "shell_prim_idx");
    duplicate_view(dest->d_shell_prim_num, src->d_shell_prim_num, "shell_prim_num");
    duplicate_view(dest->d_shell_to_nucl, src->d_shell_to_nucl, "shell_to_nucl");
    duplicate_view(dest->d_exponent, src->d_exponent, "exponent");
    duplicate_view(dest->d_coef_normalized, src->d_coef_normalized, "coef_norm");
    duplicate_view(dest->d_ao_factor, src->d_ao_factor, "ao_factor");
    duplicate_view(dest->d_ao_index, src->d_ao_index, "ao_index");

    duplicate_view(dest->d_ao_value, src->d_ao_value, "AO_Values");
    duplicate_view(dest->d_mo_value, src->d_mo_value, "MO_Values");
    duplicate_view(dest->d_ao_vgl_stacked, src->d_ao_vgl_stacked, "AO_VGL_Stacked");
    duplicate_view(dest->d_mo_vgl_stacked, src->d_mo_vgl_stacked, "MO_VGL_Stacked");
    duplicate_view(dest->d_mo_coef, src->d_mo_coef, "MO_Coefs");

    duplicate_view(dest->h_mo_coef_raw, src->h_mo_coef_raw, "MO_Coefs_Raw");
    duplicate_view(dest->h_mo_coef_mirror, src->h_mo_coef_mirror, "MO_Coefs_Mirror");
    duplicate_view(dest->h_ao_val_mirror, src->h_ao_val_mirror, "h_ao_val_mirror");
    duplicate_view(dest->h_mo_val_mirror, src->h_mo_val_mirror, "h_mo_val_mirror");
    duplicate_view(dest->h_ao_vgl_stacked_mirror, src->h_ao_vgl_stacked_mirror, "h_ao_vgl_stacked_mirror");
    duplicate_view(dest->h_mo_vgl_stacked_mirror, src->h_mo_vgl_stacked_mirror, "h_mo_vgl_stacked_mirror");

    return static_cast<void*>(dest);
}

qmckl_exit_code qmckl_kokkos_finalize_ao(qmckl_context context) {
    qmckl_context_struct* ctx = (qmckl_context_struct*) context;
    if (!ctx || !ctx->qmckl_extra) return QMCKL_INVALID_CONTEXT;

    KokkosDeviceState& state = *static_cast<KokkosDeviceState*>(ctx->qmckl_extra);

    const int64_t ao_num = ctx->ao_basis.ao_num;
    const int64_t shell_num = ctx->ao_basis.shell_num;
    const int64_t nucl_num = ctx->nucleus.num;

    if (ao_num <= 0 || shell_num <= 0 || nucl_num <= 0) return QMCKL_SUCCESS;

    if (state.basis_initialized && state.d_ao_factor.extent(0) == ao_num) return QMCKL_SUCCESS;

    int64_t total_prims = 0;
    qmckl_exit_code rc = qmckl_get_ao_basis_prim_num(context, &total_prims);
    if (rc != QMCKL_SUCCESS || total_prims <= 0) return rc;

    std::vector<double> h_coef_norm(total_prims);
    for (int64_t ishell = 0; ishell < shell_num; ++ishell) {
        for (int64_t ip = ctx->ao_basis.shell_prim_index[ishell];
             ip < ctx->ao_basis.shell_prim_index[ishell] + ctx->ao_basis.shell_prim_num[ishell]; ++ip) {
            h_coef_norm[ip] = ctx->ao_basis.coefficient[ip] * ctx->ao_basis.prim_factor[ip] * ctx->ao_basis.shell_factor[ishell];
        }
    }

    Kokkos::View<const int64_t*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_nucl_idx(ctx->ao_basis.nucleus_index, nucl_num);
    Kokkos::View<const int64_t*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_nucl_shell_num(ctx->ao_basis.nucleus_shell_num, nucl_num);
    Kokkos::View<const double*,  HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_nucl_range(ctx->ao_basis.nucleus_range, nucl_num);
    Kokkos::View<const double*,  HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_nucl_coord(ctx->nucleus.coord.data, 3 * nucl_num);
    Kokkos::View<const int32_t*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_shell_ang_mom(ctx->ao_basis.shell_ang_mom, shell_num);
    Kokkos::View<const int64_t*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_shell_prim_idx(ctx->ao_basis.shell_prim_index, shell_num);
    Kokkos::View<const int64_t*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_shell_prim_num(ctx->ao_basis.shell_prim_num, shell_num);
    Kokkos::View<const double*,  HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_exponent(ctx->ao_basis.exponent, total_prims);
    Kokkos::View<const double*,  HostSpace> h_coef_norm_view(h_coef_norm.data(), total_prims);
    Kokkos::View<const double*,  HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_ao_factor(ctx->ao_basis.ao_factor, ao_num);

    Kokkos::View<int64_t*, HostSpace> h_ao_index("H_AO_Index", shell_num);
    Kokkos::View<int64_t*, HostSpace> h_shell_to_nucl("H_Shell_To_Nucl", shell_num);

    int32_t lstart_host[32];
    for (int l = 0; l < 32; ++l) lstart_host[l] = l * (l + 1) * (l + 2) / 6;

    int64_t k_off = 0;
    for (int inucl = 0; inucl < nucl_num; ++inucl) {
        int64_t s_start = ctx->ao_basis.nucleus_index[inucl];
        int64_t s_end = s_start + ctx->ao_basis.nucleus_shell_num[inucl];
        for (int64_t ishell = s_start; ishell < s_end; ++ishell) {
            h_ao_index(ishell) = k_off;
            h_shell_to_nucl(ishell) = inucl;
            k_off += lstart_host[ctx->ao_basis.shell_ang_mom[ishell] + 1] - lstart_host[ctx->ao_basis.shell_ang_mom[ishell]];
        }
    }

    state.d_nucleus_index = Kokkos::View<int64_t*, DeviceSpace>("nucl_idx", nucl_num);
    Kokkos::deep_copy(state.d_nucleus_index, h_nucl_idx);

    state.d_nucleus_shell_num = Kokkos::View<int64_t*, DeviceSpace>("nucl_shell_num", nucl_num);
    Kokkos::deep_copy(state.d_nucleus_shell_num, h_nucl_shell_num);

    state.d_nucleus_range = Kokkos::View<double*, DeviceSpace>("nucl_range", nucl_num);
    Kokkos::deep_copy(state.d_nucleus_range, h_nucl_range);

    state.d_nucl_coord = Kokkos::View<double*, DeviceSpace>("nucl_coord", 3 * nucl_num);
    Kokkos::deep_copy(state.d_nucl_coord, h_nucl_coord);

    state.d_shell_ang_mom = Kokkos::View<int32_t*, DeviceSpace>("shell_ang_mom", shell_num);
    Kokkos::deep_copy(state.d_shell_ang_mom, h_shell_ang_mom);

    state.d_shell_prim_idx = Kokkos::View<int64_t*, DeviceSpace>("shell_prim_idx", shell_num);
    Kokkos::deep_copy(state.d_shell_prim_idx, h_shell_prim_idx);

    state.d_shell_prim_num = Kokkos::View<int64_t*, DeviceSpace>("shell_prim_num", shell_num);
    Kokkos::deep_copy(state.d_shell_prim_num, h_shell_prim_num);

    state.d_shell_to_nucl = Kokkos::View<int64_t*, DeviceSpace>("shell_to_nucl", shell_num);
    Kokkos::deep_copy(state.d_shell_to_nucl, h_shell_to_nucl);

    state.d_exponent = Kokkos::View<double*, DeviceSpace>("exponent", total_prims);
    Kokkos::deep_copy(state.d_exponent, h_exponent);

    state.d_coef_normalized = Kokkos::View<double*, DeviceSpace>("coef_norm", total_prims);
    Kokkos::deep_copy(state.d_coef_normalized, h_coef_norm_view);

    state.d_ao_factor = Kokkos::View<double*, DeviceSpace>("ao_factor", ao_num);
    Kokkos::deep_copy(state.d_ao_factor, h_ao_factor);

    state.d_ao_index = Kokkos::View<int64_t*, DeviceSpace>("ao_index", shell_num);
    Kokkos::deep_copy(state.d_ao_index, h_ao_index);

    state.lmax_global = 0;
    for (int i = 0; i < shell_num; ++i) {
        if (ctx->ao_basis.shell_ang_mom[i] > state.lmax_global) state.lmax_global = ctx->ao_basis.shell_ang_mom[i];
    }

    state.basis_initialized = true;
    return QMCKL_SUCCESS;
}

qmckl_exit_code qmckl_kokkos_finalize_mo(qmckl_context context) {
    qmckl_context_struct* ctx = (qmckl_context_struct*) context;
    if (!ctx || !ctx->qmckl_extra) return QMCKL_INVALID_CONTEXT;

    KokkosDeviceState& state = *static_cast<KokkosDeviceState*>(ctx->qmckl_extra);

    const int64_t ao_num = ctx->ao_basis.ao_num;
    const int64_t mo_num = ctx->mo_basis.mo_num;

    if (ao_num <= 0 || mo_num <= 0) return QMCKL_SUCCESS;

    const double* coefficient_t = ctx->mo_basis.coefficient_t;
    if (!coefficient_t) return QMCKL_SUCCESS;

    if (state.mo_initialized && state.d_mo_coef.extent(0) == ao_num && state.d_mo_coef.extent(1) == mo_num) {
        if (std::memcmp(state.h_mo_coef_raw.data(), coefficient_t, ao_num * mo_num * sizeof(double)) == 0) {
            return QMCKL_SUCCESS;
        }
    }

    state.d_mo_coef = Kokkos::View<double**, DeviceLayout, DeviceSpace>("MO_Coefs", ao_num, mo_num);
    state.h_mo_coef_mirror = Kokkos::View<double**, DeviceLayout, HostSpace>("MO_Coefs_Mirror", ao_num, mo_num);
    state.h_mo_coef_raw = Kokkos::View<double**, Kokkos::LayoutRight, HostSpace>("MO_Coefs_Raw", ao_num, mo_num);

    Kokkos::View<const double**, Kokkos::LayoutRight, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> v_mo_coef(coefficient_t, ao_num, mo_num);
    Kokkos::deep_copy(state.h_mo_coef_raw, v_mo_coef);
    Kokkos::deep_copy(state.h_mo_coef_mirror, state.h_mo_coef_raw);
    Kokkos::deep_copy(state.d_mo_coef, state.h_mo_coef_mirror);

    state.mo_initialized = true;
    return QMCKL_SUCCESS;
}

qmckl_exit_code qmckl_compute_ao_value_kokkos(
    const qmckl_context context, const int64_t ao_num, const int64_t shell_num,
    const int32_t* /*prim_num_per_nucleus*/, const int64_t point_num, const int64_t nucl_num,
    const double* coord, const double* /*nucl_coord*/, const int64_t* /*nucleus_index*/,
    const int64_t* /*nucleus_shell_num*/, const double* /*nucleus_range*/, const int32_t* /*shell_ang_mom*/,
    const int64_t* /*shell_prim_index*/, const int64_t* /*shell_prim_num*/, const double* /*exponent*/,
    const double* /*coefficient*/, const double* /*ao_factor*/, const double* /*shell_vgl*/,
    double* const ao_value)
{
    qmckl_context_struct* ctx = (qmckl_context_struct*) context;
    KokkosDeviceState& state = *static_cast<KokkosDeviceState*>(ctx->qmckl_extra);

    if constexpr (is_host_accessible) {
        if (ao_value != nullptr) {
            Kokkos::View<const double*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_coord(coord, 3 * point_num);
            Kokkos::View<double**, Kokkos::LayoutRight, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_ao_value(ao_value, point_num, ao_num);

            if (state.lmax_global <= 1) { kokkos_ao_gaussian_kernel_coalesced<1>(point_num, ao_num, shell_num, nucl_num, state, h_coord, h_ao_value); }
            else if (state.lmax_global == 2) { kokkos_ao_gaussian_kernel_coalesced<2>(point_num, ao_num, shell_num, nucl_num, state, h_coord, h_ao_value); }
            else if (state.lmax_global == 3) { kokkos_ao_gaussian_kernel_coalesced<3>(point_num, ao_num, shell_num, nucl_num, state, h_coord, h_ao_value); }
            else if (state.lmax_global == 4) { kokkos_ao_gaussian_kernel_coalesced<4>(point_num, ao_num, shell_num, nucl_num, state, h_coord, h_ao_value); }
            else if (state.lmax_global == 5) { kokkos_ao_gaussian_kernel_coalesced<5>(point_num, ao_num, shell_num, nucl_num, state, h_coord, h_ao_value); }
            else { return QMCKL_FAILURE; }

            Kokkos::fence();
            return QMCKL_SUCCESS;
        }
    }

    if (state.d_coord.extent(0) != 3 * point_num) {
        state.d_coord = Kokkos::View<double*, DeviceSpace>("Coordinates", 3 * point_num);
    }
    Kokkos::deep_copy(state.d_coord, Kokkos::View<const double*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(coord, 3 * point_num));

    if (state.d_ao_value.extent(0) != point_num || state.d_ao_value.extent(1) != ao_num) {
        state.d_ao_value = Kokkos::View<double**, DeviceLayout, DeviceSpace>("AO_Values", point_num, ao_num);
    }
    auto d_ao_value_local = state.d_ao_value;

    if (state.lmax_global <= 1) { kokkos_ao_gaussian_kernel_2d<1>(point_num, ao_num, shell_num, nucl_num, state, state.d_coord, d_ao_value_local); }
    else if (state.lmax_global == 2) { kokkos_ao_gaussian_kernel_2d<2>(point_num, ao_num, shell_num, nucl_num, state, state.d_coord, d_ao_value_local); }
    else if (state.lmax_global == 3) { kokkos_ao_gaussian_kernel_2d<3>(point_num, ao_num, shell_num, nucl_num, state, state.d_coord, d_ao_value_local); }
    else if (state.lmax_global == 4) { kokkos_ao_gaussian_kernel_2d<4>(point_num, ao_num, shell_num, nucl_num, state, state.d_coord, d_ao_value_local); }
    else if (state.lmax_global == 5) { kokkos_ao_gaussian_kernel_2d<5>(point_num, ao_num, shell_num, nucl_num, state, state.d_coord, d_ao_value_local); }
    else { return QMCKL_FAILURE; }

    Kokkos::fence();

    if (ao_value != nullptr && !should_bypass_ao_sync()) {
        if (state.h_ao_val_mirror.extent(0) != point_num || state.h_ao_val_mirror.extent(1) != ao_num) {
            state.h_ao_val_mirror = Kokkos::View<double**, DeviceLayout, HostSpace>("h_ao_val_mirror", point_num, ao_num);
        }
        Kokkos::deep_copy(state.h_ao_val_mirror, d_ao_value_local);
        Kokkos::deep_copy(Kokkos::View<double**, Kokkos::LayoutRight, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(ao_value, point_num, ao_num), state.h_ao_val_mirror);
    }
    return QMCKL_SUCCESS;
}

qmckl_exit_code qmckl_compute_ao_vgl_kokkos(
    const qmckl_context context, const int64_t ao_num, const int64_t shell_num,
    const int32_t* /*prim_num_per_nucleus*/, const int64_t point_num, const int64_t nucl_num,
    const double* coord, const double* /*nucl_coord*/, const int64_t* /*nucleus_index*/,
    const int64_t* /*nucleus_shell_num*/, const double* /*nucleus_range*/, const int32_t* /*shell_ang_mom*/,
    const int64_t* /*shell_prim_index*/, const int64_t* /*shell_prim_num*/, const double* /*exponent*/,
    const double* /*coefficient*/, const double* /*ao_factor*/, const double* /*shell_vgl*/,
    double* const ao_vgl)
{
    qmckl_context_struct* ctx = (qmckl_context_struct*) context;
    KokkosDeviceState& state = *static_cast<KokkosDeviceState*>(ctx->qmckl_extra);

    if constexpr (is_host_accessible) {
        if (ao_vgl != nullptr) {
            Kokkos::View<const double*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_coord(coord, 3 * point_num);
            Kokkos::View<double***, Kokkos::LayoutRight, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_ao_vgl(ao_vgl, point_num, 5, ao_num);

            auto c0 = Kokkos::subview(h_ao_vgl, Kokkos::ALL(), 0, Kokkos::ALL());
            auto c1 = Kokkos::subview(h_ao_vgl, Kokkos::ALL(), 1, Kokkos::ALL());
            auto c2 = Kokkos::subview(h_ao_vgl, Kokkos::ALL(), 2, Kokkos::ALL());
            auto c3 = Kokkos::subview(h_ao_vgl, Kokkos::ALL(), 3, Kokkos::ALL());
            auto c4 = Kokkos::subview(h_ao_vgl, Kokkos::ALL(), 4, Kokkos::ALL());

            if (state.lmax_global <= 1) { kokkos_ao_vgl_gaussian_kernel_coalesced<1>(point_num, ao_num, shell_num, nucl_num, state, h_coord, c0, c1, c2, c3, c4); }
            else if (state.lmax_global == 2) { kokkos_ao_vgl_gaussian_kernel_coalesced<2>(point_num, ao_num, shell_num, nucl_num, state, h_coord, c0, c1, c2, c3, c4); }
            else if (state.lmax_global == 3) { kokkos_ao_vgl_gaussian_kernel_coalesced<3>(point_num, ao_num, shell_num, nucl_num, state, h_coord, c0, c1, c2, c3, c4); }
            else if (state.lmax_global == 4) { kokkos_ao_vgl_gaussian_kernel_coalesced<4>(point_num, ao_num, shell_num, nucl_num, state, h_coord, c0, c1, c2, c3, c4); }
            else if (state.lmax_global == 5) { kokkos_ao_vgl_gaussian_kernel_coalesced<5>(point_num, ao_num, shell_num, nucl_num, state, h_coord, c0, c1, c2, c3, c4); }
            else { return QMCKL_FAILURE; }

            Kokkos::fence();
            return QMCKL_SUCCESS;
        }
    }

    if (state.d_coord.extent(0) != 3 * point_num) {
        state.d_coord = Kokkos::View<double*, DeviceSpace>("Coordinates", 3 * point_num);
    }
    Kokkos::deep_copy(state.d_coord, Kokkos::View<const double*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(coord, 3 * point_num));

    if (state.d_ao_vgl_stacked.extent(0) != 5 * point_num || state.d_ao_vgl_stacked.extent(1) != ao_num) {
        state.d_ao_vgl_stacked = Kokkos::View<double**, DeviceLayout, DeviceSpace>(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "AO_VGL_Stacked"), 5 * point_num, ao_num);
    }
    auto stacked = state.d_ao_vgl_stacked;

    auto c0 = Kokkos::subview(stacked, Kokkos::pair<int64_t,int64_t>(0 * point_num, 1 * point_num), Kokkos::ALL());
    auto c1 = Kokkos::subview(stacked, Kokkos::pair<int64_t,int64_t>(1 * point_num, 2 * point_num), Kokkos::ALL());
    auto c2 = Kokkos::subview(stacked, Kokkos::pair<int64_t,int64_t>(2 * point_num, 3 * point_num), Kokkos::ALL());
    auto c3 = Kokkos::subview(stacked, Kokkos::pair<int64_t,int64_t>(3 * point_num, 4 * point_num), Kokkos::ALL());
    auto c4 = Kokkos::subview(stacked, Kokkos::pair<int64_t,int64_t>(4 * point_num, 5 * point_num), Kokkos::ALL());

    if (state.lmax_global <= 1) { kokkos_ao_vgl_gaussian_kernel_2d<1>(point_num, ao_num, shell_num, nucl_num, state, state.d_coord, c0, c1, c2, c3, c4); }
    else if (state.lmax_global == 2) { kokkos_ao_vgl_gaussian_kernel_2d<2>(point_num, ao_num, shell_num, nucl_num, state, state.d_coord, c0, c1, c2, c3, c4); }
    else if (state.lmax_global == 3) { kokkos_ao_vgl_gaussian_kernel_2d<3>(point_num, ao_num, shell_num, nucl_num, state, state.d_coord, c0, c1, c2, c3, c4); }
    else if (state.lmax_global == 4) { kokkos_ao_vgl_gaussian_kernel_2d<4>(point_num, ao_num, shell_num, nucl_num, state, state.d_coord, c0, c1, c2, c3, c4); }
    else if (state.lmax_global == 5) { kokkos_ao_vgl_gaussian_kernel_2d<5>(point_num, ao_num, shell_num, nucl_num, state, state.d_coord, c0, c1, c2, c3, c4); }
    else { return QMCKL_FAILURE; }

    Kokkos::fence();

    if (ao_vgl != nullptr && !should_bypass_ao_sync()) {
        Kokkos::View<double***, Kokkos::LayoutRight, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_ao_vgl_out(ao_vgl, point_num, 5, ao_num);

        if (state.h_ao_vgl_stacked_mirror.extent(0) != 5 * point_num || state.h_ao_vgl_stacked_mirror.extent(1) != ao_num) {
            state.h_ao_vgl_stacked_mirror = Kokkos::View<double**, DeviceLayout, HostSpace>(
                Kokkos::view_alloc(Kokkos::WithoutInitializing, "h_ao_vgl_stacked_mirror"), 5 * point_num, ao_num);
        }
        Kokkos::deep_copy(state.h_ao_vgl_stacked_mirror, stacked);

        auto h_mirror = state.h_ao_vgl_stacked_mirror;
        Kokkos::parallel_for("Scatter_AO_VGL_Host", Kokkos::RangePolicy<HostExecSpace>(0, point_num),
            KOKKOS_LAMBDA(const int64_t pt) {
                for (int c = 0; c < 5; ++c) {
                    for (int64_t ao = 0; ao < ao_num; ++ao) {
                        h_ao_vgl_out(pt, c, ao) = h_mirror(c * point_num + pt, ao);
                    }
                }
        });
    }
    return QMCKL_SUCCESS;
}

qmckl_exit_code qmckl_compute_mo_basis_mo_value_kokkos(
    const qmckl_context context, const int64_t ao_num, const int64_t mo_num,
    const int64_t point_num, const double* /*coefficient_t*/,
    const double* ao_value_host, double* const mo_value)
{
    qmckl_context_struct* ctx = (qmckl_context_struct*) context;
    KokkosDeviceState& state = *static_cast<KokkosDeviceState*>(ctx->qmckl_extra);

    // --- Host path (CPU / OpenMP) -----------------------------------------------------
    // Faithful port of qmckl_compute_mo_basis_mo_value_hpc's compact-then-process
    // structure (qmckl_mo.c): per point, first scan ao_num entries and compact the
    // non-zero AO indices/values into a dense list (cheap, O(ao_num) scalar scan, no
    // FMA work), THEN process that compacted list with an 8-way unrolled, branch-free
    // SIMD loop using raw pointer arithmetic into the coefficient matrix.
    //
    // This replaces an earlier version that blocked 4 *different* points together and
    // skipped only when all 4 were simultaneously zero. For points drawn from
    // independent walkers/electrons, P(all 4 zero) = (1-density)^4 - at this dataset's
    // ~26.64% density that is ~29%, so ~71% of AOs still triggered the FMA pass, and
    // when triggered, all 4 points were computed unconditionally (including individually
    // -zero ones). That is why the earlier version cost about the same as a dense GEMM:
    // the cross-point block check barely screened anything relative to per-point
    // screening. This version screens per point, exactly matching native's granularity,
    // and avoids native's *other* source of overhead too: doing the zero-check via
    // repeated Kokkos::View::operator() calls (each a runtime strided-offset
    // computation) interleaved with the hot loop, instead of resolving it once via a
    // cheap upfront scan into a compacted list that the hot loop then runs over with no
    // further branching.
    //
    // Scratch for the compaction pass is a plain Kokkos::View sized (point_num, ao_num)
    // and indexed only by the loop variable `pt` - NOT by a thread id. This project
    // builds with Kokkos_ENABLE_CUDA_LAMBDA=ON, under which KOKKOS_LAMBDA expands to
    // `[=] __host__ __device__` unconditionally, so this lambda body must remain valid
    // to compile as device code even though (via is_host_accessible) it only ever
    // executes on host: thread_local storage and calls to host-only functions such as
    // omp_get_thread_num() are hard CUDA compile errors inside a __device__-tagged
    // closure, and a non-template `if constexpr` does not exempt the discarded branch
    // from that semantic check the way it would inside a template. Per-point Kokkos::View
    // scratch sidesteps this entirely - it is the same kind of access already used
    // throughout this file for both host- and device-targeted code.
    //
    // The scratch is CACHED in KokkosDeviceState and only resized when point_num/ao_num
    // change (see the fields' declaration there for why this must not be a fresh
    // per-call allocation - an earlier version that allocated fresh Views inside this
    // function body measurably regressed performance, since first-touch page faults on
    // freshly-mapped memory landed inside the profiled kernel time on every one of the
    // ~10 calls per Monte Carlo step).
    if constexpr (is_host_accessible) {
        if (mo_value != nullptr) {
            Kokkos::View<const double**, Kokkos::LayoutRight, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_ao(ao_value_host, point_num, ao_num);
            Kokkos::View<double**, Kokkos::LayoutRight, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_mo(mo_value, point_num, mo_num);
            // Raw contiguous pointer into the coefficient matrix (LayoutRight(ao_num, mo_num),
            // uploaded once at finalize time) - avoids View::operator() overhead in the hot loop,
            // exactly mirroring native's `coefficient_t + idx[n]*mo_num` pointer arithmetic.
            const double* const coef_data = state.d_mo_coef.data();

            if (state.h_mo_value_idx_scratch.extent(0) != point_num || state.h_mo_value_idx_scratch.extent(1) != ao_num) {
                state.h_mo_value_idx_scratch = Kokkos::View<int64_t**, Kokkos::LayoutRight, HostSpace>(
                    Kokkos::view_alloc(Kokkos::WithoutInitializing, "mo_value_idx_scratch"), point_num, ao_num);
                state.h_mo_value_av_scratch = Kokkos::View<double**, Kokkos::LayoutRight, HostSpace>(
                    Kokkos::view_alloc(Kokkos::WithoutInitializing, "mo_value_av_scratch"), point_num, ao_num);
            }
            auto idx_scratch = state.h_mo_value_idx_scratch;
            auto av_scratch = state.h_mo_value_av_scratch;

            Kokkos::parallel_for("Kokkos_Sparse_MO_Value_CPU",
                Kokkos::RangePolicy<HostExecSpace>(0, point_num),
                KOKKOS_LAMBDA(const int64_t pt) {
                    int64_t* const __restrict__ idx = &idx_scratch(pt, 0);
                    double* const __restrict__ av1 = &av_scratch(pt, 0);

                    // h_ao(pt, .) and h_mo(pt, .) are contiguous rows of a LayoutRight(point, X)
                    // view, so taking their address gives a valid contiguous raw pointer.
                    const double* const __restrict__ avgl1 = &h_ao(pt, 0);
                    double* const __restrict__ vgl1 = &h_mo(pt, 0);

                    for (int64_t i = 0; i < mo_num; ++i) {
                        vgl1[i] = 0.0;
                    }

                    // Pass 1 (compaction): cheap linear scan, no FMA work at all.
                    int64_t nidx = 0;
                    for (int64_t k = 0; k < ao_num; ++k) {
                        const double v = avgl1[k];
                        if (v != 0.0) {
                            idx[nidx] = k;
                            av1[nidx] = v;
                            ++nidx;
                        }
                    }

                    // Pass 2 (compute): branch-free, 8-way unrolled, over the compacted list
                    // only. Boundary matches native's exactly (n < nidx-8): for nidx < 8 this
                    // never fires and everything is handled by the scalar tail below.
                    int64_t n = 0;
                    for (n = 0; n < nidx - 8; n += 8) {
                        const double* const ck1 = coef_data + idx[n    ] * mo_num;
                        const double* const ck2 = coef_data + idx[n + 1] * mo_num;
                        const double* const ck3 = coef_data + idx[n + 2] * mo_num;
                        const double* const ck4 = coef_data + idx[n + 3] * mo_num;
                        const double* const ck5 = coef_data + idx[n + 4] * mo_num;
                        const double* const ck6 = coef_data + idx[n + 5] * mo_num;
                        const double* const ck7 = coef_data + idx[n + 6] * mo_num;
                        const double* const ck8 = coef_data + idx[n + 7] * mo_num;

                        const double a1 = av1[n    ];
                        const double a2 = av1[n + 1];
                        const double a3 = av1[n + 2];
                        const double a4 = av1[n + 3];
                        const double a5 = av1[n + 4];
                        const double a6 = av1[n + 5];
                        const double a7 = av1[n + 6];
                        const double a8 = av1[n + 7];

                        #pragma omp simd
                        for (int64_t i = 0; i < mo_num; ++i) {
                            vgl1[i] = vgl1[i] +
                                ck1[i] * a1 + ck2[i] * a2 + ck3[i] * a3 + ck4[i] * a4 +
                                ck5[i] * a5 + ck6[i] * a6 + ck7[i] * a7 + ck8[i] * a8;
                        }
                    }

                    for (int64_t m = n; m < nidx; ++m) {
                        const double* const ck = coef_data + idx[m] * mo_num;
                        const double a1 = av1[m];
                        #pragma omp simd
                        for (int64_t i = 0; i < mo_num; ++i) {
                            vgl1[i] = vgl1[i] + ck[i] * a1;
                        }
                    }
            });

            Kokkos::fence();
            return QMCKL_SUCCESS;
        }
    }

    // --- Device path (GPU) --------------------------------------------------------
    if (state.d_mo_value.extent(0) != point_num || state.d_mo_value.extent(1) != mo_num) {
        state.d_mo_value = Kokkos::View<double**, DeviceLayout, DeviceSpace>("MO_Values", point_num, mo_num);
    }
    auto d_mo_value_local = state.d_mo_value;

    Kokkos::View<double**, DeviceLayout, DeviceSpace> d_ao;
    if (state.d_ao_value.extent(0) == point_num && state.d_ao_value.extent(1) == ao_num) {
        d_ao = state.d_ao_value;
    } else {
        d_ao = Kokkos::View<double**, DeviceLayout, DeviceSpace>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "d_ao_temp"), point_num, ao_num);
        Kokkos::View<double**, DeviceLayout, HostSpace> h_ao_temp(Kokkos::view_alloc(Kokkos::WithoutInitializing, "h_ao_temp"), point_num, ao_num);
        Kokkos::deep_copy(h_ao_temp, Kokkos::View<const double**, Kokkos::LayoutRight, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(ao_value_host, point_num, ao_num));
        Kokkos::deep_copy(d_ao, h_ao_temp);
    }

    KokkosBlas::gemm("N", "N", 1.0, d_ao, state.d_mo_coef, 0.0, d_mo_value_local);
    Kokkos::fence();

    if (mo_value != nullptr && !should_bypass_ao_sync()) {
        if (state.h_mo_val_mirror.extent(0) != point_num || state.h_mo_val_mirror.extent(1) != mo_num) {
            state.h_mo_val_mirror = Kokkos::View<double**, DeviceLayout, HostSpace>("h_mo_val_mirror", point_num, mo_num);
        }
        Kokkos::deep_copy(state.h_mo_val_mirror, d_mo_value_local);
        Kokkos::deep_copy(Kokkos::View<double**, Kokkos::LayoutRight, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(mo_value, point_num, mo_num), state.h_mo_val_mirror);
    }
    return QMCKL_SUCCESS;
}

qmckl_exit_code qmckl_compute_mo_basis_mo_vgl_kokkos(
    const qmckl_context context, const int64_t ao_num, const int64_t mo_num,
    const int64_t point_num, const double* /*coefficient_t*/,
    const double* ao_vgl_host, double* const mo_vgl)
{
    qmckl_context_struct* ctx = (qmckl_context_struct*) context;
    KokkosDeviceState& state = *static_cast<KokkosDeviceState*>(ctx->qmckl_extra);

    // --- Host path (CPU / OpenMP) -----------------------------------------------------
    // Faithful port of qmckl_compute_mo_basis_mo_vgl_hpc's compact-then-process
    // structure (qmckl_mo.c): the zero-check granularity here (all 5 VGL components of
    // the SAME (point, ao) pair) was already correct in the previous version - native's
    // own cutoff/radial-zero branches zero all 5 components together, so checking the
    // value component alone (native's own compaction condition is `avgl1[k] != 0.`,
    // checking only the value array, exactly as done below) exactly reconstructs
    // per-point, per-AO sparsity. What was costing performance was doing that check via
    // five Kokkos::View::operator() calls (each a runtime strided-offset computation)
    // interleaved with the hot loop, for every one of ao_num AOs regardless of outcome -
    // measured at ~3.8x the per-useful-FLOP cost of the equivalent dense GEMM. This
    // version separates the check from the compute exactly as native does: one cheap
    // upfront scan compacts non-zero AO indices and their 5 component values into dense
    // arrays, then a branch-free, 4-way unrolled SIMD loop (native uses 4-way here, not
    // 8-way as in mo_value, because 5 live accumulator lines per unrolled entry costs
    // more registers than the 1-array mo_value case) processes only the compacted list
    // via raw pointer arithmetic.
    //
    // Scratch for the compaction pass is a plain Kokkos::View sized (point_num, ao_num)
    // per component and indexed only by the loop variable `pt` - see the identical note
    // in qmckl_compute_mo_basis_mo_value_kokkos above for why (this project builds with
    // Kokkos_ENABLE_CUDA_LAMBDA=ON, so KOKKOS_LAMBDA is host+device tagged
    // unconditionally, and thread_local / host-only calls such as omp_get_thread_num()
    // inside this lambda would be CUDA compile errors even though this branch only ever
    // executes on host). The scratch is CACHED in KokkosDeviceState and only resized
    // when point_num/ao_num change, for the same reason given in mo_value above.
    if constexpr (is_host_accessible) {
        if (mo_vgl != nullptr && ao_vgl_host != nullptr) {
            Kokkos::View<const double***, Kokkos::LayoutRight, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_ao_vgl(ao_vgl_host, point_num, 5, ao_num);
            Kokkos::View<double***, Kokkos::LayoutRight, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_mo_vgl(mo_vgl, point_num, 5, mo_num);
            const double* const coef_data = state.d_mo_coef.data();

            if (state.h_mo_vgl_idx_scratch.extent(0) != point_num || state.h_mo_vgl_idx_scratch.extent(1) != ao_num) {
                state.h_mo_vgl_idx_scratch = Kokkos::View<int64_t**, Kokkos::LayoutRight, HostSpace>(
                    Kokkos::view_alloc(Kokkos::WithoutInitializing, "mo_vgl_idx_scratch"), point_num, ao_num);
                state.h_mo_vgl_av1_scratch = Kokkos::View<double**, Kokkos::LayoutRight, HostSpace>(
                    Kokkos::view_alloc(Kokkos::WithoutInitializing, "mo_vgl_av1_scratch"), point_num, ao_num);
                state.h_mo_vgl_av2_scratch = Kokkos::View<double**, Kokkos::LayoutRight, HostSpace>(
                    Kokkos::view_alloc(Kokkos::WithoutInitializing, "mo_vgl_av2_scratch"), point_num, ao_num);
                state.h_mo_vgl_av3_scratch = Kokkos::View<double**, Kokkos::LayoutRight, HostSpace>(
                    Kokkos::view_alloc(Kokkos::WithoutInitializing, "mo_vgl_av3_scratch"), point_num, ao_num);
                state.h_mo_vgl_av4_scratch = Kokkos::View<double**, Kokkos::LayoutRight, HostSpace>(
                    Kokkos::view_alloc(Kokkos::WithoutInitializing, "mo_vgl_av4_scratch"), point_num, ao_num);
                state.h_mo_vgl_av5_scratch = Kokkos::View<double**, Kokkos::LayoutRight, HostSpace>(
                    Kokkos::view_alloc(Kokkos::WithoutInitializing, "mo_vgl_av5_scratch"), point_num, ao_num);
            }
            auto idx_scratch = state.h_mo_vgl_idx_scratch;
            auto av1_scratch = state.h_mo_vgl_av1_scratch;
            auto av2_scratch = state.h_mo_vgl_av2_scratch;
            auto av3_scratch = state.h_mo_vgl_av3_scratch;
            auto av4_scratch = state.h_mo_vgl_av4_scratch;
            auto av5_scratch = state.h_mo_vgl_av5_scratch;

            Kokkos::parallel_for("Kokkos_Sparse_MO_VGL_CPU",
                Kokkos::RangePolicy<HostExecSpace>(0, point_num),
                KOKKOS_LAMBDA(const int64_t pt) {
                    int64_t* const __restrict__ idx = &idx_scratch(pt, 0);
                    double* const __restrict__ av1 = &av1_scratch(pt, 0);
                    double* const __restrict__ av2 = &av2_scratch(pt, 0);
                    double* const __restrict__ av3 = &av3_scratch(pt, 0);
                    double* const __restrict__ av4 = &av4_scratch(pt, 0);
                    double* const __restrict__ av5 = &av5_scratch(pt, 0);

                    // h_ao_vgl(pt, c, ao) offset = pt*5*ao_num + c*ao_num + ao: component c is a
                    // CONTIGUOUS length-ao_num row starting at that offset, matching native's
                    // avgl1..avgl5 pointer layout exactly.
                    const double* const __restrict__ avgl1 = &h_ao_vgl(pt, 0, 0);
                    const double* const __restrict__ avgl2 = &h_ao_vgl(pt, 1, 0);
                    const double* const __restrict__ avgl3 = &h_ao_vgl(pt, 2, 0);
                    const double* const __restrict__ avgl4 = &h_ao_vgl(pt, 3, 0);
                    const double* const __restrict__ avgl5 = &h_ao_vgl(pt, 4, 0);

                    double* const __restrict__ vgl1 = &h_mo_vgl(pt, 0, 0);
                    double* const __restrict__ vgl2 = &h_mo_vgl(pt, 1, 0);
                    double* const __restrict__ vgl3 = &h_mo_vgl(pt, 2, 0);
                    double* const __restrict__ vgl4 = &h_mo_vgl(pt, 3, 0);
                    double* const __restrict__ vgl5 = &h_mo_vgl(pt, 4, 0);

                    for (int64_t i = 0; i < mo_num; ++i) {
                        vgl1[i] = 0.0; vgl2[i] = 0.0; vgl3[i] = 0.0; vgl4[i] = 0.0; vgl5[i] = 0.0;
                    }

                    // Pass 1 (compaction): a single check on the value component (avgl1)
                    // suffices, since native zeros all 5 components together whenever this
                    // AO is screened out for this point, and native's own compaction check
                    // (qmckl_compute_mo_basis_mo_vgl_hpc) also tests only avgl1[k] != 0.
                    int64_t nidx = 0;
                    for (int64_t k = 0; k < ao_num; ++k) {
                        const double v1 = avgl1[k];
                        if (v1 != 0.0) {
                            idx[nidx] = k;
                            av1[nidx] = v1;
                            av2[nidx] = avgl2[k];
                            av3[nidx] = avgl3[k];
                            av4[nidx] = avgl4[k];
                            av5[nidx] = avgl5[k];
                            ++nidx;
                        }
                    }

                    // Pass 2 (compute): branch-free, 4-way unrolled (matching native's own
                    // unroll depth for the 5-component case), over the compacted list only.
                    int64_t n = 0;
                    for (n = 0; n < nidx - 4; n += 4) {
                        const double* const ck1 = coef_data + idx[n    ] * mo_num;
                        const double* const ck2 = coef_data + idx[n + 1] * mo_num;
                        const double* const ck3 = coef_data + idx[n + 2] * mo_num;
                        const double* const ck4 = coef_data + idx[n + 3] * mo_num;

                        const double a11 = av1[n    ], a21 = av1[n + 1], a31 = av1[n + 2], a41 = av1[n + 3];
                        const double a12 = av2[n    ], a22 = av2[n + 1], a32 = av2[n + 2], a42 = av2[n + 3];
                        const double a13 = av3[n    ], a23 = av3[n + 1], a33 = av3[n + 2], a43 = av3[n + 3];
                        const double a14 = av4[n    ], a24 = av4[n + 1], a34 = av4[n + 2], a44 = av4[n + 3];
                        const double a15 = av5[n    ], a25 = av5[n + 1], a35 = av5[n + 2], a45 = av5[n + 3];

                        #pragma omp simd
                        for (int64_t i = 0; i < mo_num; ++i) {
                            vgl1[i] = vgl1[i] + ck1[i] * a11 + ck2[i] * a21 + ck3[i] * a31 + ck4[i] * a41;
                            vgl2[i] = vgl2[i] + ck1[i] * a12 + ck2[i] * a22 + ck3[i] * a32 + ck4[i] * a42;
                            vgl3[i] = vgl3[i] + ck1[i] * a13 + ck2[i] * a23 + ck3[i] * a33 + ck4[i] * a43;
                            vgl4[i] = vgl4[i] + ck1[i] * a14 + ck2[i] * a24 + ck3[i] * a34 + ck4[i] * a44;
                            vgl5[i] = vgl5[i] + ck1[i] * a15 + ck2[i] * a25 + ck3[i] * a35 + ck4[i] * a45;
                        }
                    }

                    for (int64_t m = n; m < nidx; ++m) {
                        const double* const ck = coef_data + idx[m] * mo_num;
                        const double a1 = av1[m], a2 = av2[m], a3 = av3[m], a4 = av4[m], a5 = av5[m];
                        #pragma omp simd
                        for (int64_t i = 0; i < mo_num; ++i) {
                            vgl1[i] = vgl1[i] + ck[i] * a1;
                            vgl2[i] = vgl2[i] + ck[i] * a2;
                            vgl3[i] = vgl3[i] + ck[i] * a3;
                            vgl4[i] = vgl4[i] + ck[i] * a4;
                            vgl5[i] = vgl5[i] + ck[i] * a5;
                        }
                    }
            });

            Kokkos::fence();
            return QMCKL_SUCCESS;
        }
    }

    // --- Device path (GPU) --------------------------------------------------------
    if (state.d_mo_vgl_stacked.extent(0) != 5 * point_num || state.d_mo_vgl_stacked.extent(1) != mo_num) {
        state.d_mo_vgl_stacked = Kokkos::View<double**, DeviceLayout, DeviceSpace>(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "MO_VGL_Stacked"), 5 * point_num, mo_num);
    }

    bool use_device_ao = (state.d_ao_vgl_stacked.extent(0) == 5 * point_num && state.d_ao_vgl_stacked.extent(1) == ao_num);

    if (!use_device_ao) {
        if (state.d_ao_vgl_stacked.extent(0) != 5 * point_num || state.d_ao_vgl_stacked.extent(1) != ao_num) {
            state.d_ao_vgl_stacked = Kokkos::View<double**, DeviceLayout, DeviceSpace>(
                Kokkos::view_alloc(Kokkos::WithoutInitializing, "AO_VGL_Stacked"), 5 * point_num, ao_num);
        }
        if (state.h_ao_vgl_stacked_mirror.extent(0) != 5 * point_num || state.h_ao_vgl_stacked_mirror.extent(1) != ao_num) {
            state.h_ao_vgl_stacked_mirror = Kokkos::View<double**, DeviceLayout, HostSpace>(
                Kokkos::view_alloc(Kokkos::WithoutInitializing, "h_ao_vgl_stacked_mirror"), 5 * point_num, ao_num);
        }

        Kokkos::View<const double***, Kokkos::LayoutRight, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_ao_vgl_in(ao_vgl_host, point_num, 5, ao_num);
        auto h_mirror = state.h_ao_vgl_stacked_mirror;

        // Single combined host-side gather ...
        Kokkos::parallel_for("Gather_AO_VGL_Fallback_Host", Kokkos::RangePolicy<HostExecSpace>(0, point_num),
            KOKKOS_LAMBDA(const int64_t pt) {
                for (int c = 0; c < 5; ++c) {
                    for (int64_t a = 0; a < ao_num; ++a) {
                        h_mirror(c * point_num + pt, a) = h_ao_vgl_in(pt, c, a);
                    }
                }
        });
        // ... followed by a SINGLE H2D copy (instead of five).
        Kokkos::deep_copy(state.d_ao_vgl_stacked, h_mirror);
    }

    // Single combined GEMM: (5*point_num, ao_num) x (ao_num, mo_num) -> (5*point_num, mo_num).
    KokkosBlas::gemm("N", "N", 1.0, state.d_ao_vgl_stacked, state.d_mo_coef, 0.0, state.d_mo_vgl_stacked);
    Kokkos::fence();

    if (mo_vgl != nullptr && !should_bypass_ao_sync()) {
        Kokkos::View<double***, Kokkos::LayoutRight, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_mo_vgl_out(mo_vgl, point_num, 5, mo_num);

        if (state.h_mo_vgl_stacked_mirror.extent(0) != 5 * point_num || state.h_mo_vgl_stacked_mirror.extent(1) != mo_num) {
            state.h_mo_vgl_stacked_mirror = Kokkos::View<double**, DeviceLayout, HostSpace>(
                Kokkos::view_alloc(Kokkos::WithoutInitializing, "h_mo_vgl_stacked_mirror"), 5 * point_num, mo_num);
        }
        Kokkos::deep_copy(state.h_mo_vgl_stacked_mirror, state.d_mo_vgl_stacked);

        auto h_mirror = state.h_mo_vgl_stacked_mirror;
        Kokkos::parallel_for("Scatter_MO_VGL_Host", Kokkos::RangePolicy<HostExecSpace>(0, point_num),
            KOKKOS_LAMBDA(const int64_t pt) {
                for (int c = 0; c < 5; ++c) {
                    for (int64_t m = 0; m < mo_num; ++m) {
                        h_mo_vgl_out(pt, c, m) = h_mirror(c * point_num + pt, m);
                    }
                }
        });
    }

    return QMCKL_SUCCESS;
}

// -------------------------------------------------------------------------------------------------
// "Extract from a richer, already-fresh quantity" dispatch functions.
// -------------------------------------------------------------------------------------------------

qmckl_exit_code qmckl_extract_ao_value_from_vgl_kokkos(
    const qmckl_context context, const int64_t ao_num, const int64_t point_num,
    const double* ao_vgl_host, double* const ao_value)
{
    qmckl_context_struct* ctx = (qmckl_context_struct*) context;
    if (!ctx || !ctx->qmckl_extra) return QMCKL_INVALID_CONTEXT;
    KokkosDeviceState& state = *static_cast<KokkosDeviceState*>(ctx->qmckl_extra);

    if constexpr (is_host_accessible) {
        if (ao_value == nullptr || ao_vgl_host == nullptr) return QMCKL_FAILURE;
        for (int64_t i = 0; i < point_num; ++i) {
            std::memcpy(ao_value + i * ao_num, ao_vgl_host + i * 5 * ao_num, ao_num * sizeof(double));
        }
        return QMCKL_SUCCESS;
    }

    if (state.d_ao_vgl_stacked.extent(0) != 5 * point_num || state.d_ao_vgl_stacked.extent(1) != ao_num) {
        return QMCKL_FAILURE;
    }

    // Allocate explicitly contiguous LayoutLeft view instead of generating a LayoutStride subview
    if (state.d_ao_value.extent(0) != point_num || state.d_ao_value.extent(1) != ao_num) {
        state.d_ao_value = Kokkos::View<double**, DeviceLayout, DeviceSpace>("AO_Values", point_num, ao_num);
    }
    
    auto d_ao_val = state.d_ao_value;
    auto stacked = state.d_ao_vgl_stacked;
    
    Kokkos::parallel_for("Extract_AO_Val", 
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {point_num, ao_num}),
        KOKKOS_LAMBDA(const int64_t pt, const int64_t a) {
            d_ao_val(pt, a) = stacked(pt, a);
        });
    Kokkos::fence();

    if (ao_value != nullptr && !should_bypass_ao_sync()) {
        if (state.h_ao_val_mirror.extent(0) != point_num || state.h_ao_val_mirror.extent(1) != ao_num) {
            state.h_ao_val_mirror = Kokkos::View<double**, DeviceLayout, HostSpace>("h_ao_val_mirror", point_num, ao_num);
        }
        Kokkos::deep_copy(state.h_ao_val_mirror, state.d_ao_value);
        Kokkos::deep_copy(Kokkos::View<double**, Kokkos::LayoutRight, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(ao_value, point_num, ao_num), state.h_ao_val_mirror);
    }
    return QMCKL_SUCCESS;
}

qmckl_exit_code qmckl_extract_mo_value_from_vgl_kokkos(
    const qmckl_context context, const int64_t mo_num, const int64_t point_num,
    const double* mo_vgl_host, double* const mo_value)
{
    qmckl_context_struct* ctx = (qmckl_context_struct*) context;
    if (!ctx || !ctx->qmckl_extra) return QMCKL_INVALID_CONTEXT;
    KokkosDeviceState& state = *static_cast<KokkosDeviceState*>(ctx->qmckl_extra);

    if constexpr (is_host_accessible) {
        if (mo_value == nullptr || mo_vgl_host == nullptr) return QMCKL_FAILURE;
        for (int64_t i = 0; i < point_num; ++i) {
            std::memcpy(mo_value + i * mo_num, mo_vgl_host + i * 5 * mo_num, mo_num * sizeof(double));
        }
        return QMCKL_SUCCESS;
    }

    if (state.d_mo_vgl_stacked.extent(0) != 5 * point_num || state.d_mo_vgl_stacked.extent(1) != mo_num) {
        return QMCKL_FAILURE;
    }

    // Allocate explicitly contiguous LayoutLeft view instead of generating a LayoutStride subview
    if (state.d_mo_value.extent(0) != point_num || state.d_mo_value.extent(1) != mo_num) {
        state.d_mo_value = Kokkos::View<double**, DeviceLayout, DeviceSpace>("MO_Values", point_num, mo_num);
    }
    
    auto d_mo_val = state.d_mo_value;
    auto stacked = state.d_mo_vgl_stacked;

    Kokkos::parallel_for("Extract_MO_Val", 
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {point_num, mo_num}),
        KOKKOS_LAMBDA(const int64_t pt, const int64_t m) {
            d_mo_val(pt, m) = stacked(pt, m);
        });
    Kokkos::fence();

    if (mo_value != nullptr && !should_bypass_ao_sync()) {
        if (state.h_mo_val_mirror.extent(0) != point_num || state.h_mo_val_mirror.extent(1) != mo_num) {
            state.h_mo_val_mirror = Kokkos::View<double**, DeviceLayout, HostSpace>("h_mo_val_mirror", point_num, mo_num);
        }
        Kokkos::deep_copy(state.h_mo_val_mirror, state.d_mo_value);
        Kokkos::deep_copy(Kokkos::View<double**, Kokkos::LayoutRight, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(mo_value, point_num, mo_num), state.h_mo_val_mirror);
    }
    return QMCKL_SUCCESS;
}

static std::atomic<size_t> kokkos_ref_count{0};

void qmckl_kokkos_initialize() {
    if (kokkos_ref_count.fetch_add(1) == 0) {
        if (!Kokkos::is_initialized()) {
            Kokkos::initialize();
        }
#ifdef HAVE_KTUNE
        if (!KTune::is_initialized()) {
            KTune::initialize();
        }
        // Configure the tiling search used by the GPU AO kernels (the only remaining
        // KTune::parallel_for call sites in this file, after the CPU MO kernels were
        // reverted to plain Kokkos::parallel_for - see the compact-then-process notes
        // on qmckl_compute_mo_basis_mo_value_kokkos / _mo_vgl_kokkos above). A prior run
        // with default settings (Nelder-Mead, GPU's default 10 min iterations / 0.1s
        // per-candidate budget) converged back to essentially the original hardcoded
        // {32,4} tile's performance, with no measurable gain.
        //
        // Two changes, both applied only to KTune's global tuning configuration (there
        // is nothing else left for this to affect, since the CPU MO kernels no longer
        // use KTune at all):
        //
        // 1. Switch to PrunedSearch (type 1) instead of the default Nelder-Mead (0).
        //    Nelder-Mead is a continuous simplex search; a GPU MDRangePolicy tile size
        //    is a small, discrete, integer-valued 2D space (confirmed from KTune's own
        //    source: PrunedSearch_gpu explicitly respects policy.max_total_tile_size(),
        //    a hardware occupancy constraint that a generic simplex search has no
        //    special awareness of), so a systematic pruned search over candidate
        //    tilings is a better-matched strategy than a small number of simplex
        //    evaluations for this specific kind of search space.
        // 2. Raise the per-candidate benchmarking budget. GPU's default budget (10
        //    minimum iterations, 0.1s max time per candidate) is smaller than CPU's
        //    default (20 / 0.2s) to begin with, and this AO kernel completes in
        //    single-digit milliseconds - too few repeated measurements per candidate
        //    risks ordinary run-to-run timing noise misleading the search into
        //    reporting a spuriously "best" tile size rather than a genuinely faster
        //    one. The values below (30 iterations, 0.5s) are modest increases: since
        //    tuning happens once per run and is cached to disk (KTune's own on-disk
        //    cache within a job's run directory), this adds at most a few seconds to
        //    the one-time "Tuning" pass, not to any subsequently-cached "Running" pass.
        KTune::Core::set_tiling_optimizer_type(1);
        KTune::Core::set_min_tuning_iterations(30);
        KTune::Core::set_max_tuning_time(0.5);
#endif
    }
}

void qmckl_kokkos_finalize() {
    if (kokkos_ref_count.fetch_sub(1) == 1) {
        if (Kokkos::is_initialized()) {
            Kokkos::fence();
            reset_all_states();
            Kokkos::fence();
            Kokkos::finalize();
        }
    }
}

} // extern "C"
