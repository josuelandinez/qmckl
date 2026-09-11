#include "qmckl.h"
#include <Kokkos_Core.hpp>
#include <unordered_map>
#include <cmath>
#include <cstring>
#include <iostream>
#include <atomic>
#include <mutex>
#include <vector>

using DeviceSpace = Kokkos::DefaultExecutionSpace::memory_space;
using HostSpace = Kokkos::DefaultHostExecutionSpace::memory_space;

// ===================================================================
// 1. STATEFUL DEVICE REGISTRY (Thread-Safe Persistent Cache)
// ===================================================================
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
    Kokkos::View<double*, DeviceSpace> d_exponent;
    Kokkos::View<double*, DeviceSpace> d_coef_normalized;
    Kokkos::View<double*, DeviceSpace> d_ao_factor;
    Kokkos::View<int64_t*, DeviceSpace> d_ao_index;

    // Standard LayoutRight across Device and Host guarantees contiguous DMA transfers
    Kokkos::View<double**, Kokkos::LayoutRight, DeviceSpace> d_mo_coef;
    Kokkos::View<double**, Kokkos::LayoutRight, DeviceSpace> d_ao_value;
    Kokkos::View<double**, Kokkos::LayoutRight, DeviceSpace> d_mo_value;
    Kokkos::View<double***, Kokkos::LayoutRight, DeviceSpace> d_ao_vgl;
    Kokkos::View<double***, Kokkos::LayoutRight, DeviceSpace> d_mo_vgl;

    Kokkos::View<double**, Kokkos::LayoutRight, HostSpace> h_mo_coef_mirror;
};

static std::unordered_map<qmckl_context, KokkosDeviceState> device_cache;
static std::mutex cache_mutex;
static std::atomic<size_t> kokkos_ref_count{0};

// ===================================================================
// 2. COMPUTE KERNELS (Fully Saturated across point_num)
// ===================================================================

template <int LMAX>
void kokkos_ao_gaussian_kernel(
    const int64_t point_num, const int64_t ao_num, const int64_t shell_num, const int64_t nucl_num,
    const KokkosDeviceState& state)
{
    constexpr int N_POLY = (LMAX + 1) * (LMAX + 2) * (LMAX + 3) / 6;

    auto d_coord = state.d_coord;
    auto d_nucl_coord = state.d_nucl_coord;
    auto d_nucleus_index = state.d_nucleus_index;
    auto d_nucleus_shell_num = state.d_nucleus_shell_num;
    auto d_nucleus_range = state.d_nucleus_range;
    auto d_shell_ang_mom = state.d_shell_ang_mom;
    auto d_shell_prim_idx = state.d_shell_prim_idx;
    auto d_shell_prim_num = state.d_shell_prim_num;
    auto d_exponent = state.d_exponent;
    auto d_coef_normalized = state.d_coef_normalized;
    auto d_ao_factor = state.d_ao_factor;
    auto d_ao_index = state.d_ao_index;
    auto d_ao_value = state.d_ao_value;

    Kokkos::parallel_for("AO_Gaussian_Value", point_num, KOKKOS_LAMBDA(const int ipoint) {
        double cutoff = 36.043653389117154; 
        int32_t local_lstart[32];
        for (int l = 0; l < 32; ++l) local_lstart[l] = l * (l + 1) * (l + 2) / 6;
        for (int k = 0; k < ao_num; ++k) d_ao_value(ipoint, k) = 0.0;

        double e_x = d_coord(ipoint);
        double e_y = d_coord(ipoint + point_num);
        double e_z = d_coord(ipoint + 2 * point_num);

        for (int inucl = 0; inucl < nucl_num; ++inucl) {
            double n_x = d_nucl_coord(inucl);
            double n_y = d_nucl_coord(inucl + nucl_num);
            double n_z = d_nucl_coord(inucl + 2 * nucl_num);

            double x = e_x - n_x;
            double y = e_y - n_y;
            double z = e_z - n_z;
            double r2 = x*x + y*y + z*z;

            if (r2 > cutoff * d_nucleus_range(inucl)) continue;

            double poly[N_POLY];
            poly[0] = 1.0;
            if constexpr (LMAX >= 1) {
                double shift = 1.e-20;
                poly[1] = x + shift; poly[2] = y + shift; poly[3] = z + shift;
                if constexpr (LMAX >= 2) {
                    poly[4] = x*x + shift; poly[5] = x*y + shift; poly[6] = x*z + shift;
                    poly[7] = y*y + shift; poly[8] = y*z + shift; poly[9] = z*z + shift;
                }
                if constexpr (LMAX >= 3) {
                    double pows[3][LMAX + 3];
                    for (int i = 0; i < 3; ++i) { pows[0][i] = 1.0; pows[1][i] = 1.0; pows[2][i] = 1.0; }
                    for (int i = 3; i <= LMAX + 2; ++i) {
                        pows[0][i] = pows[0][i - 1] * x; pows[1][i] = pows[1][i - 1] * y; pows[2][i] = pows[2][i - 1] * z;
                    }
                    int m = 10;
                    for (int d = 3; d <= LMAX; ++d) {
                        for (int a = d; a >= 0; --a) {
                            for (int b = d - a; b >= 0; --b) {
                                int c = d - a - b;
                                double xy = pows[0][a + 2] * pows[1][b + 2];
                                poly[m] = xy * pows[2][c + 2] + shift;
                                ++m;
                            }
                        }
                    }
                }
            }

            int64_t ishell_start = d_nucleus_index(inucl);
            int64_t ishell_end = ishell_start + d_nucleus_shell_num(inucl);

            for (int64_t ishell = ishell_start; ishell < ishell_end; ++ishell) {
                int32_t l = d_shell_ang_mom(ishell);
                if (l > LMAX) continue;

                double radial_sum = 0.0;
                int64_t iprim_start = d_shell_prim_idx(ishell);
                int64_t iprim_end = iprim_start + d_shell_prim_num(ishell);

                for (int64_t iprim = iprim_start; iprim < iprim_end; ++iprim) {
                    double ar2 = d_exponent(iprim) * r2;
                    if (ar2 <= cutoff) {
                        radial_sum += d_coef_normalized(iprim) * exp(-ar2);
                    }
                }
                if (radial_sum == 0.0) continue;

                int64_t ao_offset = d_ao_index(ishell);
                int32_t n_angular = local_lstart[l + 1] - local_lstart[l];
                int32_t p_idx = local_lstart[l];

                for (int il = 0; il < n_angular; ++il) {
                    d_ao_value(ipoint, ao_offset + il) += poly[p_idx + il] * radial_sum * d_ao_factor(ao_offset + il);
                }
            }
        }
    });
}

template <int LMAX>
void kokkos_ao_vgl_gaussian_kernel(
    const int64_t point_num, const int64_t ao_num, const int64_t shell_num, const int64_t nucl_num,
    const KokkosDeviceState& state)
{
    constexpr int N_POLY = (LMAX + 1) * (LMAX + 2) * (LMAX + 3) / 6;

    auto d_coord = state.d_coord;
    auto d_nucl_coord = state.d_nucl_coord;
    auto d_nucleus_index = state.d_nucleus_index;
    auto d_nucleus_shell_num = state.d_nucleus_shell_num;
    auto d_nucleus_range = state.d_nucleus_range;
    auto d_shell_ang_mom = state.d_shell_ang_mom;
    auto d_shell_prim_idx = state.d_shell_prim_idx;
    auto d_shell_prim_num = state.d_shell_prim_num;
    auto d_exponent = state.d_exponent;
    auto d_coef_normalized = state.d_coef_normalized;
    auto d_ao_factor = state.d_ao_factor;
    auto d_ao_index = state.d_ao_index;
    auto d_ao_vgl = state.d_ao_vgl;

    Kokkos::parallel_for("AO_Gaussian_VGL", point_num, KOKKOS_LAMBDA(const int ipoint) {
        double cutoff = 36.043653389117154; 
        int32_t local_lstart[32];
        for (int l = 0; l < 32; ++l) local_lstart[l] = l * (l + 1) * (l + 2) / 6;
        for (int k = 0; k < ao_num; ++k) {
            for(int comp=0; comp<5; ++comp) d_ao_vgl(ipoint, comp, k) = 0.0;
        }

        double e_x = d_coord(ipoint);
        double e_y = d_coord(ipoint + point_num);
        double e_z = d_coord(ipoint + 2 * point_num);

        for (int inucl = 0; inucl < nucl_num; ++inucl) {
            double n_x = d_nucl_coord(inucl);
            double n_y = d_nucl_coord(inucl + nucl_num);
            double n_z = d_nucl_coord(inucl + 2 * nucl_num);

            double x = e_x - n_x;
            double y = e_y - n_y;
            double z = e_z - n_z;
            double r2 = x*x + y*y + z*z;

            if (r2 > cutoff * d_nucleus_range(inucl)) continue;

            double poly_vgl[5][N_POLY];
            for(int c=0; c<5; ++c) { for(int p=0; p<N_POLY; ++p) poly_vgl[c][p] = 0.0; }
            
            poly_vgl[0][0] = 1.0;
            if constexpr (LMAX >= 1) {
                double shift = 1.e-20;
                poly_vgl[0][1] = x + shift; poly_vgl[0][2] = y + shift; poly_vgl[0][3] = z + shift;
                poly_vgl[1][1] = 1.0;       poly_vgl[2][2] = 1.0;       poly_vgl[3][3] = 1.0;
                
                if constexpr (LMAX >= 2) {
                    poly_vgl[0][4] = x*x + shift; poly_vgl[0][5] = x*y + shift; poly_vgl[0][6] = x*z + shift;
                    poly_vgl[0][7] = y*y + shift; poly_vgl[0][8] = y*z + shift; poly_vgl[0][9] = z*z + shift;
                    
                    poly_vgl[1][4] = 2.0*x; poly_vgl[1][5] = y; poly_vgl[1][6] = z;
                    poly_vgl[2][5] = x;     poly_vgl[2][7] = 2.0*y; poly_vgl[2][8] = z;
                    poly_vgl[3][6] = x;     poly_vgl[3][8] = y;     poly_vgl[3][9] = 2.0*z;
                    
                    poly_vgl[4][4] = 2.0;   poly_vgl[4][7] = 2.0;   poly_vgl[4][9] = 2.0;
                }
                if constexpr (LMAX >= 3) {
                    double pows[3][LMAX + 3];
                    for (int i = 0; i < 3; ++i) { pows[0][i] = 1.0; pows[1][i] = 1.0; pows[2][i] = 1.0; }
                    for (int i = 3; i <= LMAX + 2; ++i) {
                        pows[0][i] = pows[0][i - 1] * x; pows[1][i] = pows[1][i - 1] * y; pows[2][i] = pows[2][i - 1] * z;
                    }
                    
                    int m = 10;
                    double dd = 3.0;
                    for (int d = 3; d <= LMAX; ++d) {
                        double da = dd;
                        for (int a = d; a >= 0; --a) {
                            double db = dd - da;
                            for (int b = d - a; b >= 0; --b) {
                                int c = d - a - b;
                                double dc = dd - da - db;

                                double xy = pows[0][a+2] * pows[1][b+2];
                                double yz = pows[1][b+2] * pows[2][c+2];
                                double xz = pows[0][a+2] * pows[2][c+2];

                                poly_vgl[0][m] = xy * pows[2][c+2] + shift;
                                xy *= dc; xz *= db; yz *= da;

                                poly_vgl[1][m] = pows[0][a+1] * yz;
                                poly_vgl[2][m] = pows[1][b+1] * xz;
                                poly_vgl[3][m] = pows[2][c+1] * xy;
                                poly_vgl[4][m] = (da-1.) * pows[0][a] * yz + (db-1.) * pows[1][b] * xz + (dc-1.) * pows[2][c] * xy;
                                
                                db -= 1.0;
                                ++m;
                            }
                            da -= 1.0;
                        }
                        dd += 1.0;
                    }
                }
            }

            int64_t ishell_start = d_nucleus_index(inucl);
            int64_t ishell_end = ishell_start + d_nucleus_shell_num(inucl);

            for (int64_t ishell = ishell_start; ishell < ishell_end; ++ishell) {
                int32_t l = d_shell_ang_mom(ishell);
                if (l > LMAX) continue;

                double s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0, s5 = 0.0;
                int64_t iprim_start = d_shell_prim_idx(ishell);
                int64_t iprim_end = iprim_start + d_shell_prim_num(ishell);

                for (int64_t iprim = iprim_start; iprim < iprim_end; ++iprim) {
                    double alpha = d_exponent(iprim);
                    double ar2_val = alpha * r2;
                    if (ar2_val <= cutoff) {
                        double exp_val = d_coef_normalized(iprim) * exp(-ar2_val);
                        double f = -2.0 * alpha * exp_val;
                        
                        s1 += exp_val;
                        s2 += f * x;
                        s3 += f * y;
                        s4 += f * z;
                        s5 += f * (3.0 - 2.0 * ar2_val);
                    }
                }

                if (s1 == 0.0) continue;

                int64_t ao_offset = d_ao_index(ishell);
                int32_t n_angular = local_lstart[l + 1] - local_lstart[l];
                int32_t p_idx = local_lstart[l];

                for (int il = 0; il < n_angular; ++il) {
                    double p1 = poly_vgl[0][p_idx + il];
                    double p2 = poly_vgl[1][p_idx + il];
                    double p3 = poly_vgl[2][p_idx + il];
                    double p4 = poly_vgl[3][p_idx + il];
                    double p5 = poly_vgl[4][p_idx + il];
                    
                    double fact = d_ao_factor(ao_offset + il);
                    
                    d_ao_vgl(ipoint, 0, ao_offset + il) += p1 * s1 * fact;
                    d_ao_vgl(ipoint, 1, ao_offset + il) += (p2 * s1 + p1 * s2) * fact;
                    d_ao_vgl(ipoint, 2, ao_offset + il) += (p3 * s1 + p1 * s3) * fact;
                    d_ao_vgl(ipoint, 3, ao_offset + il) += (p4 * s1 + p1 * s4) * fact;
                    d_ao_vgl(ipoint, 4, ao_offset + il) += (p5 * s1 + p1 * s5 + 2.0 * (p2 * s2 + p3 * s3 + p4 * s4)) * fact;
                }
            }
        }
    });
}

// ===================================================================
// 3. INITIALIZATION HELPER
// ===================================================================
qmckl_exit_code ensure_basis_initialized(
    KokkosDeviceState& state, const qmckl_context context, const int64_t point_num, const int64_t ao_num, const int64_t shell_num,
    const int64_t nucl_num, const double* coord, const double* nucl_coord, const int64_t* nucleus_index,
    const int64_t* nucleus_shell_num, const double* nucleus_range, const int32_t* shell_ang_mom,
    const int64_t* shell_prim_index, const int64_t* shell_prim_num, const double* exponent,
    const double* coefficient, const double* ao_factor) 
{
    if (state.d_coord.extent(0) != 3 * point_num) {
        state.d_coord = Kokkos::View<double*, DeviceSpace>("Coordinates", 3 * point_num);
    }
    if (state.d_nucl_coord.extent(0) != 3 * nucl_num) {
        state.d_nucl_coord = Kokkos::View<double*, DeviceSpace>("nucl_coord", 3 * nucl_num);
    }

    Kokkos::View<const double*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_coord(coord, 3 * point_num);
    Kokkos::View<const double*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_nucl_coord(nucl_coord, 3 * nucl_num);
    Kokkos::deep_copy(state.d_coord, h_coord);
    Kokkos::deep_copy(state.d_nucl_coord, h_nucl_coord);

    if (!state.basis_initialized) {
        int64_t total_prims = shell_prim_index[shell_num - 1] + shell_prim_num[shell_num - 1];
        std::vector<double> h_coef_norm(total_prims);
        double* prim_factor = (double*) malloc(total_prims * sizeof(double));
        double* shell_factor = (double*) malloc(shell_num * sizeof(double));
        qmckl_get_ao_basis_prim_factor(context, prim_factor, total_prims);
        qmckl_get_ao_basis_shell_factor(context, shell_factor, shell_num);

        for (int64_t ishell = 0; ishell < shell_num; ++ishell) {
            for (int64_t ip = shell_prim_index[ishell]; ip < shell_prim_index[ishell] + shell_prim_num[ishell]; ++ip) {
                h_coef_norm[ip] = coefficient[ip] * prim_factor[ip] * shell_factor[ishell];
            }
        }
        free(prim_factor);
        free(shell_factor);

        Kokkos::View<const int64_t*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_nucl_idx(nucleus_index, nucl_num);
        Kokkos::View<const int64_t*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_nucl_shell_num(nucleus_shell_num, nucl_num);
        Kokkos::View<const double*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_nucl_range(nucleus_range, nucl_num);
        Kokkos::View<const int32_t*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_shell_ang_mom(shell_ang_mom, shell_num);
        Kokkos::View<const int64_t*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_shell_prim_idx(shell_prim_index, shell_num);
        Kokkos::View<const int64_t*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_shell_prim_num(shell_prim_num, shell_num);
        Kokkos::View<const double*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_exponent(exponent, total_prims);
        Kokkos::View<const double*, HostSpace> h_coef_norm_view(h_coef_norm.data(), total_prims);
        Kokkos::View<const double*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_ao_factor(ao_factor, ao_num);

        Kokkos::View<int64_t*, HostSpace> h_ao_index("H_AO_Index", shell_num);
        int32_t lstart_host[32];
        for (int l = 0; l < 32; ++l) lstart_host[l] = l * (l + 1) * (l + 2) / 6;
        
        int64_t k_off = 0;
        for (int inucl = 0; inucl < nucl_num; ++inucl) {
            int64_t s_start = nucleus_index[inucl];
            int64_t s_end = s_start + nucleus_shell_num[inucl];
            for (int64_t ishell = s_start; ishell < s_end; ++ishell) {
                h_ao_index(ishell) = k_off;
                k_off += lstart_host[shell_ang_mom[ishell] + 1] - lstart_host[shell_ang_mom[ishell]];
            }
        }

        state.d_nucleus_index = Kokkos::View<int64_t*, DeviceSpace>("nucl_idx", nucl_num);
        Kokkos::deep_copy(state.d_nucleus_index, h_nucl_idx);
        state.d_nucleus_shell_num = Kokkos::View<int64_t*, DeviceSpace>("nucl_shell_num", nucl_num);
        Kokkos::deep_copy(state.d_nucleus_shell_num, h_nucl_shell_num);
        state.d_nucleus_range = Kokkos::View<double*, DeviceSpace>("nucl_range", nucl_num);
        Kokkos::deep_copy(state.d_nucleus_range, h_nucl_range);
        
        state.d_shell_ang_mom = Kokkos::View<int32_t*, DeviceSpace>("shell_ang_mom", shell_num);
        Kokkos::deep_copy(state.d_shell_ang_mom, h_shell_ang_mom);
        state.d_shell_prim_idx = Kokkos::View<int64_t*, DeviceSpace>("shell_prim_idx", shell_num);
        Kokkos::deep_copy(state.d_shell_prim_idx, h_shell_prim_idx);
        state.d_shell_prim_num = Kokkos::View<int64_t*, DeviceSpace>("shell_prim_num", shell_num);
        Kokkos::deep_copy(state.d_shell_prim_num, h_shell_prim_num);
        state.d_exponent = Kokkos::View<double*, DeviceSpace>("exponent", total_prims);
        Kokkos::deep_copy(state.d_exponent, h_exponent);
        state.d_coef_normalized = Kokkos::View<double*, DeviceSpace>("coef_norm", total_prims);
        Kokkos::deep_copy(state.d_coef_normalized, h_coef_norm_view);
        state.d_ao_factor = Kokkos::View<double*, DeviceSpace>("ao_factor", ao_num);
        Kokkos::deep_copy(state.d_ao_factor, h_ao_factor);
        state.d_ao_index = Kokkos::View<int64_t*, DeviceSpace>("ao_index", shell_num);
        Kokkos::deep_copy(state.d_ao_index, h_ao_index);

        for (int i = 0; i < shell_num; ++i) {
            if (shell_ang_mom[i] > state.lmax_global) state.lmax_global = shell_ang_mom[i];
        }
        state.basis_initialized = true;
    }
    return QMCKL_SUCCESS;
}

// ===================================================================
// 4. HARDWARE DISPATCH ENTRY POINTS (C Linkage)
// ===================================================================
extern "C" {

qmckl_exit_code qmckl_compute_ao_value_kokkos(
    const qmckl_context context, const int64_t ao_num, const int64_t shell_num,
    const int32_t* prim_num_per_nucleus, const int64_t point_num, const int64_t nucl_num,
    const double* coord, const double* nucl_coord, const int64_t* nucleus_index,
    const int64_t* nucleus_shell_num, const double* nucleus_range, const int32_t* shell_ang_mom,
    const int64_t* shell_prim_index, const int64_t* shell_prim_num, const double* exponent,
    const double* coefficient, const double* ao_factor, const double* shell_vgl, 
    double* const ao_value) 
{
    KokkosDeviceState* state_ptr;
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        state_ptr = &device_cache[context];
        ensure_basis_initialized(*state_ptr, context, point_num, ao_num, shell_num, nucl_num, coord, nucl_coord, nucleus_index, nucleus_shell_num, nucleus_range, shell_ang_mom, shell_prim_index, shell_prim_num, exponent, coefficient, ao_factor);

        if (state_ptr->d_ao_value.extent(0) != point_num || state_ptr->d_ao_value.extent(1) != ao_num) {
            state_ptr->d_ao_value = Kokkos::View<double**, Kokkos::LayoutRight, DeviceSpace>("AO_Values", point_num, ao_num);
        }
    }
    
    KokkosDeviceState& state = *state_ptr;

    if (state.lmax_global <= 1) { kokkos_ao_gaussian_kernel<1>(point_num, ao_num, shell_num, nucl_num, state); }
    else if (state.lmax_global == 2) { kokkos_ao_gaussian_kernel<2>(point_num, ao_num, shell_num, nucl_num, state); }
    else if (state.lmax_global == 3) { kokkos_ao_gaussian_kernel<3>(point_num, ao_num, shell_num, nucl_num, state); }
    else if (state.lmax_global == 4) { kokkos_ao_gaussian_kernel<4>(point_num, ao_num, shell_num, nucl_num, state); }
    else if (state.lmax_global == 5) { kokkos_ao_gaussian_kernel<5>(point_num, ao_num, shell_num, nucl_num, state); }
    else { return QMCKL_FAILURE; }
    
    Kokkos::fence();

    if (ao_value != nullptr) {
        Kokkos::View<double**, Kokkos::LayoutRight, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_ao_out(ao_value, point_num, ao_num);
        Kokkos::deep_copy(h_ao_out, state.d_ao_value); 
    }
    return QMCKL_SUCCESS;
}

qmckl_exit_code qmckl_compute_ao_vgl_kokkos(
    const qmckl_context context, const int64_t ao_num, const int64_t shell_num,
    const int32_t* prim_num_per_nucleus, const int64_t point_num, const int64_t nucl_num,
    const double* coord, const double* nucl_coord, const int64_t* nucleus_index,
    const int64_t* nucleus_shell_num, const double* nucleus_range, const int32_t* shell_ang_mom,
    const int64_t* shell_prim_index, const int64_t* shell_prim_num, const double* exponent,
    const double* coefficient, const double* ao_factor, const double* shell_vgl, 
    double* const ao_vgl) 
{
    KokkosDeviceState* state_ptr;
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        state_ptr = &device_cache[context];
        ensure_basis_initialized(*state_ptr, context, point_num, ao_num, shell_num, nucl_num, coord, nucl_coord, nucleus_index, nucleus_shell_num, nucleus_range, shell_ang_mom, shell_prim_index, shell_prim_num, exponent, coefficient, ao_factor);

        if (state_ptr->d_ao_vgl.extent(0) != point_num || state_ptr->d_ao_vgl.extent(1) != 5 || state_ptr->d_ao_vgl.extent(2) != ao_num) {
            state_ptr->d_ao_vgl = Kokkos::View<double***, Kokkos::LayoutRight, DeviceSpace>("AO_VGL", point_num, 5, ao_num);
        }
    }
    
    KokkosDeviceState& state = *state_ptr;

    if (state.lmax_global <= 1) { kokkos_ao_vgl_gaussian_kernel<1>(point_num, ao_num, shell_num, nucl_num, state); }
    else if (state.lmax_global == 2) { kokkos_ao_vgl_gaussian_kernel<2>(point_num, ao_num, shell_num, nucl_num, state); }
    else if (state.lmax_global == 3) { kokkos_ao_vgl_gaussian_kernel<3>(point_num, ao_num, shell_num, nucl_num, state); }
    else if (state.lmax_global == 4) { kokkos_ao_vgl_gaussian_kernel<4>(point_num, ao_num, shell_num, nucl_num, state); }
    else if (state.lmax_global == 5) { kokkos_ao_vgl_gaussian_kernel<5>(point_num, ao_num, shell_num, nucl_num, state); }
    else { return QMCKL_FAILURE; }
    
    Kokkos::fence();

    if (ao_vgl != nullptr) {
        Kokkos::View<double***, Kokkos::LayoutRight, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_ao_vgl_out(ao_vgl, point_num, 5, ao_num);
        Kokkos::deep_copy(h_ao_vgl_out, state.d_ao_vgl);
    }
    return QMCKL_SUCCESS;
}

qmckl_exit_code qmckl_compute_mo_basis_mo_value_kokkos(
    const qmckl_context context, const int64_t ao_num, const int64_t mo_num, 
    const int64_t point_num, const double* coefficient_t, 
    const double* ao_value_host, double* const mo_value) 
{
    KokkosDeviceState* state_ptr;
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        state_ptr = &device_cache[context];
        
        if (state_ptr->d_mo_value.extent(0) != point_num || state_ptr->d_mo_value.extent(1) != mo_num) {
            state_ptr->d_mo_value = Kokkos::View<double**, Kokkos::LayoutRight, DeviceSpace>("MO_Values", point_num, mo_num);
        }

        if (state_ptr->d_mo_coef.extent(0) != ao_num || state_ptr->d_mo_coef.extent(1) != mo_num) {
            state_ptr->d_mo_coef = Kokkos::View<double**, Kokkos::LayoutRight, DeviceSpace>("MO_Coefs", ao_num, mo_num);
            state_ptr->h_mo_coef_mirror = Kokkos::View<double**, Kokkos::LayoutRight, HostSpace>("MO_Coefs_Mirror", ao_num, mo_num);
            state_ptr->mo_initialized = false;
        }

        if (state_ptr->mo_initialized) {
            if (std::memcmp(state_ptr->h_mo_coef_mirror.data(), coefficient_t, ao_num * mo_num * sizeof(double)) != 0) {
                state_ptr->mo_initialized = false;
            }
        }

        if (!state_ptr->mo_initialized) {
            Kokkos::View<const double**, Kokkos::LayoutRight, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> v_mo_coef(coefficient_t, ao_num, mo_num);
            Kokkos::deep_copy(state_ptr->h_mo_coef_mirror, v_mo_coef);
            Kokkos::deep_copy(state_ptr->d_mo_coef, state_ptr->h_mo_coef_mirror); 
            state_ptr->mo_initialized = true;
        }
    }
    
    KokkosDeviceState& state = *state_ptr;

    bool use_device_ao = (state.d_ao_value.extent(0) == point_num && state.d_ao_value.extent(1) == ao_num);
    Kokkos::View<double**, Kokkos::LayoutRight, DeviceSpace> d_ao;
    
    if (use_device_ao) {
        d_ao = state.d_ao_value; 
    } else {
        Kokkos::View<const double**, Kokkos::LayoutRight, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_ao(ao_value_host, point_num, ao_num);
        d_ao = Kokkos::View<double**, Kokkos::LayoutRight, DeviceSpace>("d_ao_temp", point_num, ao_num);
        Kokkos::deep_copy(d_ao, h_ao);
    }

    using MDRange2D = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
    Kokkos::parallel_for("MO_Sparse_Universal", MDRange2D({0, 0}, {point_num, mo_num}), 
    KOKKOS_LAMBDA(const int i_pt, const int i_mo) {
        double sum = 0.0;
        for (int k = 0; k < ao_num; ++k) {
            double ao_val = d_ao(i_pt, k);
            if (ao_val != 0.0) sum += state.d_mo_coef(k, i_mo) * ao_val;
        }
        state.d_mo_value(i_pt, i_mo) = sum;
    });
    
    Kokkos::fence();

    if (mo_value != nullptr) {
        Kokkos::View<double**, Kokkos::LayoutRight, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_mo_out(mo_value, point_num, mo_num);
        Kokkos::deep_copy(h_mo_out, state.d_mo_value); 
    }

    return QMCKL_SUCCESS;
}

qmckl_exit_code qmckl_compute_mo_basis_mo_vgl_kokkos(
    const qmckl_context context, const int64_t ao_num, const int64_t mo_num, 
    const int64_t point_num, const double* coefficient_t, 
    const double* ao_vgl_host, double* const mo_vgl) 
{
    // Sync coefficients
    qmckl_compute_mo_basis_mo_value_kokkos(context, ao_num, mo_num, point_num, coefficient_t, nullptr, nullptr);

    KokkosDeviceState* state_ptr;
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        state_ptr = &device_cache[context];

        if (state_ptr->d_mo_vgl.extent(0) != point_num || state_ptr->d_mo_vgl.extent(1) != 5 || state_ptr->d_mo_vgl.extent(2) != mo_num) {
            state_ptr->d_mo_vgl = Kokkos::View<double***, Kokkos::LayoutRight, DeviceSpace>("MO_VGL", point_num, 5, mo_num);
        }
    }
    KokkosDeviceState& state = *state_ptr;

    bool use_device_ao = (state.d_ao_vgl.extent(0) == point_num && state.d_ao_vgl.extent(1) == 5 && state.d_ao_vgl.extent(2) == ao_num);
    Kokkos::View<double***, Kokkos::LayoutRight, DeviceSpace> d_ao_vgl_local;
    
    if (use_device_ao) {
        d_ao_vgl_local = state.d_ao_vgl; 
    } else {
        Kokkos::View<const double***, Kokkos::LayoutRight, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_ao_vgl(ao_vgl_host, point_num, 5, ao_num);
        d_ao_vgl_local = Kokkos::View<double***, Kokkos::LayoutRight, DeviceSpace>("d_ao_vgl_temp", point_num, 5, ao_num);
        Kokkos::deep_copy(d_ao_vgl_local, h_ao_vgl);
    }

    using MDRange2D = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
    Kokkos::parallel_for("MO_VGL_Universal", MDRange2D({0, 0}, {point_num, mo_num}), 
    KOKKOS_LAMBDA(const int i_pt, const int i_mo) {
        double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0;
        for (int k = 0; k < ao_num; ++k) {
            double c = state.d_mo_coef(k, i_mo);
            if (c != 0.0) {
                sum0 += d_ao_vgl_local(i_pt, 0, k) * c;
                sum1 += d_ao_vgl_local(i_pt, 1, k) * c;
                sum2 += d_ao_vgl_local(i_pt, 2, k) * c;
                sum3 += d_ao_vgl_local(i_pt, 3, k) * c;
                sum4 += d_ao_vgl_local(i_pt, 4, k) * c;
            }
        }
        state.d_mo_vgl(i_pt, 0, i_mo) = sum0;
        state.d_mo_vgl(i_pt, 1, i_mo) = sum1;
        state.d_mo_vgl(i_pt, 2, i_mo) = sum2;
        state.d_mo_vgl(i_pt, 3, i_mo) = sum3;
        state.d_mo_vgl(i_pt, 4, i_mo) = sum4;
    });
    
    Kokkos::fence();

    if (mo_vgl != nullptr) {
        Kokkos::View<double***, Kokkos::LayoutRight, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_mo_vgl_out(mo_vgl, point_num, 5, mo_num);
        Kokkos::deep_copy(h_mo_vgl_out, state.d_mo_vgl); 
    }

    return QMCKL_SUCCESS;
}

void qmckl_kokkos_initialize() {
    if (kokkos_ref_count.fetch_add(1) == 0) {
        if (!Kokkos::is_initialized()) {
            Kokkos::initialize();
            Kokkos::print_configuration(std::cout, true);
        }
    }
}

void qmckl_kokkos_finalize() {
    if (kokkos_ref_count.fetch_sub(1) == 1) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        device_cache.clear(); 
        if (Kokkos::is_initialized()) {
            Kokkos::finalize();
        }
    }
}

} // extern "C"
