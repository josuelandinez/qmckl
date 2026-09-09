#include "qmckl.h"
#include <Kokkos_Core.hpp>
#include <KokkosBlas3_gemm.hpp>
#include <iostream>
#include <cmath>

using DeviceSpace = Kokkos::DefaultExecutionSpace::memory_space;
using HostSpace = Kokkos::DefaultHostExecutionSpace::memory_space;

// ===================================================================
// 1. TEMPLATED ATOMIC ORBITAL KERNEL (Register Allocated)
// ===================================================================
template <int LMAX>
void kokkos_ao_gaussian_kernel(
    const int64_t point_num, const int64_t ao_num, const int64_t shell_num, const int64_t nucl_num,
    Kokkos::View<const double*, DeviceSpace> d_coord,
    Kokkos::View<const double*, DeviceSpace> d_nucl_coord,
    Kokkos::View<const int64_t*, DeviceSpace> d_nucleus_index,
    Kokkos::View<const int64_t*, DeviceSpace> d_nucleus_shell_num,
    Kokkos::View<const double*, DeviceSpace> d_nucleus_range,
    Kokkos::View<const int32_t*, DeviceSpace> d_shell_ang_mom,
    Kokkos::View<const int64_t*, DeviceSpace> d_shell_prim_idx,
    Kokkos::View<const int64_t*, DeviceSpace> d_shell_prim_num,
    Kokkos::View<const double*, DeviceSpace> d_exponent,
    Kokkos::View<const double*, DeviceSpace> d_coef_normalized,
    Kokkos::View<const double*, DeviceSpace> d_ao_factor,
    Kokkos::View<const int64_t*, DeviceSpace> d_ao_index,
    Kokkos::View<double**, Kokkos::LayoutRight, DeviceSpace> d_ao_value)
{
    constexpr int N_POLY = (LMAX + 1) * (LMAX + 2) * (LMAX + 3) / 6;

    Kokkos::parallel_for("AO_Gaussian_Templated", point_num, KOKKOS_LAMBDA(const int ipoint) {
        double cutoff = 27.631021115928547;
        int32_t local_lstart[32];
        for (int l = 0; l < 32; ++l) local_lstart[l] = l * (l + 1) * (l + 2) / 6;

        for (int k = 0; k < ao_num; ++k) {
            d_ao_value(ipoint, k) = 0.0;
        }

        // Correct QMCkl SOA coordinate layout
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
                poly[1] = x; poly[2] = y; poly[3] = z;
            }
            if constexpr (LMAX >= 2) {
                poly[4] = x*x; poly[5] = x*y; poly[6] = x*z;
                poly[7] = y*y; poly[8] = y*z; poly[9] = z*z;
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
                    d_ao_value(ipoint, ao_offset + il) = poly[p_idx + il] * radial_sum * d_ao_factor(ao_offset + il);
                }
            }
        }
    });
}

// ===================================================================
// 2. HARDWARE DISPATCH ENTRY POINTS (C Linkage)
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
    // Retrieve normalized coefficients from context directly through getter API
    int64_t total_prims = shell_prim_index[shell_num - 1] + shell_prim_num[shell_num - 1];
    std::vector<double> h_coef_norm(total_prims);
    
    // Compute normalized coefficients on host exactly as finalized by QMCkl
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

    // Unmanaged host views
    Kokkos::View<const double*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_coord(coord, 3 * point_num);
    Kokkos::View<const double*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_nucl_coord(nucl_coord, 3 * nucl_num);
    Kokkos::View<const int64_t*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_nucl_idx(nucleus_index, nucl_num);
    Kokkos::View<const int64_t*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_nucl_shell_num(nucleus_shell_num, nucl_num);
    Kokkos::View<const double*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_nucl_range(nucleus_range, nucl_num);
    Kokkos::View<const int32_t*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_shell_ang_mom(shell_ang_mom, shell_num);
    Kokkos::View<const int64_t*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_shell_prim_idx(shell_prim_index, shell_num);
    Kokkos::View<const int64_t*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_shell_prim_num(shell_prim_num, shell_num);
    Kokkos::View<const double*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_exponent(exponent, total_prims);
    Kokkos::View<const double*, HostSpace> h_coef_norm_view(h_coef_norm.data(), total_prims);
    Kokkos::View<const double*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_ao_factor(ao_factor, ao_num);

    // Compute AO offsets
    Kokkos::View<int64_t*, HostSpace> h_ao_index("H_AO_Index", shell_num);
    int32_t lstart[32];
    for (int l = 0; l < 32; ++l) lstart[l] = l * (l + 1) * (l + 2) / 6;
    int64_t k_off = 0;
    for (int ishell = 0; ishell < shell_num; ++ishell) {
        h_ao_index(ishell) = k_off;
        k_off += lstart[shell_ang_mom[ishell] + 1] - lstart[shell_ang_mom[ishell]];
    }

    // Mirror to device
    auto d_coord = Kokkos::create_mirror_view_and_copy(DeviceSpace(), h_coord);
    auto d_nucl_coord = Kokkos::create_mirror_view_and_copy(DeviceSpace(), h_nucl_coord);
    auto d_nucl_idx = Kokkos::create_mirror_view_and_copy(DeviceSpace(), h_nucl_idx);
    auto d_nucl_shell_num = Kokkos::create_mirror_view_and_copy(DeviceSpace(), h_nucl_shell_num);
    auto d_nucl_range = Kokkos::create_mirror_view_and_copy(DeviceSpace(), h_nucl_range);
    auto d_shell_ang_mom = Kokkos::create_mirror_view_and_copy(DeviceSpace(), h_shell_ang_mom);
    auto d_shell_prim_idx = Kokkos::create_mirror_view_and_copy(DeviceSpace(), h_shell_prim_idx);
    auto d_shell_prim_num = Kokkos::create_mirror_view_and_copy(DeviceSpace(), h_shell_prim_num);
    auto d_exponent = Kokkos::create_mirror_view_and_copy(DeviceSpace(), h_exponent);
    auto d_coef_normalized = Kokkos::create_mirror_view_and_copy(DeviceSpace(), h_coef_norm_view);
    auto d_ao_factor = Kokkos::create_mirror_view_and_copy(DeviceSpace(), h_ao_factor);
    auto d_ao_index = Kokkos::create_mirror_view_and_copy(DeviceSpace(), h_ao_index);

    Kokkos::View<double**, Kokkos::LayoutRight, DeviceSpace> d_ao_value("AO_Values", point_num, ao_num);

    int32_t lmax_global = 0;
    for (int i = 0; i < shell_num; ++i) {
        if (shell_ang_mom[i] > lmax_global) lmax_global = shell_ang_mom[i];
    }

    if (lmax_global <= 1) {
        kokkos_ao_gaussian_kernel<1>(point_num, ao_num, shell_num, nucl_num, d_coord, d_nucl_coord, d_nucl_idx, d_nucl_shell_num, d_nucl_range, d_shell_ang_mom, d_shell_prim_idx, d_shell_prim_num, d_exponent, d_coef_normalized, d_ao_factor, d_ao_index, d_ao_value);
    } else {
        kokkos_ao_gaussian_kernel<2>(point_num, ao_num, shell_num, nucl_num, d_coord, d_nucl_coord, d_nucl_idx, d_nucl_shell_num, d_nucl_range, d_shell_ang_mom, d_shell_prim_idx, d_shell_prim_num, d_exponent, d_coef_normalized, d_ao_factor, d_ao_index, d_ao_value);
    }
    Kokkos::fence();

    Kokkos::View<double**, Kokkos::LayoutRight, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_ao_out(ao_value, point_num, ao_num);
    Kokkos::deep_copy(h_ao_out, d_ao_value);

    return QMCKL_SUCCESS;
}

qmckl_exit_code qmckl_compute_mo_basis_mo_value_kokkos(
    const qmckl_context context, const int64_t ao_num, const int64_t mo_num, 
    const int64_t point_num, const double* coefficient_t, 
    const double* ao_value_host, double* const mo_value) 
{
    Kokkos::View<const double**, Kokkos::LayoutRight, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_ao(ao_value_host, point_num, ao_num);
    auto d_ao = Kokkos::create_mirror_view_and_copy(DeviceSpace(), h_ao);

    Kokkos::View<const double**, Kokkos::LayoutRight, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> v_mo_coef(coefficient_t, ao_num, mo_num);
    auto d_mo_coef = Kokkos::create_mirror_view_and_copy(DeviceSpace(), v_mo_coef);

    Kokkos::View<double**, Kokkos::LayoutRight, DeviceSpace> d_mo("MO_Values", point_num, mo_num);

    constexpr bool is_cpu_backend = std::is_same_v<DeviceSpace::memory_space, HostSpace::memory_space>;

    if constexpr (is_cpu_backend) {
        using MDRange2D = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
        Kokkos::parallel_for("MO_Sparse_CPU", MDRange2D({0, 0}, {point_num, mo_num}), 
        KOKKOS_LAMBDA(const int i_pt, const int i_mo) {
            double sum = 0.0;
            for (int k = 0; k < ao_num; ++k) {
                double ao_val = d_ao(i_pt, k);
                if (ao_val != 0.0) sum += d_mo_coef(k, i_mo) * ao_val;
            }
            d_mo(i_pt, i_mo) = sum;
        });
        Kokkos::fence();
    } else {
        KokkosBlas::gemm("N", "N", 1.0, d_ao, d_mo_coef, 0.0, d_mo);
        Kokkos::fence();
    }

    Kokkos::View<double**, Kokkos::LayoutRight, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_mo_out(mo_value, point_num, mo_num);
    Kokkos::deep_copy(h_mo_out, d_mo);

    return QMCKL_SUCCESS;
}

void qmckl_kokkos_initialize() {
    if (!Kokkos::is_initialized()) Kokkos::initialize();
    Kokkos::print_configuration(std::cout, true);
}

void qmckl_kokkos_finalize() {
    if (Kokkos::is_initialized()) Kokkos::finalize();
}

} // extern "C"
