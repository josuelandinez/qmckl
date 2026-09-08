#include "qmckl.h"
#include <Kokkos_Core.hpp>
#include <KokkosBlas3_gemm.hpp>
#include <type_traits>

using DeviceSpace = Kokkos::DefaultExecutionSpace::memory_space;
using HostSpace = Kokkos::DefaultHostExecutionSpace::memory_space;

extern "C" {

// ===================================================================
// 1. ATOMIC ORBITALS (Temporarily Disabled - Pending Angular Momentum)
// ===================================================================
qmckl_exit_code qmckl_compute_ao_value_kokkos(
    const qmckl_context context, const int64_t ao_num, const int64_t shell_num,
    const int32_t* prim_num_per_nucleus, const int64_t point_num, const int64_t nucl_num,
    const double* coord, const double* nucl_coord, const int64_t* nucleus_index,
    const int64_t* nucleus_shell_num, const double* nucleus_range, const int32_t* shell_ang_mom,
    const int64_t* shell_prim_index, const int64_t* shell_prim_num, const double* exponent,
    const double* coefficient, const double* ao_factor, const double* shell_vgl, 
    double* const ao_value) 
{
    // Returning FAILURE forces QMCkl to automatically fallback to the native C 
    // CPU implementation, ensuring mathematically correct AOs for the MO step.
    return QMCKL_FAILURE; 
}

// ===================================================================
// 2. MOLECULAR ORBITALS (Hardware Dispatch Engine)
// ===================================================================
qmckl_exit_code qmckl_compute_mo_basis_mo_value_kokkos(
    const qmckl_context context, const int64_t ao_num, const int64_t mo_num, 
    const int64_t point_num, const double* coefficient_t, 
    const double* ao_value_host, double* const mo_value) 
{
    // 1. Map and Upload native AO values to Device
    Kokkos::View<const double**, Kokkos::LayoutRight, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> 
        h_ao(ao_value_host, point_num, ao_num);
    auto d_ao = Kokkos::create_mirror_view_and_copy(DeviceSpace(), h_ao);

    // 2. Map and Upload correctly sized MO coefficients (ao_num x mo_num)
    Kokkos::View<const double**, Kokkos::LayoutRight, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> 
        v_mo_coef(coefficient_t, ao_num, mo_num);
    auto d_mo_coef = Kokkos::create_mirror_view_and_copy(DeviceSpace(), v_mo_coef);

    Kokkos::View<double**, Kokkos::LayoutRight, DeviceSpace> d_mo("MO_Values", point_num, mo_num);

    constexpr bool is_cpu_backend = std::is_same_v<DeviceSpace::memory_space, HostSpace::memory_space>;

    if constexpr (is_cpu_backend) {
        // CPU PATH: OpenMP Optimized Sparse Loop
        using MDRange2D = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
        Kokkos::parallel_for("MO_Sparse_CPU", MDRange2D({0, 0}, {point_num, mo_num}), 
        KOKKOS_LAMBDA(const int i_pt, const int i_mo) {
            double sum = 0.0;
            for (int k = 0; k < ao_num; ++k) {
                double ao_val = d_ao(i_pt, k);
                // Corrected indexing: (k, i_mo)
                if (ao_val != 0.0) sum += d_mo_coef(k, i_mo) * ao_val;
            }
            d_mo(i_pt, i_mo) = sum;
        });
        Kokkos::fence();
    } else {
        // GPU PATH: Tensor Core Dense BLAS ( C = A * B )
        KokkosBlas::gemm("N", "N", 1.0, d_ao, d_mo_coef, 0.0, d_mo);
        Kokkos::fence();
    }

    Kokkos::View<double**, Kokkos::LayoutRight, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> 
        h_mo_out(mo_value, point_num, mo_num);
    Kokkos::deep_copy(h_mo_out, d_mo);

    return QMCKL_SUCCESS;
}

// ===================================================================
// 3. KOKKOS LIFECYCLE MANAGEMENT
// ===================================================================
void qmckl_kokkos_initialize() {
    if (!Kokkos::is_initialized()) Kokkos::initialize();
    // Kokkos prints the active hardware spaces to standard output
    Kokkos::print_configuration(std::cout, true);
}

void qmckl_kokkos_finalize() {
    if (Kokkos::is_initialized()) Kokkos::finalize();
}
 
} // extern "C"
