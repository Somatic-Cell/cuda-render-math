#pragma once 

#include <cuda_render_math/config.hpp>
#include <cuda_render_math/scalar.hpp>

#include <limits>

namespace cuda_render_math {

static_assert(std::numeric_limits<float>::radix == 2 && std::numeric_limits<float>::digits == 24);
static_assert(std::numeric_limits<double>::radix == 2 && std::numeric_limits<double>::digits == 53);

namespace detail {

template<supported_arithmetic T>
inline constexpr T cast_max = (std::numeric_limits<T>::max)();

template<real_scalar T>
inline constexpr T cast_infinity = std::numeric_limits<T>::infinity();

template<real_scalar T>
inline constexpr T cast_nan = std::numeric_limits<T>::quiet_NaN();

} // namespace detail

// 成功時だけ output を更新する
// 実数 -> 整数: 0 方向への切り捨ての値域を検査
// 実数 -> 実数: NaN / Inf を許容する
template<supported_arithmetic To, supported_arithmetic From>
[[nodiscard]] CUDA_RENDER_MATH_HD constexpr bool
try_scalar_cast(From value, To& output) noexcept {
    
    if constexpr (!std::same_as<To, From>) {
    // 整数 -> 整数
        if constexpr (integer_scalar<To> && integer_scalar<From>) {
            if constexpr (std::same_as<From, std::int32_t>) {
                // この分岐にたどり着いている時点で，int -> uint 
                // uint なので value が負なら false
                if (value < From(0)) return false;
            // From が uint32_t 
            } else {
                // To は uint か int なので，格納できる最大値を超えていたら false 
                if (value > static_cast<From>(detail::cast_max<To>)) return false;
            }
        // 実数 -> 整数
        } else if constexpr (integer_scalar<To> && real_scalar<From>) {
            // INT32_MAX を float に変換して上限にすると 2^31 に丸まり
            // 範囲外を受理するので，上界を阪堺区間で比較する
            if constexpr (std::same_as<To, std::int32_t>) {
                const From lower = From(-2147483648.0);
                const From upper = From(2147483648.0);
                if constexpr (std::same_as<From, float>) {
                    if (!(value >= lower && value < upper)) return false;
                } else {
                    if (!(value > lower - From(1) && value < upper)) return false;
                }
            } else {
                if (!(value > From(-1) && value < From(4294967296.0))) return false;
            }
        // 実数 -> 実数
        } else if constexpr (real_scalar<To> && real_scalar<From>){
            // NaN が入っていたら NaN を入れて返す
            if (value != value) { // NaN 同士は比較できないので false になる (IEEE 754)
                output = detail::cast_nan<To>;
                return true;
            }
            
            // Inf が入っていたら Inf を返す
            if (value == detail::cast_infinity<From>) {
                output = detail::cast_infinity<To>;
                return true;
            }
            
            // -Inf が入っていたら -Inf を返す
            if (value == -detail::cast_infinity<From>) {
                output = -detail::cast_infinity<To>;
                return true;
            }
            
            // To が float の場合，from が float の表現できる範囲を超えていないかを判定
            if constexpr (std::same_as<To, float>) {
                const From limit = static_cast<From>(detail::cast_max<To>);
                if(value < -limit || value > limit) return false;
            }
        }
    // 整数 -> 実数
    // int32 / uint32 -> float / double は必ず有限の値域に収まるので考えなくて OK
    }
    output = static_cast<To>(value);
    return true;
}

} // namespace cuda_render_math