/*
 *  Copyright 2006 The WebRTC Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef RTC_BASE_CHECKS_H_
#define RTC_BASE_CHECKS_H_

// If you for some reson need to know if DCHECKs are on, test the value of
// RTC_DCHECK_IS_ON. (Test its value, not if it's defined; it'll always be
// defined, to either a true or a false value.)
#if !defined(NDEBUG) || defined(DCHECK_ALWAYS_ON)
#define RTC_DCHECK_IS_ON 1
#else
#define RTC_DCHECK_IS_ON 0
#endif

// Annotate a function that will not return control flow to the caller.
#if defined(_MSC_VER)
#define RTC_NORETURN __declspec(noreturn)
#elif defined(__GNUC__)
#define RTC_NORETURN __attribute__ ((__noreturn__))
#else
#define RTC_NORETURN
#endif

#ifdef __cplusplus
extern "C" {
#endif
RTC_NORETURN void rtc_FatalMessage(const char* file, int line, const char* msg);
#ifdef __cplusplus
}  // extern "C"
#endif

#ifdef __cplusplus
// C++ version.

#include <string>

#include "webrtc/rtc_base/numerics/safe_compare.h"
#include "webrtc/rtc_base/system/inline.h"

// The macros here print a message to stderr and abort under various
// conditions. All will accept additional stream messages. For example:
// RTC_DCHECK_EQ(foo, bar) << "I'm printed when foo != bar.";
//
// - RTC_CHECK(x) is an assertion that x is always true, and that if it isn't,
//   it's better to terminate the process than to continue. During development,
//   the reason that it's better to terminate might simply be that the error
//   handling code isn't in place yet; in production, the reason might be that
//   the author of the code truly believes that x will always be true, but that
//   she recognizes that if she is wrong, abrupt and unpleasant process
//   termination is still better than carrying on with the assumption violated.
//
//   RTC_CHECK always evaluates its argument, so it's OK for x to have side
//   effects.
//
// - RTC_DCHECK(x) is the same as RTC_CHECK(x)---an assertion that x is always
//   true---except that x will only be evaluated in debug builds; in production
//   builds, x is simply assumed to be true. This is useful if evaluating x is
//   expensive and the expected cost of failing to detect the violated
//   assumption is acceptable. You should not handle cases where a production
//   build fails to spot a violated condition, even those that would result in
//   crashes. If the code needs to cope with the error, make it cope, but don't
//   call RTC_DCHECK; if the condition really can't occur, but you'd sleep
//   better at night knowing that the process will suicide instead of carrying
//   on in case you were wrong, use RTC_CHECK instead of RTC_DCHECK.
//
//   RTC_DCHECK only evaluates its argument in debug builds, so if x has visible
//   side effects, you need to write e.g.
//     bool w = x; RTC_DCHECK(w);
//
// - RTC_CHECK_EQ, _NE, _GT, ..., and RTC_DCHECK_EQ, _NE, _GT, ... are
//   specialized variants of RTC_CHECK and RTC_DCHECK that print prettier
//   messages if the condition doesn't hold. Prefer them to raw RTC_CHECK and
//   RTC_DCHECK.
//
// - FATAL() aborts unconditionally.
//
// TODO(ajm): Ideally, checks.h would be combined with logging.h, but
// consolidation with system_wrappers/logging.h should happen first.

namespace rtc {
namespace webrtc_checks_impl {
enum class CheckArgType : int8_t {
  kEnd = 0,
  kInt,
  kLong,
  kLongLong,
  kUInt,
  kULong,
  kULongLong,
  kDouble,
  kLongDouble,
  kCharP,
  kStdString,
  kVoidP,

  // kCheckOp doesn't represent an argument type. Instead, it is sent as the
  // first argument from RTC_CHECK_OP to make FatalLog use the next two
  // arguments to build the special CHECK_OP error message
  // (the "a == b (1 vs. 2)" bit).
  kCheckOp,
};

RTC_NORETURN void FatalLog(const char* file,
                           int line,
                           const char* message,
                           const CheckArgType* fmt,
                           ...);

// Wrapper for log arguments. Only ever make values of this type with the
// MakeVal() functions.
template <CheckArgType N, typename T>
struct Val {
  static constexpr CheckArgType Type() { return N; }
  T GetVal() const { return val; }
  T val;
};

inline Val<CheckArgType::kInt, int> MakeVal(int x) {
  return {x};
}
inline Val<CheckArgType::kLong, long> MakeVal(long x) {
  return {x};
}
inline Val<CheckArgType::kLongLong, long long> MakeVal(long long x) {
  return {x};
}
inline Val<CheckArgType::kUInt, unsigned int> MakeVal(unsigned int x) {
  return {x};
}
inline Val<CheckArgType::kULong, unsigned long> MakeVal(unsigned long x) {
  return {x};
}
inline Val<CheckArgType::kULongLong, unsigned long long> MakeVal(
    unsigned long long x) {
  return {x};
}

inline Val<CheckArgType::kDouble, double> MakeVal(double x) {
  return {x};
}
inline Val<CheckArgType::kLongDouble, long double> MakeVal(long double x) {
  return {x};
}

inline Val<CheckArgType::kCharP, const char*> MakeVal(const char* x) {
  return {x};
}
inline Val<CheckArgType::kStdString, const std::string*> MakeVal(
    const std::string& x) {
  return {&x};
}

inline Val<CheckArgType::kVoidP, const void*> MakeVal(const void* x) {
  return {x};
}

// Ephemeral type that represents the result of the logging << operator.
template <typename... Ts>
class LogStreamer;

// Base case: Before the first << argument.
template <>
class LogStreamer<> final {
 public:
  template <
      typename U,
      typename std::enable_if<std::is_arithmetic<U>::value>::type* = nullptr>
  RTC_FORCE_INLINE LogStreamer<decltype(MakeVal(std::declval<U>()))> operator<<(
      U arg) const {
    return LogStreamer<decltype(MakeVal(std::declval<U>()))>(MakeVal(arg),
                                                             this);
  }

  template <
      typename U,
      typename std::enable_if<!std::is_arithmetic<U>::value>::type* = nullptr>
  RTC_FORCE_INLINE LogStreamer<decltype(MakeVal(std::declval<U>()))> operator<<(
      const U& arg) const {
    return LogStreamer<decltype(MakeVal(std::declval<U>()))>(MakeVal(arg),
                                                             this);
  }

  template <typename... Us>
  RTC_NORETURN RTC_FORCE_INLINE static void Call(const char* file,
                                                 const int line,
                                                 const char* message,
                                                 const Us&... args) {
    static constexpr CheckArgType t[] = {Us::Type()..., CheckArgType::kEnd};
    FatalLog(file, line, message, t, args.GetVal()...);
  }

  template <typename... Us>
  RTC_NORETURN RTC_FORCE_INLINE static void CallCheckOp(const char* file,
                                                        const int line,
                                                        const char* message,
                                                        const Us&... args) {
    static constexpr CheckArgType t[] = {CheckArgType::kCheckOp, Us::Type()...,
                                         CheckArgType::kEnd};
    FatalLog(file, line, message, t, args.GetVal()...);
  }
};

// Inductive case: We've already seen at least one << argument. The most recent
// one had type `T`, and the earlier ones had types `Ts`.
template <typename T, typename... Ts>
class LogStreamer<T, Ts...> final {
 public:
  RTC_FORCE_INLINE LogStreamer(T arg, const LogStreamer<Ts...>* prior)
      : arg_(arg), prior_(prior) {}

  template <
      typename U,
      typename std::enable_if<std::is_arithmetic<U>::value>::type* = nullptr>
  RTC_FORCE_INLINE LogStreamer<decltype(MakeVal(std::declval<U>())), T, Ts...>
  operator<<(U arg) const {
    return LogStreamer<decltype(MakeVal(std::declval<U>())), T, Ts...>(
        MakeVal(arg), this);
  }

  template <
      typename U,
      typename std::enable_if<!std::is_arithmetic<U>::value>::type* = nullptr>
  RTC_FORCE_INLINE LogStreamer<decltype(MakeVal(std::declval<U>())), T, Ts...>
  operator<<(const U& arg) const {
    return LogStreamer<decltype(MakeVal(std::declval<U>())), T, Ts...>(
        MakeVal(arg), this);
  }

  template <typename... Us>
  RTC_NORETURN RTC_FORCE_INLINE void Call(const char* file,
                                          const int line,
                                          const char* message,
                                          const Us&... args) const {
    prior_->Call(file, line, message, arg_, args...);
  }

  template <typename... Us>
  RTC_NORETURN RTC_FORCE_INLINE void CallCheckOp(const char* file,
                                                 const int line,
                                                 const char* message,
                                                 const Us&... args) const {
    prior_->CallCheckOp(file, line, message, arg_, args...);
  }

 private:
  // The most recent argument.
  T arg_;

  // Earlier arguments.
  const LogStreamer<Ts...>* prior_;
};

template <bool isCheckOp>
class FatalLogCall final {
 public:
  FatalLogCall(const char* file, int line, const char* message)
      : file_(file), line_(line), message_(message) {}

  // This can be any binary operator with precedence lower than <<.
  template <typename... Ts>
  RTC_NORETURN RTC_FORCE_INLINE void operator&(
      const LogStreamer<Ts...>& streamer) {
    isCheckOp ? streamer.CallCheckOp(file_, line_, message_)
              : streamer.Call(file_, line_, message_);
  }

 private:
  const char* file_;
  int line_;
  const char* message_;
};
}  // namespace webrtc_checks_impl

// The actual stream used isn't important. We reference |ignored| in the code
// but don't evaluate it; this is to avoid "unused variable" warnings (we do so
// in a particularly convoluted way with an extra ?: because that appears to be
// the simplest construct that keeps Visual Studio from complaining about
// condition being unused).
#define RTC_EAT_STREAM_PARAMETERS(ignored)                        \
  (true ? true : ((void)(ignored), true))                         \
      ? static_cast<void>(0)                                      \
      : rtc::webrtc_checks_impl::FatalLogCall<false>("", 0, "") & \
            rtc::webrtc_checks_impl::LogStreamer<>()

// Call RTC_EAT_STREAM_PARAMETERS with an argument that fails to compile if
// values of the same types as |a| and |b| can't be compared with the given
// operation, and that would evaluate |a| and |b| if evaluated.
#define RTC_EAT_STREAM_PARAMETERS_OP(op, a, b) \
  RTC_EAT_STREAM_PARAMETERS(((void)rtc::Safe##op(a, b)))

// RTC_CHECK dies with a fatal error if condition is not true. It is *not*
// controlled by NDEBUG or anything else, so the check will be executed
// regardless of compilation mode.
//
// We make sure RTC_CHECK et al. always evaluates |condition|, as
// doing RTC_CHECK(FunctionWithSideEffect()) is a common idiom.
#define RTC_CHECK(condition)                                       \
  while (!(condition))                                             \
  rtc::webrtc_checks_impl::FatalLogCall<false>(__FILE__, __LINE__, \
                                               #condition) &       \
      rtc::webrtc_checks_impl::LogStreamer<>()

// Helper macro for binary operators.
// Don't use this macro directly in your code, use RTC_CHECK_EQ et al below.
#define RTC_CHECK_OP(name, op, val1, val2)                               \
  while (!rtc::Safe##name((val1), (val2)))                               \
  rtc::webrtc_checks_impl::FatalLogCall<true>(__FILE__, __LINE__,        \
                                              #val1 " " #op " " #val2) & \
      rtc::webrtc_checks_impl::LogStreamer<>() << (val1) << (val2)

#define RTC_CHECK_EQ(val1, val2) RTC_CHECK_OP(Eq, ==, val1, val2)
#define RTC_CHECK_NE(val1, val2) RTC_CHECK_OP(Ne, !=, val1, val2)
#define RTC_CHECK_LE(val1, val2) RTC_CHECK_OP(Le, <=, val1, val2)
#define RTC_CHECK_LT(val1, val2) RTC_CHECK_OP(Lt, <, val1, val2)
#define RTC_CHECK_GE(val1, val2) RTC_CHECK_OP(Ge, >=, val1, val2)
#define RTC_CHECK_GT(val1, val2) RTC_CHECK_OP(Gt, >, val1, val2)

// The RTC_DCHECK macro is equivalent to RTC_CHECK except that it only generates
// code in debug builds. It does reference the condition parameter in all cases,
// though, so callers won't risk getting warnings about unused variables.
#if RTC_DCHECK_IS_ON
#define RTC_DCHECK(condition) RTC_CHECK(condition)
#define RTC_DCHECK_EQ(v1, v2) RTC_CHECK_EQ(v1, v2)
#define RTC_DCHECK_NE(v1, v2) RTC_CHECK_NE(v1, v2)
#define RTC_DCHECK_LE(v1, v2) RTC_CHECK_LE(v1, v2)
#define RTC_DCHECK_LT(v1, v2) RTC_CHECK_LT(v1, v2)
#define RTC_DCHECK_GE(v1, v2) RTC_CHECK_GE(v1, v2)
#define RTC_DCHECK_GT(v1, v2) RTC_CHECK_GT(v1, v2)
#else
#define RTC_DCHECK(condition) RTC_EAT_STREAM_PARAMETERS(condition)
#define RTC_DCHECK_EQ(v1, v2) RTC_EAT_STREAM_PARAMETERS_OP(Eq, v1, v2)
#define RTC_DCHECK_NE(v1, v2) RTC_EAT_STREAM_PARAMETERS_OP(Ne, v1, v2)
#define RTC_DCHECK_LE(v1, v2) RTC_EAT_STREAM_PARAMETERS_OP(Le, v1, v2)
#define RTC_DCHECK_LT(v1, v2) RTC_EAT_STREAM_PARAMETERS_OP(Lt, v1, v2)
#define RTC_DCHECK_GE(v1, v2) RTC_EAT_STREAM_PARAMETERS_OP(Ge, v1, v2)
#define RTC_DCHECK_GT(v1, v2) RTC_EAT_STREAM_PARAMETERS_OP(Gt, v1, v2)
#endif

#define RTC_UNREACHABLE_CODE_HIT false
#define RTC_NOTREACHED() RTC_DCHECK(RTC_UNREACHABLE_CODE_HIT)

// TODO(bugs.webrtc.org/8454): Add an RTC_ prefix or rename differently.
#define FATAL()                                                    \
  rtc::webrtc_checks_impl::FatalLogCall<false>(__FILE__, __LINE__, \
                                               "FATAL()") &        \
      rtc::webrtc_checks_impl::LogStreamer<>()

// Performs the integer division a/b and returns the result. CHECKs that the
// remainder is zero.
template <typename T>
inline T CheckedDivExact(T a, T b) {
  RTC_CHECK_EQ(a % b, 0) << a << " is not evenly divisible by " << b;
  return a / b;
}

}  // namespace rtc

#else  // __cplusplus not defined
// C version. Lacks many features compared to the C++ version, but usage
// guidelines are the same.

#define RTC_CHECK(condition)                                             \
  do {                                                                   \
    if (!(condition)) {                                                  \
      rtc_FatalMessage(__FILE__, __LINE__, "CHECK failed: " #condition); \
    }                                                                    \
  } while (0)

#define RTC_CHECK_EQ(a, b) RTC_CHECK((a) == (b))
#define RTC_CHECK_NE(a, b) RTC_CHECK((a) != (b))
#define RTC_CHECK_LE(a, b) RTC_CHECK((a) <= (b))
#define RTC_CHECK_LT(a, b) RTC_CHECK((a) < (b))
#define RTC_CHECK_GE(a, b) RTC_CHECK((a) >= (b))
#define RTC_CHECK_GT(a, b) RTC_CHECK((a) > (b))

#define RTC_DCHECK(condition)                                             \
  do {                                                                    \
    if (RTC_DCHECK_IS_ON && !(condition)) {                               \
      rtc_FatalMessage(__FILE__, __LINE__, "DCHECK failed: " #condition); \
    }                                                                     \
  } while (0)

#define RTC_DCHECK_EQ(a, b) RTC_DCHECK((a) == (b))
#define RTC_DCHECK_NE(a, b) RTC_DCHECK((a) != (b))
#define RTC_DCHECK_LE(a, b) RTC_DCHECK((a) <= (b))
#define RTC_DCHECK_LT(a, b) RTC_DCHECK((a) < (b))
#define RTC_DCHECK_GE(a, b) RTC_DCHECK((a) >= (b))
#define RTC_DCHECK_GT(a, b) RTC_DCHECK((a) > (b))

#endif  // __cplusplus

#endif  // RTC_BASE_CHECKS_H_
