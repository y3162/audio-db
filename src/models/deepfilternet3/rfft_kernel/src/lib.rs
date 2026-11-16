//! Float32 real-to-complex FFT used by DeepFilterNet's libdf 0.5.6 (`realfft` 3.3.0).
//!
//! The plan is cached. The transform is unnormalized, matching `DFState::analysis`
//! before that function applies `wnorm`.

use std::os::raw::{c_float, c_int};
use std::sync::{Mutex, OnceLock};

use realfft::RealFftPlanner;

/// C ABI version. Python refuses to load a library that returns anything else.
pub const ABI_VERSION: c_int = 1;

#[no_mangle]
pub extern "C" fn df_rfft_abi() -> c_int {
    ABI_VERSION
}

fn planner() -> &'static Mutex<RealFftPlanner<f32>> {
    static PLANNER: OnceLock<Mutex<RealFftPlanner<f32>>> = OnceLock::new();
    PLANNER.get_or_init(|| Mutex::new(RealFftPlanner::<f32>::new()))
}

#[no_mangle]
pub extern "C" fn df_rfft_f32(
    input: *const c_float,
    n: usize,
    out_re: *mut c_float,
    out_im: *mut c_float,
) -> c_int {
    if input.is_null() || out_re.is_null() || out_im.is_null() || n < 2 {
        return 1;
    }
    let mut indata = unsafe { std::slice::from_raw_parts(input, n) }.to_vec();
    let mut guard = match planner().lock() {
        Ok(guard) => guard,
        Err(_) => return 2,
    };
    let fft = guard.plan_fft_forward(n);
    let mut output = fft.make_output_vec();
    if fft.process(&mut indata, &mut output).is_err() {
        return 3;
    }
    if output.len() != n / 2 + 1 {
        return 4;
    }
    unsafe {
        for (index, value) in output.iter().enumerate() {
            *out_re.add(index) = value.re;
            *out_im.add(index) = value.im;
        }
    }
    0
}

/// `(value + 1e-10).log10() * 10`, the ERB decibel conversion in libdf.
#[no_mangle]
pub extern "C" fn df_log10_db(input: *const c_float, n: usize, output: *mut c_float) -> c_int {
    if input.is_null() || output.is_null() {
        return 1;
    }
    let values = unsafe { std::slice::from_raw_parts(input, n) };
    let out = unsafe { std::slice::from_raw_parts_mut(output, n) };
    for (value, dest) in values.iter().zip(out.iter_mut()) {
        *dest = (*value + 1e-10).log10() * 10.;
    }
    0
}

/// In-place unit norm. State is `0.001 + i * (0.0001 - 0.001) / (bins - 1)` per channel.
///
/// This is the libdf `band_unit_norm` recurrence. A PyTorch vectorized form differs
/// by one ulp, which is enough to flip near-zero AdamW gradients.
#[no_mangle]
pub extern "C" fn df_unit_norm(
    real: *mut c_float,
    imag: *mut c_float,
    channels: usize,
    frames: usize,
    bins: usize,
    alpha: c_float,
) -> c_int {
    if real.is_null() || imag.is_null() || channels == 0 || bins < 2 {
        return 1;
    }
    let real = unsafe { std::slice::from_raw_parts_mut(real, channels * frames * bins) };
    let imag = unsafe { std::slice::from_raw_parts_mut(imag, channels * frames * bins) };
    let step = (0.0001_f32 - 0.001_f32) / (bins - 1) as f32;
    let one_minus = 1.0_f32 - alpha;
    for channel in 0..channels {
        let mut state = vec![0.0_f32; bins];
        for (index, value) in state.iter_mut().enumerate() {
            *value = 0.001 + index as f32 * step;
        }
        for frame in 0..frames {
            for bin in 0..bins {
                let offset = (channel * frames + frame) * bins + bin;
                let re = real[offset];
                let im = imag[offset];
                let magnitude = re.hypot(im);
                let updated = magnitude * one_minus + state[bin] * alpha;
                state[bin] = updated;
                let scale = updated.sqrt();
                real[offset] = re / scale;
                imag[offset] = im / scale;
            }
        }
    }
    0
}

/// `f32::sqrt` for the unit-norm divisor. `libm` sqrt matches libdf; `torch.sqrt` does not.
#[no_mangle]
pub extern "C" fn df_sqrt_f32(input: *const c_float, n: usize, output: *mut c_float) -> c_int {
    if input.is_null() || output.is_null() {
        return 1;
    }
    let values = unsafe { std::slice::from_raw_parts(input, n) };
    let out = unsafe { std::slice::from_raw_parts_mut(output, n) };
    for (value, dest) in values.iter().zip(out.iter_mut()) {
        *dest = value.sqrt();
    }
    0
}
