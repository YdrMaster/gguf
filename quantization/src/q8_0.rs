use super::{QuantBlock, _32};
use half::f16;

#[repr(C)]
pub struct Q8_0 {
    delta: f16,
    quants: [i8; _32],
}

impl QuantBlock<f32, _32> for Q8_0 {
    fn quantize(data: &[f32; _32]) -> Self {
        let amax = data.iter().fold(0., |acc, x| x.abs().max(acc));
        let delta = amax / i8::MAX as f32;
        let recip = if delta == 0. { 0. } else { delta.recip() };
        Self {
            delta: f16::from_f32(delta),
            quants: data.map(|x| (x * recip).round() as _),
        }
    }

    #[inline]
    fn dequantize(&self) -> [f32; _32] {
        let delta = self.delta.to_f32();
        self.quants.map(|x| x as f32 * delta)
    }
}
