mod cube;
mod data;
mod model;
mod train;

use crate::train::{TrainConfig, train};
use anyhow::Result;
use burn::backend::{Autodiff, Wgpu};

type AutodiffBackend = Autodiff<Wgpu>;
fn main() -> Result<()> {
    let device = burn::backend::wgpu::WgpuDevice::DefaultDevice;
    let config = TrainConfig::new();

    train::<AutodiffBackend>(&device, &config)?;
    Ok(())
}
