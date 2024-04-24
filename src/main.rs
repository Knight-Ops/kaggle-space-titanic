use burn::backend::{Autodiff, LibTorch};

mod data;
mod dataset;
mod model;
mod training;

use training::run;

fn main() {
    // type WgpuBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type TorchBackend = LibTorch<f32>;

    // type AutodiffWgpu = Autodiff<WgpuBackend>;
    type AutodiffTorch = Autodiff<TorchBackend>;

    // let device = burn::backend::wgpu::WgpuDevice::default();
    // let cuda_device = burn::backend::wgpu::WgpuDevice::IntegratedGpu(0);
    // println!("Cuda : {:?}", cuda_device);
    // train::<AutodiffWgpu>("/tmp/burn", TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()), cuda_device)
    let device = burn::backend::libtorch::LibTorchDevice::Cpu;
    let cuda_device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
    // println!("{:?}", cuda_device);
    run::<AutodiffTorch>(cuda_device)
}
