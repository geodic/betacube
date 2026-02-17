#![recursion_limit = "256"]

use betacube::{
    data::{CubeBatcher, CubeDataset},
    model::{CubeGNN, CubeGNNConfig},
};
use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    module::Module,
    nn::loss::{CrossEntropyLossConfig, MseLoss, Reduction},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::ToElement,
    record::{CompactRecorder, Recorder},
    tensor::{Shape, Tensor, TensorData, backend::AutodiffBackend},
};

#[derive(Config, Debug)]
pub struct TrainingConfig {
    #[config(default = 2)]
    pub cube_size: usize,
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 1e-3)]
    pub learning_rate: f64,
    #[config(default = 0.95)]
    pub mastery_threshold: f64,
    #[config(default = 100)]
    pub max_epochs_per_level: usize,
    #[config(default = 100)]
    pub batches_per_epoch: usize,
}

fn run_training<B: AutodiffBackend>(device: B::Device) {
    let config = TrainingConfig::new();

    println!(
        "Initializing model for {}x{} cube...",
        config.cube_size, config.cube_size
    );

    let mut model: CubeGNN<B> = CubeGNNConfig {
        cube_size: config.cube_size,
        num_layers: 4,
        d_hidden: 1024,
    }
    .init(&device);

    let mut optim = AdamConfig::new().init();

    let max_length = if config.cube_size == 2 { 11 } else { 20 };

    let loss_ce = CrossEntropyLossConfig::new().init(&device);
    let loss_mse = MseLoss::new();

    for length in 1..=max_length {
        println!(
            "\n=== Starting Level {} (Scramble Length: {}) ===",
            length, length
        );

        let dataset_size = config.batches_per_epoch * config.batch_size;

        for epoch in 1..=config.max_epochs_per_level {
            let mut total_loss = Tensor::from_floats([0.0], &device);
            let mut total_axis_loss = Tensor::from_floats([0.0], &device);
            let mut total_layer_loss = Tensor::from_floats([0.0], &device);
            let mut total_depth_loss = Tensor::from_floats([0.0], &device);
            let mut total_dir_loss = Tensor::from_floats([0.0], &device);
            let mut total_value_loss = Tensor::from_floats([0.0], &device);
            let mut correct_preds = Tensor::from_floats([0.0], &device);

            let dataset = CubeDataset::new(config.cube_size, length, dataset_size);

            let batcher = CubeBatcher::<B>::new();

            let dataloader = DataLoaderBuilder::new(batcher)
                .batch_size(config.batch_size)
                .num_workers(1)
                .build(dataset);

            for batch in dataloader.iter() {
                let current_value = length as f32;
                let value_targets = Tensor::from_data(
                    TensorData::new(
                        vec![current_value; config.batch_size],
                        Shape::new([config.batch_size, 1]),
                    ),
                    &device,
                );

                let inputs = batch.cubes;
                let output = model.forward(inputs);

                let loss_axis = loss_ce.forward(output.axis.clone(), batch.axis_targets.clone());
                let loss_layer = loss_ce.forward(output.slice.clone(), batch.layer_targets.clone());
                let loss_depth = loss_ce.forward(output.depth.clone(), batch.depth_targets.clone());
                let loss_dir =
                    loss_ce.forward(output.direction.clone(), batch.direction_targets.clone());

                let loss_val =
                    loss_mse.forward(output.value.clone(), value_targets, Reduction::Mean);

                let loss = loss_axis.clone()
                    + loss_layer.clone()
                    + loss_depth.clone()
                    + loss_dir.clone()
                    + loss_val.clone();

                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &model);
                model = optim.step(config.learning_rate, model, grads);

                total_loss = total_loss + loss;
                total_axis_loss = total_axis_loss + loss_axis;
                total_layer_loss = total_layer_loss + loss_layer;
                total_depth_loss = total_depth_loss + loss_depth;
                total_dir_loss = total_dir_loss + loss_dir;
                total_value_loss = total_value_loss + loss_val;

                let pred_axis = output.axis.argmax(1).squeeze::<1>();
                let pred_layer = output.slice.argmax(1).squeeze::<1>();
                let pred_depth = output.depth.argmax(1).squeeze::<1>();
                let pred_dir = output.direction.argmax(1).squeeze::<1>();

                let mask = pred_axis
                    .equal(batch.axis_targets)
                    .bool_and(pred_layer.equal(batch.layer_targets))
                    .bool_and(pred_depth.equal(batch.depth_targets))
                    .bool_and(pred_dir.equal(batch.direction_targets));

                correct_preds = correct_preds + mask.float().sum();
            }

            let avg_loss = total_loss.into_scalar().to_f64() / config.batches_per_epoch as f64;
            let avg_axis_loss =
                total_axis_loss.into_scalar().to_f64() / config.batches_per_epoch as f64;
            let avg_layer_loss =
                total_layer_loss.into_scalar().to_f64() / config.batches_per_epoch as f64;
            let avg_depth_loss =
                total_depth_loss.into_scalar().to_f64() / config.batches_per_epoch as f64;
            let avg_dir_loss =
                total_dir_loss.into_scalar().to_f64() / config.batches_per_epoch as f64;
            let avg_value_loss =
                total_value_loss.into_scalar().to_f64() / config.batches_per_epoch as f64;
            let accuracy = correct_preds.into_scalar().to_f64()
                / (config.batches_per_epoch * config.batch_size) as f64;

            println!(
                "Epoch {}: Total {:.4} | Axis {:.4} | Layer {:.4} | Depth {:.4} | Dir {:.4} | Val {:.4} | Acc {:.2}%",
                epoch,
                avg_loss,
                avg_axis_loss,
                avg_layer_loss,
                avg_depth_loss,
                avg_dir_loss,
                avg_value_loss,
                accuracy * 100.0
            );

            if accuracy >= config.mastery_threshold {
                println!(">> Level {} Mastered! Advancing...", length);
                break;
            }
        }

        println!("Saving model...");
        let recorder = CompactRecorder::new();
        recorder
            .record(
                model.clone().into_record(),
                format!("model_level_{}", length).into(),
            )
            .expect("Failed to save model");
    }
}

fn main() {
    type MyBackend = burn::backend::Wgpu;
    type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::DefaultDevice;
    run_training::<MyAutodiffBackend>(device);
}
