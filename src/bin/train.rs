#![recursion_limit = "256"]

use betacube::{
    cube::cube_to_tensor,
    data::generate_training_sample,
    model::{CubeGNN, CubeGNNConfig},
};
use burn::{
    config::Config,
    data::{
        dataloader::{DataLoaderBuilder, batcher::Batcher},
        dataset::Dataset,
    },
    module::Module,
    nn::loss::{CrossEntropyLossConfig, MseLoss, Reduction},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::ToElement,
    record::{CompactRecorder, Recorder},
    tensor::{
        Bool, Int, Shape, Tensor, TensorData,
        backend::{AutodiffBackend, Backend},
    },
};
use erno::{Axis, Direction};

#[derive(Config, Debug)]
pub struct TrainingConfig {
    #[config(default = 2)]
    pub cube_size: usize,
    #[config(default = 2)]
    pub batch_size: usize,
    #[config(default = 1e-3)]
    pub learning_rate: f64,
    #[config(default = 0.95)]
    pub mastery_threshold: f64,
    #[config(default = 100)]
    pub max_epochs_per_level: usize,
    #[config(default = 1)]
    pub batches_per_epoch: usize,
}

struct CubeDataset {
    cube_size: usize,
    scramble_length: usize,
    size: usize,
}

impl Dataset<(erno::Cube, erno::Move)> for CubeDataset {
    fn get(&self, _index: usize) -> Option<(erno::Cube, erno::Move)> {
        Some(generate_training_sample(
            self.cube_size,
            self.scramble_length,
        ))
    }

    fn len(&self) -> usize {
        self.size
    }
}

struct CubeBatcher<B: Backend> {
    _b: std::marker::PhantomData<B>,
}

#[derive(Clone, Debug)]
struct CubeBatch<B: Backend> {
    cubes: Tensor<B, 3>,
    axis_targets: Tensor<B, 1, Int>,
    layer_targets: Tensor<B, 1, Int>,
    depth_targets: Tensor<B, 1, Int>,
    direction_targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<B, (erno::Cube, erno::Move), CubeBatch<B>> for CubeBatcher<B> {
    fn batch(&self, items: Vec<(erno::Cube, erno::Move)>, device: &B::Device) -> CubeBatch<B> {
        let cubes: Vec<Tensor<B, 2>> = items
            .iter()
            .map(|(cube, _)| cube_to_tensor(cube, device))
            .collect();

        let cubes = Tensor::stack(cubes, 0);

        let mut axis_targets = Vec::new();
        let mut layer_targets = Vec::new();
        let mut depth_targets = Vec::new();
        let mut direction_targets = Vec::new();

        for (_, m) in items {
            let axis_idx = match m.axis {
                Axis::X => 0,
                Axis::Y => 1,
                Axis::Z => 2,
            };

            let dir_idx = match m.direction {
                Direction::Clockwise => 1,
                Direction::CounterClockwise => 0,
                Direction::Double => 2,
            };

            let depth_idx = (m.depth as i64) - 1;

            axis_targets.push(axis_idx as i64);
            layer_targets.push(m.start_layer as i64);
            depth_targets.push(depth_idx);
            direction_targets.push(dir_idx as i64);
        }

        let n = axis_targets.len();

        let axis_targets =
            Tensor::from_data(TensorData::new(axis_targets, Shape::new([n])), device);
        let layer_targets =
            Tensor::from_data(TensorData::new(layer_targets, Shape::new([n])), device);
        let depth_targets =
            Tensor::from_data(TensorData::new(depth_targets, Shape::new([n])), device);
        let direction_targets =
            Tensor::from_data(TensorData::new(direction_targets, Shape::new([n])), device);

        CubeBatch {
            cubes,
            axis_targets,
            layer_targets,
            depth_targets,
            direction_targets,
        }
    }
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
        d_hidden: 128,
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

        // Define dataset size for this epoch
        let dataset_size = config.batches_per_epoch * config.batch_size;

        for epoch in 1..=config.max_epochs_per_level {
            let mut total_loss = Tensor::from_floats([0.0], &device);
            let mut correct_preds = Tensor::from_floats([0.0], &device);

            let dataset = CubeDataset {
                cube_size: config.cube_size,
                scramble_length: length,
                size: dataset_size,
            };

            let batcher = CubeBatcher::<B> {
                _b: std::marker::PhantomData,
            };

            let dataloader = DataLoaderBuilder::new(batcher)
                .batch_size(config.batch_size)
                //.shuffle(42) // Random generation makes shuffling redundant
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

                let loss = loss_axis + loss_layer + loss_depth + loss_dir + loss_val;

                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &model);
                model = optim.step(config.learning_rate, model, grads);

                total_loss = total_loss + loss;

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
            let accuracy = correct_preds.into_scalar().to_f64()
                / (config.batches_per_epoch * config.batch_size) as f64;

            println!(
                "Epoch {}: Loss = {:.4}, Accuracy = {:.2}%",
                epoch,
                avg_loss,
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
