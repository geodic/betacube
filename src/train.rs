use crate::data::{Batch, Data};
use crate::model::RubiksModel;
use anyhow::Result;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, GradientsParams};
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::{
    optim::{AdamConfig, Optimizer},
    prelude::*,
    tensor::backend::AutodiffBackend,
};

#[derive(Config)]
pub struct TrainConfig {
    #[config(default = 10)]
    epoch_init: usize,
    #[config(default = 3)]
    epoch_mult: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 1.5)]
    pub target_accuracy_decay: f64,
    #[config(default = 5)]
    pub accuracy_consistency: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1e-4)]
    pub learning_rate: f64,
}

fn compute_accuracy<B: AutodiffBackend>(
    prediction: Tensor<B, 2>,
    truth: Tensor<B, 1, Int>,
) -> Option<f64> {
    let total = prediction.dims()[0];
    let correct = prediction
        .argmax(1)
        .squeeze::<1>(1)
        .equal(truth)
        .int()
        .to_data()
        .iter::<i64>()
        .reduce(|acc, e| acc + e);

    match correct {
        Some(correct) => Some(correct as f64 / total as f64),
        None => None,
    }
}

fn train_batch<B: AutodiffBackend>(
    config: &TrainConfig,
    batch: Batch<B>,
    model: RubiksModel<B>,
    optimizer: &mut OptimizerAdaptor<Adam, RubiksModel<B>, B>,
) -> Result<(Tensor<B, 1>, Option<f64>, RubiksModel<B>)> {
    let output = model.forward(batch.states.clone());
    let loss = CrossEntropyLossConfig::new()
        .with_logits(false)
        .init(&output.device())
        .forward(output.clone(), batch.scrambles.clone());
    let accuracy = compute_accuracy(output.clone(), batch.scrambles.clone());

    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, &model);

    let model = optimizer.step(config.learning_rate, model, grads);

    Ok((loss, accuracy, model))
}

pub fn train<B: AutodiffBackend>(device: &B::Device, config: &TrainConfig) -> Result<()> {
    let mut model = RubiksModel::<B>::new(device);
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let mut optimizer = AdamConfig::new().init();

    B::seed(config.seed);

    for target_moves in 1..=20 {
        println!("Training for {} moves", target_moves);
        for scramble_moves in 1..=target_moves {
            println!("Training with {} scramble moves", scramble_moves);
            let target_accuracy = 100.0 - config.target_accuracy_decay * scramble_moves as f64;

            let mut dataset = Data::new(device, config.batch_size, scramble_moves);

            println!("Starting warmup run...");
            for warmup in 1..3 {
                let batch = dataset
                    .next()
                    .expect("Failed to get batch from dataset during warmup");

                let (loss, accuracy, new_model) =
                    train_batch(config, batch, model.clone(), &mut optimizer)?;
                let accuracy = accuracy.map_or(0.0, |acc| acc * 100.0);
                model = new_model;

                println!(
                    "Warmup {}: Loss: {:.4}, Accuracy: {:.2}%",
                    warmup,
                    loss.clone().into_scalar(),
                    accuracy
                );
            }

            println!("Starting training loop...");
            let mut iteration = 0;
            let mut consistent_accuracy = 0;
            loop {
                let batch = dataset
                    .next()
                    .expect("Failed to get batch from dataset during training");

                let (loss, accuracy, new_model) =
                    train_batch(config, batch, model.clone(), &mut optimizer)?;
                let accuracy = accuracy.map_or(0.0, |acc| acc * 100.0);
                model = new_model;

                iteration += 1;

                if iteration % 10 == 0 {
                    println!(
                        "Iteration {}: Loss: {:.4}, Accuracy: {:.2}%",
                        iteration,
                        loss.clone().into_scalar(),
                        accuracy
                    );
                }

                if accuracy >= target_accuracy {
                    consistent_accuracy += 1;
                } else {
                    consistent_accuracy = 0;
                }

                if consistent_accuracy >= config.accuracy_consistency {
                    println!(
                        "Achieved target accuracy of {:.2}% with consistency of {} and {} moves",
                        accuracy, consistent_accuracy, scramble_moves
                    );
                    break;
                } else {
                    // println!(
                    //     "Accuracy {:.2}% not sufficient, retrying with {} scramble moves",
                    //     accuracy, scramble_moves
                    // );
                }
            }
        }
        println!("Saving model for {} moves", target_moves);
        let model_path = format!("model_{}moves.mpk", target_moves);
        model.clone().save_file(&model_path, &recorder)?;
    }

    Ok(())
}
