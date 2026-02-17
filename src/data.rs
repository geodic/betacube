use crate::cube::cube_to_tensor;
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    tensor::{Int, Shape, Tensor, TensorData, backend::Backend},
};
use erno::{Axis, Cube, Direction, Move, ReferenceFrame};
use rand::prelude::*;
use std::collections::HashSet;

pub struct CubeDataset {
    data: Vec<(Cube, Move)>,
    size: usize,
}

impl CubeDataset {
    pub fn new(cube_size: usize, scramble_length: usize, requested_size: usize) -> Self {
        let mut data = Vec::with_capacity(requested_size);
        let mut seen_states = std::collections::HashSet::new();

        let mut attempts = 0;
        let max_attempts = requested_size * 5; // Allow some retries but prevent infinite loops

        while data.len() < requested_size && attempts < max_attempts {
            let sample = generate_training_sample(cube_size, scramble_length);
            let state_repr = format!("{:?}", sample.0.faces); // Use Debug rep for hashing

            if !seen_states.contains(&state_repr) {
                seen_states.insert(state_repr);
                data.push(sample);
            }
            attempts += 1;
        }

        if data.len() < requested_size {
            println!(
                "Warning: Only generated {} unique samples out of {} requested (saturation). Cycling data.",
                data.len(),
                requested_size
            );
        }

        CubeDataset {
            data,
            size: requested_size,
        }
    }
}

impl Dataset<(Cube, Move)> for CubeDataset {
    fn get(&self, index: usize) -> Option<(Cube, Move)> {
        if self.data.is_empty() {
            return None;
        }
        self.data.get(index % self.data.len()).cloned()
    }

    fn len(&self) -> usize {
        self.size
    }
}

pub struct CubeBatcher<B: Backend> {
    pub _b: std::marker::PhantomData<B>,
}

impl<B: Backend> CubeBatcher<B> {
    pub fn new() -> Self {
        Self {
            _b: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> Default for CubeBatcher<B> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub struct CubeBatch<B: Backend> {
    pub cubes: Tensor<B, 3>,
    pub axis_targets: Tensor<B, 1, Int>,
    pub layer_targets: Tensor<B, 1, Int>,
    pub depth_targets: Tensor<B, 1, Int>,
    pub direction_targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<B, (Cube, Move), CubeBatch<B>> for CubeBatcher<B> {
    fn batch(&self, items: Vec<(Cube, Move)>, device: &B::Device) -> CubeBatch<B> {
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

pub fn generate_training_sample(cube_size: usize, scramble_length: usize) -> (Cube, Move) {
    let mut rng = rand::rng();
    let mut cube = Cube::new(cube_size);
    let mut current_axis: Option<Axis> = None;
    let mut used_layers_on_axis: HashSet<usize> = HashSet::new();

    for _ in 0..scramble_length {
        let mut valid_move = None;

        while valid_move.is_none() {
            let axis = match rng.random_range(0..3) {
                0 => Axis::X,
                1 => Axis::Y,
                _ => Axis::Z,
            };

            let max_depth = cube_size / 2;
            let depth = if max_depth > 1 {
                rng.random_range(1..=max_depth)
            } else {
                1
            };

            let mut layer = rng.random_range(0..=(cube_size - depth));

            if cube_size % 2 == 0 && depth == cube_size / 2 {
                layer = cube_size / 2;
            }

            let direction = match rng.random_range(0..3) {
                0 => Direction::Clockwise,
                1 => Direction::CounterClockwise,
                _ => Direction::Double,
            };

            let is_valid = if let Some(last) = current_axis {
                if axis == last {
                    let disjoint =
                        !(layer..layer + depth).any(|l| used_layers_on_axis.contains(&l));

                    let ordered = if let Some(&min_used) = used_layers_on_axis.iter().min() {
                        layer < min_used
                    } else {
                        true
                    };

                    disjoint && ordered
                } else {
                    true
                }
            } else {
                true
            };

            if is_valid {
                let candidate = Move::new(
                    axis,
                    layer,
                    depth,
                    direction,
                    ReferenceFrame::Relative,
                    cube_size,
                );

                valid_move = Some(candidate);

                if Some(axis) != current_axis {
                    current_axis = Some(axis);
                    used_layers_on_axis.clear();
                }
                for l in layer..layer + depth {
                    used_layers_on_axis.insert(l);
                }
            }
        }

        let _ = cube.apply_move(valid_move.unwrap());
    }

    let applied_move = cube.history.last().unwrap().clone();
    let normalized_move = applied_move.to_frame(ReferenceFrame::Normalized).unwrap();

    (cube.normalized(), normalized_move)
}
